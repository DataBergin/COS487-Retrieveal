"""Fine-tune SPLADE on TRAIN queries only (contrastive loss + sparsity regularizer).

Same train/validation query split as the dense model (read from
models/dense_ft/split.json) so the two models are directly comparable.

Loss = in-batch contrastive cross-entropy over dot products of SPLADE vectors
       + lambda_q * FLOPS(queries) + lambda_d * FLOPS(documents)
FLOPS(x) = sum over vocabulary terms of (mean activation of that term in the batch)^2;
it pushes the representation to stay sparse (SPLADE, Formal et al.). Validation
loss is the contrastive part only, so train/validation curves are comparable.

    python -m apps.neural.training.train_splade --out models/splade_ft
"""
import argparse
import csv
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ..common import DATA, ROOT, load_queries
from ..encoders.splade import DEFAULT_MODEL
from .train_dense import paragraph_texts, positives_by_query


def rep(model, tok, texts, max_len, device):
    b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    logits = model(**b).logits
    w = torch.log1p(torch.relu(logits.float())) * b["attention_mask"].unsqueeze(-1)
    return w.max(dim=1).values


def flops(w):
    return (w.mean(dim=0) ** 2).sum()


def make_batches(pairs, bs, rng):
    """Shuffle pairs; a query never appears twice in a batch (would create false negatives)."""
    pool = pairs[:]
    rng.shuffle(pool)
    batches = []
    while pool:
        batch, seen, rest = [], set(), []
        for p in pool:
            if p[0] not in seen and len(batch) < bs:
                batch.append(p); seen.add(p[0])
            else:
                rest.append(p)
        batches.append(batch)
        pool = rest
    return [b for b in batches if len(b) > 1]


def contrastive(q, d):
    scores = q @ d.T
    return F.cross_entropy(scores, torch.arange(len(q), device=q.device))


@torch.no_grad()
def val_loss(model, tok, batches, a, device):
    model.eval()
    tot, n = 0.0, 0
    for b in batches:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            q = rep(model, tok, [x[1] for x in b], a.max_query_len, device)
            d = rep(model, tok, [x[2] for x in b], a.max_len, device)
        tot += contrastive(q, d).item() * len(b); n += len(b)
    model.train()
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=DEFAULT_MODEL)
    p.add_argument("--split", default="models/dense_ft/split.json")
    p.add_argument("--out", default="models/splade_ft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=128, help="paragraph tokens (matches SPLADE encoding)")
    p.add_argument("--max_query_len", type=int, default=64)
    p.add_argument("--lambda_q", type=float, default=3e-4)
    p.add_argument("--lambda_d", type=float, default=1e-4)
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit_pairs", type=int, default=0, help="smoke test")
    a = p.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    split = json.load(open(a.split if os.path.isabs(a.split) else os.path.join(ROOT, a.split)))
    train_q, val_q = split["train"], split["val"]
    held_out = set(load_queries("study")[0]) | set(load_queries("test")[0])
    assert not ((set(train_q) | set(val_q)) & held_out), "Study/Test query id found in training data"
    assert not (set(train_q) & set(val_q))

    ids, texts = load_queries("train")
    qtext = dict(zip(ids, texts))
    pos = positives_by_query(os.path.join(DATA, "qrels", "Train_paragraph_qrel.tsv"))
    ptext = paragraph_texts({pid for q in train_q + val_q for pid in pos.get(q, [])})

    def pairs(qs):
        return [(q, qtext[q], ptext[pid]) for q in qs for pid in pos.get(q, []) if pid in ptext]

    train_pairs, val_pairs = pairs(train_q), pairs(val_q)
    if a.limit_pairs:
        train_pairs, val_pairs = train_pairs[: a.limit_pairs], val_pairs[: a.limit_pairs // 5]
    print("train pairs: %d | validation pairs: %d" % (len(train_pairs), len(val_pairs)))

    tok = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForMaskedLM.from_pretrained(a.base_model).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    n_steps = a.epochs * len(make_batches(train_pairs, a.batch_size, random.Random(0)))
    warm = max(1, int(0.1 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (n_steps - s) / max(1, n_steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    val_batches = make_batches(val_pairs, a.batch_size, random.Random(1))

    log, best, step, run_loss, run_n = [], float("inf"), 0, 0.0, 0
    for ep in range(a.epochs):
        for b in make_batches(train_pairs, a.batch_size, rng):
            step += 1
            ramp = min(1.0, step / max(1, n_steps // 3))  # regularizer ramp-up, as in SPLADE training
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                q = rep(model, tok, [x[1] for x in b], a.max_query_len, device)
                d = rep(model, tok, [x[2] for x in b], a.max_len, device)
                c = contrastive(q, d)
                loss = c + ramp * (a.lambda_q * flops(q) + a.lambda_d * flops(d))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += c.item(); run_n += 1
            if step % a.eval_steps == 0 or step == n_steps:
                v = val_loss(model, tok, val_batches, a, device)
                row = {"step": step, "epoch": round(step / (n_steps / a.epochs), 3),
                       "train_loss": run_loss / run_n, "val_loss": v}
                log.append(row); run_loss, run_n = 0.0, 0
                print(row, flush=True)
                if v < best:
                    best = v
                    model.save_pretrained(out); tok.save_pretrained(out)
    with open(os.path.join(out, "loss_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "val_loss"])
        w.writeheader(); w.writerows(log)
    print("best validation loss %.4f; saved best checkpoint + loss_log.csv in %s" % (best, out))


if __name__ == "__main__":
    main()
