"""Fine-tune the dense bi-encoder on TRAIN queries only (contrastive loss).

Recipe follows the course sample (fine_tune_sbert.py): split train queries
90/10 by query, build (query, relevant paragraph) pairs from the train qrels,
train with MultipleNegativesRankingLoss (other paragraphs in the batch are the
negatives; the no-duplicates sampler keeps a query/paragraph out of a batch
twice, which would create false negatives). Train and validation loss are
logged every --eval_steps and written to a CSV for the loss figure.

    python -m apps.neural.training.train_dense --out models/dense_ft
"""
import argparse
import csv
import json
import os
import random

import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from ..common import DATA, PARAGRAPHS_TSV, ROOT, load_queries


def positives_by_query(path):
    pos = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, pid, rel = line.split()[:4]
            if int(rel) > 0:  # Relevant (2) and Partially Relevant (1)
                pos.setdefault(qid, []).append(pid)
    return pos


def paragraph_texts(needed):
    """Stream the extracted corpus once, keeping only the needed paragraph ids."""
    found = {}
    with open(PARAGRAPHS_TSV, encoding="utf-8") as f:
        for line in f:
            pid, _, text = line.rstrip("\n").partition("\t")
            if pid in needed:
                found[pid] = text
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="models/dense_ft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit_queries", type=int, default=0, help="smoke test: use only N train queries")
    args = p.parse_args()

    random.seed(args.seed)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

    train_ids, train_text = load_queries("train")
    qtext = dict(zip(train_ids, train_text))
    pos = positives_by_query(os.path.join(DATA, "qrels", "Train_paragraph_qrel.tsv"))
    usable = [q for q in train_ids if q in pos]
    if args.limit_queries:
        usable = usable[: args.limit_queries]

    # Leakage guard: Study/Test query ids must never appear in training.
    held_out = set(load_queries("study")[0]) | set(load_queries("test")[0])
    assert not (set(usable) & held_out), "Study/Test query id found in training set"

    random.shuffle(usable)
    n_val = max(1, int(len(usable) * args.val_ratio))
    val_q, train_q = usable[:n_val], usable[n_val:]
    assert not (set(train_q) & set(val_q))
    print("train queries: %d | validation queries: %d" % (len(train_q), len(val_q)))

    needed = {pid for q in usable for pid in pos[q]}
    text = paragraph_texts(needed)

    def pairs(qs):
        rows = [(qtext[q], text[pid]) for q in qs for pid in pos[q] if pid in text]
        return Dataset.from_dict({"anchor": [a for a, _ in rows], "positive": [b for _, b in rows]})

    train_ds, val_ds = pairs(train_q), pairs(val_q)
    print("train pairs: %d | validation pairs: %d" % (len(train_ds), len(val_ds)))

    model = SentenceTransformer(args.base_model, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = args.max_seq_length
    loss = MultipleNegativesRankingLoss(model)

    targs = SentenceTransformerTrainingArguments(
        output_dir=out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.eval_steps,
        seed=args.seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train_ds,
                                         eval_dataset=val_ds, loss=loss)
    trainer.train()
    model.save_pretrained(out)

    rows = {}
    for e in trainer.state.log_history:
        step = e.get("step")
        if step is None:
            continue
        r = rows.setdefault(step, {"step": step, "epoch": e.get("epoch")})
        if "loss" in e:
            r["train_loss"] = e["loss"]
        if "eval_loss" in e:
            r["val_loss"] = e["eval_loss"]
    csv_path = os.path.join(out, "loss_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "val_loss"])
        w.writeheader()
        for step in sorted(rows):
            w.writerow(rows[step])
    with open(os.path.join(out, "split.json"), "w") as f:
        json.dump({"train": sorted(train_q, key=int), "val": sorted(val_q, key=int)}, f)
    print("saved model + loss_log.csv + split.json in", out)


if __name__ == "__main__":
    main()
