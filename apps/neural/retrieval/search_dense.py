"""Dense retrieval over saved paragraph embeddings -> TREC run files.

Writes both levels from one search:
  paragraph run: top-100 paragraphs
  paper run:     each paper scored by its best paragraph (max), top-100 papers
Papers are formed from the top --depth paragraphs so 100 distinct papers exist.

Names follow the submission spec: <team>_<level>_dense_<set>.tsv

    python -m apps.neural.retrieval.search_dense --emb data/neural/dense_zeroshot --set study
"""
import argparse
import glob
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..common import ROOT, load_queries, paper_of


def load_embeddings(emb_dir):
    shards = sorted(glob.glob(os.path.join(emb_dir, "shard_*.npy")))
    emb = np.concatenate([np.load(s) for s in shards])
    with open(os.path.join(emb_dir, "ids.txt"), encoding="utf-8") as f:
        ids = f.read().split("\n")
    assert len(ids) == len(emb), "ids/embeddings mismatch: %d vs %d" % (len(ids), len(emb))
    return ids, emb


def write_run(path, rows, run_name):
    """rows: {qid: [(docid, score), ...]} already sorted best first."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, docs in rows.items():
            for rank, (doc, score) in enumerate(docs, 1):
                f.write("%s\tQ0\t%s\t%d\t%.6f\t%s\n" % (qid, doc, rank, score, run_name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="data/neural/dense_zeroshot")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--set", default="study", choices=["train", "study", "test"])
    p.add_argument("--team", default="JackSearchParty")
    p.add_argument("--run_tag", default="dense")
    p.add_argument("--out_dir", default="data/runs")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--depth", type=int, default=1000)
    p.add_argument("--max_seq_length", type=int, default=256)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids, emb = load_embeddings(os.path.join(ROOT, args.emb) if not os.path.isabs(args.emb) else args.emb)
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length
    qids, qtexts = load_queries(args.set)
    qemb = model.encode(qtexts, batch_size=64, normalize_embeddings=True, convert_to_tensor=True,
                        show_progress_bar=False)

    dtype = torch.float16 if device == "cuda" else torch.float32
    docs = torch.from_numpy(emb).to(device=device, dtype=dtype)
    para_rows, paper_rows = {}, {}
    for i, qid in enumerate(qids):
        scores = docs @ qemb[i].to(dtype)
        top_s, top_i = torch.topk(scores, k=args.depth)
        top_s, top_i = top_s.float().cpu().tolist(), top_i.cpu().tolist()
        para_rows[qid] = [(ids[j], s) for j, s in zip(top_i[: args.k], top_s[: args.k])]
        best = {}
        for j, s in zip(top_i, top_s):  # already sorted: first hit per paper is its max
            best.setdefault(paper_of(ids[j]), s)
        paper_rows[qid] = list(best.items())[: args.k]

    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out, exist_ok=True)
    for level, rows in (("paragraph", para_rows), ("paper", paper_rows)):
        path = os.path.join(out, "%s_%s_%s_%s.tsv" % (args.team, level, args.run_tag, args.set))
        write_run(path, rows, "%s_%s" % (args.team, args.run_tag))
        print("wrote", path)


if __name__ == "__main__":
    main()
