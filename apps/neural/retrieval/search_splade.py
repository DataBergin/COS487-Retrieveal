"""Sparse retrieval over SPLADE shards -> TREC run files (paragraph top-100 + paper max top-100).

Scores a query by a dot product over its non-zero terms against the inverted
structure (a CSC matrix: one column of postings per vocabulary term).

    python -m apps.neural.retrieval.search_splade --emb data/neural/splade_zeroshot --set study --run_tag sparse_zs
"""
import argparse
import glob
import os

import numpy as np
import scipy.sparse as sp

from ..common import ROOT, load_queries, paper_of
from .search_dense import write_run
from ..encoders.splade import DEFAULT_MODEL, Splade


def load_index(emb_dir, vocab_size):
    shards = sorted(glob.glob(os.path.join(emb_dir, "shard_*.npz")))
    mats = []
    for s in shards:
        z = np.load(s)
        mats.append(sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]),
                                  shape=(len(z["indptr"]) - 1, vocab_size)))
    d = sp.vstack(mats, format="csc")
    with open(os.path.join(emb_dir, "ids.txt"), encoding="utf-8") as f:
        ids = f.read().split("\n")
    assert len(ids) == d.shape[0], "ids/index mismatch: %d vs %d" % (len(ids), d.shape[0])
    return ids, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="data/neural/splade_zeroshot")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--set", default="study", choices=["train", "study", "test"])
    p.add_argument("--team", default="JackSearchParty")
    p.add_argument("--run_tag", default="sparse")
    p.add_argument("--out_dir", default="data/runs")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--depth", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=256)
    args = p.parse_args()

    enc = Splade(args.model, max_length=args.max_length)
    emb = args.emb if os.path.isabs(args.emb) else os.path.join(ROOT, args.emb)
    ids, docs = load_index(emb, enc.vocab_size)
    qids, qtexts = load_queries(args.set)

    para_rows, paper_rows = {}, {}
    for qid, qt in zip(qids, qtexts):
        qv = enc.encode_dense([qt])[0].cpu().numpy()
        nz = np.nonzero(qv >= 0.05)[0]
        scores = docs[:, nz] @ qv[nz]
        top = np.argpartition(-scores, args.depth)[: args.depth]
        top = top[np.argsort(-scores[top])]
        para_rows[qid] = [(ids[j], float(scores[j])) for j in top[: args.k]]
        best = {}
        for j in top:
            best.setdefault(paper_of(ids[j]), float(scores[j]))
        paper_rows[qid] = list(best.items())[: args.k]

    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out, exist_ok=True)
    for level, rows in (("paragraph", para_rows), ("paper", paper_rows)):
        path = os.path.join(out, "%s_%s_%s_%s.tsv" % (args.team, level, args.run_tag, args.set))
        write_run(path, rows, "%s_%s" % (args.team, args.run_tag))
        print("wrote", path)


if __name__ == "__main__":
    main()
