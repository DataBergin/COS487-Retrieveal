"""Score TREC run files against Study qrels with ranx (same metrics as Part 1).

    python -m apps.neural.evaluation.score data/runs/TheSearchParty_paragraph_dense_study.tsv --level paragraph
"""
import argparse
import os

from ranx import Qrels, Run, evaluate

from ..common import DATA

ANY = ["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"]
FULL = ["mrr", "precision@5"]


def load_run(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            q, _, doc, _, score, _ = line.split()  # tab- or space-separated
            d.setdefault(q, {})[doc] = float(score)
    return Run(d)


def load_qrels(level, split="Study"):
    d = {}
    with open(os.path.join(DATA, "qrels", "%s_%s_qrel.tsv" % (split, level)), encoding="utf-8") as f:
        for line in f:
            q, _, doc, rel = line.split()[:4]
            d.setdefault(q, {})[doc] = int(rel)
    return Qrels(d)


def score(run_path, level):
    run, qrels = load_run(run_path), load_qrels(level)
    out = dict(zip(ANY, [evaluate(qrels, run, m) for m in ANY]))
    qrels.set_relevance_level(2)
    for m in FULL:
        out[m + " (full)"] = evaluate(qrels, run, m)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--level", choices=["paper", "paragraph"], required=True)
    a = p.parse_args()
    for k, v in score(a.run, a.level).items():
        print("%-18s %.3f" % (k, v))
