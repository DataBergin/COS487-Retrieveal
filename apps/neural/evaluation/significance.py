"""Paired significance tests between runs (per-query scores, Student's t-test), via ranx.

Each metric is tested twice, like Part 1: both grades relevant, then Relevant-only
(MRR and Precision@5). For every pair of runs we report each one's mean, the
p-value, and whether the difference is significant at max_p.

    python -m apps.neural.evaluation.significance --level paragraph \
        bm25=data/runs/TheSearchParty_paragraph_bm25_study.tsv \
        dense_zs=data/runs/TheSearchParty_paragraph_dense_zs_study.tsv \
        dense_ft=data/runs/TheSearchParty_paragraph_dense_ft_study.tsv
"""
import argparse
import itertools
import json
import os

import numpy as np
from ranx import compare
from scipy import stats

from .score import ANY, FULL, load_qrels, load_run


def per_query(run, qrels, metric):
    from ranx import evaluate
    return evaluate(qrels, run, metric, return_mean=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="name=path pairs")
    p.add_argument("--level", choices=["paper", "paragraph"], required=True)
    p.add_argument("--max_p", type=float, default=0.05)
    p.add_argument("--out", default="")
    a = p.parse_args()

    named = [r.split("=", 1) for r in a.runs]
    runs = {n: load_run(path) for n, path in named}
    report = {}
    for label, metrics, rel in (("relevant_or_partial", ANY, 1), ("relevant_only", FULL, 2)):
        qrels = load_qrels(a.level)
        qrels.set_relevance_level(rel)
        for m in metrics:
            scores = {n: np.asarray(per_query(r, qrels, m)) for n, r in runs.items()}
            for x, y in itertools.combinations(runs, 2):
                t, pv = stats.ttest_rel(scores[x], scores[y])
                better = None
                if pv < a.max_p:
                    better = x if scores[x].mean() > scores[y].mean() else y
                diff = scores[x] - scores[y]
                report["%s | %s vs %s | %s" % (label, x, y, m)] = {
                    "mean_" + x: round(float(scores[x].mean()), 4),
                    "mean_" + y: round(float(scores[y].mean()), 4),
                    "p": round(float(pv), 5), "better": better,
                    "W": int((diff > 0).sum()), "T": int((diff == 0).sum()), "L": int((diff < 0).sum()),
                }
    for k, v in report.items():
        print("%-62s p=%-8s better=%s  W/T/L=%d/%d/%d" % (k, v["p"], v["better"], v["W"], v["T"], v["L"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
