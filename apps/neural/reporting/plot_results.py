"""Bar chart of nDCG@5 (BM25 vs dense vs SPLADE) at paper and paragraph level.

    python -m apps.neural.reporting.plot_results docs/figures/results_ndcg5.png

Means come from score.py on the Study run files. A star above a bar means the paired t-test
against BM25 gave p < 0.05 (p-values read from data/evaluation/part2_*_significance_final.json).
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import DATA
from ..evaluation.score import load_qrels, load_run
from ranx import evaluate

RUNS = {  # label -> (run file pattern, key used in the significance json)
    "BM25": ("TheSearchParty_{level}_bm25_study.tsv", None),
    "Dense (fine-tuned)": ("JackSearchParty_{level}_dense_study.tsv", "dense"),
    "SPLADE (fine-tuned)": ("JackSearchParty_{level}_sparse_study.tsv", "sparse"),
}
COLORS = ["#8a93a6", "#2b5fd9", "#e08a1e"]


def p_vs_bm25(level, key):
    path = os.path.join(DATA, "evaluation", "part2_%s_significance_final.json" % level)
    with open(path, encoding="utf-8") as f:
        sig = json.load(f)
    return sig["relevant_or_partial | bm25 vs %s | ndcg@5" % key]["p"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    a = p.parse_args()

    values, pvals = {}, {}
    for level in ("paper", "paragraph"):
        qrels = load_qrels(level)
        for name, (pattern, key) in RUNS.items():
            run = load_run(os.path.join(DATA, "runs", pattern.format(level=level)))
            values[(level, name)] = evaluate(qrels, run, "ndcg@5")
            pvals[(level, name)] = p_vs_bm25(level, key) if key else None

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    width = 0.26
    for i, name in enumerate(RUNS):
        xs = [j + (i - 1) * width for j in range(2)]
        ys = [values[(lv, name)] for lv in ("paper", "paragraph")]
        bars = ax.bar(xs, ys, width, label=name, color=COLORS[i])
        for b, lv in zip(bars, ("paper", "paragraph")):
            star = "*" if (pvals[(lv, name)] is not None and pvals[(lv, name)] < 0.05) else ""
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, "%.3f%s" % (b.get_height(), star),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Paper level", "Paragraph level"])
    ax.set_ylabel("nDCG@5")
    ax.set_ylim(0, 0.75)
    ax.set_title("nDCG@5 on the Study queries")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.01, 0.01, "* significantly different from BM25 (paired t-test, p < 0.05)", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    for k, v in values.items():
        print(k, round(v, 4), "p vs BM25:", pvals[k])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
