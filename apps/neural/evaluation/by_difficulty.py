"""Per-difficulty results (Easy / Medium / Hard / All) for several runs on the Study queries.

    python -m apps.neural.evaluation.by_difficulty --level paper
    python -m apps.neural.evaluation.by_difficulty --level paragraph --markdown

Columns (same layout as the Part 1 difficulty tables):
    nDCG=0       queries with nothing relevant anywhere in the top 100
    nDCG, nDCG@5 Relevant + Partial counted as relevant
    MRR, P@5     two variants: "any" (Relevant + Partial) and "full" (Relevant only)
"""
import argparse
import json
import os
from collections import defaultdict

from ranx import evaluate

from ..common import DATA
from .score import load_qrels, load_run

RUNS = {
    "BM25": "TheSearchParty_{level}_bm25_study.tsv",
    "Dense (fine-tuned)": "JackSearchParty_{level}_dense_study.tsv",
    "SPLADE (fine-tuned)": "JackSearchParty_{level}_sparse_study.tsv",
}
GROUPS = ("Easy", "Medium", "Hard", "All")
COLUMNS = ["zero", "ndcg", "ndcg@5", "mrr_any", "p5_any", "mrr_full", "p5_full"]


def per_query_scores(run, qrels_any, qrels_full):
    """Per-query value of every metric, as {name: list aligned with run.get_query_ids()}."""
    return {
        "ndcg": evaluate(qrels_any, run, "ndcg", return_mean=False),
        "ndcg@5": evaluate(qrels_any, run, "ndcg@5", return_mean=False),
        "mrr_any": evaluate(qrels_any, run, "mrr", return_mean=False),
        "p5_any": evaluate(qrels_any, run, "precision@5", return_mean=False),
        "mrr_full": evaluate(qrels_full, run, "mrr", return_mean=False),
        "p5_full": evaluate(qrels_full, run, "precision@5", return_mean=False),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=["paper", "paragraph"], required=True)
    p.add_argument("--out", default="", help="write the numbers as JSON")
    p.add_argument("--markdown", action="store_true", help="print Markdown tables for the paper")
    a = p.parse_args()

    label = {str(q["query_id"]): q["label"]
             for q in json.load(open(os.path.join(DATA, "Study.json"), encoding="utf-8"))}
    members = defaultdict(list)
    for q, l in label.items():
        members[l].append(q)
        members["All"].append(q)

    qrels_any = load_qrels(a.level)
    qrels_full = load_qrels(a.level)
    qrels_full.set_relevance_level(2)  # only Relevant (score 2) counts

    result = {}
    for name, pattern in RUNS.items():
        run = load_run(os.path.join(DATA, "runs", pattern.format(level=a.level)))
        per = per_query_scores(run, qrels_any, qrels_full)
        index = {q: i for i, q in enumerate(run.get_query_ids())}
        result[name] = {}
        for group in GROUPS:
            ids = [index[q] for q in members[group] if q in index]
            row = {"n": len(ids), "zero": sum(1 for i in ids if per["ndcg"][i] == 0)}
            for key, values in per.items():
                row[key] = sum(values[i] for i in ids) / len(ids)
            result[name][group] = row

    print("level:", a.level)
    if a.markdown:
        print("\n| Model | Difficulty | Queries | nDCG=0 | nDCG | nDCG@5 | MRR any | P@5 any | MRR full | P@5 full |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for name, groups in result.items():
            for group, r in groups.items():
                print("| %s | %s | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                    name, group, r["n"], r["zero"], r["ndcg"], r["ndcg@5"], r["mrr_any"],
                    r["p5_any"], r["mrr_full"], r["p5_full"]))
    else:
        print("%-21s %-7s %4s %5s %6s %7s %8s %7s %8s %7s" % (
            "system", "diff", "n", "zero", "nDCG", "nDCG@5", "MRR any", "P@5 any", "MRR full", "P@5 full"))
        for name, groups in result.items():
            for group, r in groups.items():
                print("%-21s %-7s %4d %5d %6.3f %7.3f %8.3f %7.3f %8.3f %7.3f" % (
                    name, group, r["n"], r["zero"], r["ndcg"], r["ndcg@5"], r["mrr_any"],
                    r["p5_any"], r["mrr_full"], r["p5_full"]))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
