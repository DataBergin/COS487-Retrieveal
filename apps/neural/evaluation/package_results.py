"""Copy the 8 run files + the qrel into 'Retrieval Results/' and check their format.

    python -m apps.neural.evaluation.package_results --team JackSearchParty

Checks per run file: tab-separated, 6 columns, 'Q0' in column 2, every query of the
split present, ranks 1..n in order, scores non-increasing, no repeated doc per query,
paper ids without '_' and paragraph ids with '_'.
"""
import argparse
import os
import shutil
from collections import defaultdict

from ..common import ROOT

RANGES = {"study": range(2001, 2251), "test": range(2251, 2501)}


def check_run(path, split, level):
    errors = []
    by_q = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 6:
                errors.append("line %d: %d columns (need 6, tab-separated)" % (n, len(cols)))
                break
            qid, q0, doc, rank, score, tag = cols
            if q0 != "Q0":
                errors.append("line %d: column 2 is %r, expected Q0" % (n, q0))
                break
            if ("_" in doc) != (level == "paragraph"):
                errors.append("line %d: doc id %r does not look like a %s id" % (n, doc, level))
                break
            by_q[qid].append((int(rank), float(score), doc))
    want = {str(q) for q in RANGES[split]}
    if set(by_q) != want:
        errors.append("query ids: %d missing, %d unexpected" % (len(want - set(by_q)), len(set(by_q) - want)))
    for q, rows in by_q.items():
        if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
            errors.append("query %s: ranks are not 1..n in order" % q)
            break
        if any(rows[i][1] < rows[i + 1][1] for i in range(len(rows) - 1)):
            errors.append("query %s: scores increase down the list" % q)
            break
        if len({r[2] for r in rows}) != len(rows):
            errors.append("query %s: repeated doc id" % q)
            break
    sizes = {len(v) for v in by_q.values()}
    return errors, len(by_q), sizes


def check_qrel(path):
    errors, n, qs = [], 0, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            n += 1
            if len(cols) != 4 or cols[1] != "0" or cols[3] not in ("0", "1", "2"):
                errors.append("bad qrel line: %r" % line)
                break
            qs.add(cols[0])
    if qs != {str(q) for q in RANGES["test"]}:
        errors.append("qrel does not cover exactly the 250 Test queries")
    return errors, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--team", default="JackSearchParty")
    p.add_argument("--runs", default="data/runs")
    p.add_argument("--out", default="Retrieval Results")
    a = p.parse_args()
    runs = os.path.join(ROOT, a.runs)
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    bad = 0
    for split in ("study", "test"):
        for level in ("paper", "paragraph"):
            for kind in ("dense", "sparse"):
                name = "%s_%s_%s_%s.tsv" % (a.team, level, kind, split)
                src = os.path.join(runs, name)
                if not os.path.exists(src):
                    print("MISSING", name)
                    bad += 1
                    continue
                shutil.copyfile(src, os.path.join(out, name))
                errors, nq, sizes = check_run(os.path.join(out, name), split, level)
                print("%-4s %-45s queries=%d results/query=%s" % ("OK" if not errors else "FAIL", name, nq, sorted(sizes)))
                for e in errors:
                    print("       -", e)
                bad += bool(errors)
    qrel = os.path.join(out, "%s_qrel_part2_test.tsv" % a.team)
    if os.path.exists(qrel):
        errors, n = check_qrel(qrel)
        print("%-4s %-45s rows=%d" % ("OK" if not errors else "FAIL", os.path.basename(qrel), n))
        for e in errors:
            print("       -", e)
        bad += bool(errors)
    else:
        print("MISSING", os.path.basename(qrel))
        bad += 1
    print("\nall good" if not bad else "\n%d problem(s)" % bad)


if __name__ == "__main__":
    main()
