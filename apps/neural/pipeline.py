"""Run the Part 2 (neural retrieval) experiments, one stage at a time or all in order.

    bash bin/run.sh --neural                 # all stages (several hours, needs a CUDA GPU)
    bash bin/run.sh --neural train encode    # only some stages
    bash bin/run.sh --neural --dry-run       # print the commands without running them

Stages, in order (names are what you pass on the command line):
    baseline    Part 1 BM25 runs on Study/Test (apps.evaluation.evaluate); needs bin/install.sh first
    paragraphs  extract the cleaned paragraph corpus  -> data/neural/paragraphs.tsv
    train       fine-tune dense and SPLADE on Train queries -> models/dense_ft, models/splade_ft
    encode      encode all paragraphs with the fine-tuned models -> data/neural/dense_ft, splade_ft
    search      write the 8 run files for Study and Test -> data/runs/JackSearchParty_*.tsv
    zeroshot    encode and search with the pretrained models (the "no fine-tuning" comparison)
    evaluate    metrics, significance tests and the Easy/Medium/Hard breakdown -> data/evaluation/
    figures     loss curves and the nDCG@5 bar chart -> docs/figures/
    package     copy the run files and qrel into 'Retrieval Results/' and check their format

Test-query relevance judgments (the *_qrel_part2_test.tsv file) are made by hand with
apps/neural/reporting/build_annotation.py and are not a pipeline stage.
"""
import argparse
import subprocess
import sys

PY = sys.executable
LEVELS = ("paper", "paragraph")


def m(module, *args):
    return [PY, "-m", "apps.neural." + module, *args]


def stage_commands():
    cmds = {}
    cmds["baseline"] = [[PY, "-m", "apps.evaluation.evaluate"]]
    cmds["paragraphs"] = [m("corpus.extract_paragraphs")]
    cmds["train"] = [
        m("training.train_dense", "--out", "models/dense_ft"),
        m("training.train_splade", "--out", "models/splade_ft"),  # reuses the dense train/validation split
    ]
    cmds["encode"] = [
        m("indexing.encode_dense", "--model", "models/dense_ft", "--out", "data/neural/dense_ft"),
        m("indexing.encode_splade", "--model", "models/splade_ft", "--out", "data/neural/splade_ft", "--max_length", "128"),
    ]
    cmds["search"] = [
        m("retrieval.search_dense", "--emb", "data/neural/dense_ft", "--model", "models/dense_ft", "--set", s)
        for s in ("study", "test")
    ] + [
        m("retrieval.search_splade", "--emb", "data/neural/splade_ft", "--model", "models/splade_ft", "--set", s)
        for s in ("study", "test")
    ]
    cmds["zeroshot"] = [
        m("indexing.encode_dense", "--out", "data/neural/dense_zeroshot"),
        m("indexing.encode_splade", "--out", "data/neural/splade_zeroshot", "--max_length", "128"),
        m("retrieval.search_dense", "--emb", "data/neural/dense_zeroshot", "--set", "study",
          "--team", "TheSearchParty", "--run_tag", "dense_zs"),
        m("retrieval.search_splade", "--emb", "data/neural/splade_zeroshot", "--set", "study",
          "--team", "TheSearchParty", "--run_tag", "sparse_zs"),
    ]
    ev = []
    for level in LEVELS:
        runs = [
            "bm25=data/runs/TheSearchParty_%s_bm25_study.tsv" % level,
            "dense_zs=data/runs/TheSearchParty_%s_dense_zs_study.tsv" % level,
            "dense=data/runs/JackSearchParty_%s_dense_study.tsv" % level,
            "sparse_zs=data/runs/TheSearchParty_%s_sparse_zs_study.tsv" % level,
            "sparse=data/runs/JackSearchParty_%s_sparse_study.tsv" % level,
        ]
        ev.append(m("evaluation.significance", "--level", level,
                    "--out", "data/evaluation/part2_%s_significance_final.json" % level, *runs))
        ev.append(m("evaluation.by_difficulty", "--level", level,
                    "--out", "data/evaluation/part2_%s_by_difficulty.json" % level))
    cmds["evaluate"] = ev
    cmds["figures"] = [
        m("reporting.plot_loss", "models/dense_ft/loss_log.csv", "docs/figures/loss_dense.png",
          "--title", "Dense (MiniLM) fine-tuning"),
        m("reporting.plot_loss", "models/splade_ft/loss_log.csv", "docs/figures/loss_splade.png",
          "--title", "SPLADE fine-tuning"),
        m("reporting.plot_results", "docs/figures/results_ndcg5.png"),
    ]
    cmds["package"] = [m("evaluation.package_results")]
    return cmds


ORDER = ["baseline", "paragraphs", "train", "encode", "search", "zeroshot", "evaluate", "figures", "package"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="*", help="stages to run (default: all, in order)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = p.parse_args()

    cmds = stage_commands()
    bad = [s for s in a.stages if s not in cmds]
    if bad:
        p.error("unknown stage(s): %s (choose from: %s)" % (", ".join(bad), ", ".join(ORDER)))
    todo = [s for s in ORDER if not a.stages or s in a.stages]

    for stage in todo:
        print("\n=== stage: %s ===" % stage, flush=True)
        for cmd in cmds[stage]:
            print("$", " ".join(cmd), flush=True)
            if a.dry_run:
                continue
            if subprocess.call(cmd) != 0:
                sys.exit("stage '%s' failed on: %s" % (stage, " ".join(cmd)))
    print("\ndone")


if __name__ == "__main__":
    main()
