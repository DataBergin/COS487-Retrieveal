"""Plot training vs validation loss from a loss_log.csv (spec: loss diagram per step/epoch).

    python -m apps.neural.reporting.plot_loss models/dense_ft/loss_log.csv docs/figures/loss_dense.png --title "Dense (MiniLM) fine-tuning"
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("out")
    p.add_argument("--title", default="Fine-tuning loss")
    a = p.parse_args()

    steps, tr, va = [], [], []
    with open(a.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["train_loss"]:
                steps.append(float(r["step"])); tr.append(float(r["train_loss"]))
    with open(a.csv, encoding="utf-8") as f:
        vs, vv = [], []
        for r in csv.DictReader(f):
            if r["val_loss"]:
                vs.append(float(r["step"])); vv.append(float(r["val_loss"]))

    plt.figure(figsize=(6.4, 3.8), dpi=200)
    plt.plot(steps, tr, marker="o", ms=3, label="training loss")
    plt.plot(vs, vv, marker="o", ms=3, label="validation loss")
    plt.xlabel("training step"); plt.ylabel("MNRL loss"); plt.title(a.title)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
