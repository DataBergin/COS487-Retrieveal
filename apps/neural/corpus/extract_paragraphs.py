"""Extract raw (lightly cleaned) paragraph text for the neural models.

Output: data/neural/paragraphs.tsv, one row per paragraph: <paragraph_id>\t<text>
Paragraph IDs match the qrels (e.g. 2408.00001_8).

    python -m apps.neural.corpus.extract_paragraphs
"""
import glob
import json
import os

from ..common import NEURAL_DIR, PARAGRAPHS_TSV, RAW_DIR, clean_text


def main():
    os.makedirs(NEURAL_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    n = 0
    with open(PARAGRAPHS_TSV, "w", encoding="utf-8", newline="\n") as out:
        for i, path in enumerate(files, 1):
            with open(path, encoding="utf-8") as f:
                paper = json.load(f)
            for pid, text in (paper.get("paragraphs") or {}).items():
                out.write("%s\t%s\n" % (pid, clean_text(text)))
                n += 1
            if i % 5000 == 0:
                print("%d / %d papers (%d paragraphs)" % (i, len(files), n), flush=True)
    print("done: %d paragraphs -> %s" % (n, PARAGRAPHS_TSV))


if __name__ == "__main__":
    main()
