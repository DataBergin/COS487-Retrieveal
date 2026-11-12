"""Shared helpers for the neural (Part 2) retrieval code.

Neural models use their own tokenizer, so cleaning here is deliberately light:
strip markup and normalize whitespace. No stemming, no stopword removal.
"""
import html
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA, "JSON Files")
NEURAL_DIR = os.path.join(DATA, "neural")
PARAGRAPHS_TSV = os.path.join(NEURAL_DIR, "paragraphs.tsv")

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(text):
    """Strip HTML tags, unescape entities, collapse whitespace."""
    text = _TAG.sub(" ", text or "")
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def load_queries(name):
    """Return (ids, texts) for 'train', 'study' or 'test' from data/<Name>.json."""
    path = os.path.join(DATA, name.capitalize() + ".json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [str(r["query_id"]) for r in rows], [clean_text(r["query"]) for r in rows]


def read_paragraphs(path=PARAGRAPHS_TSV):
    """Return (ids, texts) from the extracted paragraph file."""
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, _, text = line.rstrip("\n").partition("\t")
            ids.append(pid)
            texts.append(text)
    return ids, texts


def paper_of(para_id):
    """'2408.00001_8' -> '2408.00001'."""
    return para_id.rsplit("_", 1)[0]
