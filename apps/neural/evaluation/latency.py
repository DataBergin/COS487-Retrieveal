"""Measure per-query search latency and index size for the dense and SPLADE systems.

    python -m apps.neural.evaluation.latency --kind dense  --emb data/neural/dense_ft  --model models/dense_ft
    python -m apps.neural.evaluation.latency --kind splade --emb data/neural/splade_ft --model models/splade_ft

A query's time covers encoding the query, scoring it against the whole index and taking
the top 1000 paragraphs. Index loading and the paper regrouping step are not included.
The first few queries are a warm-up and are dropped. Close other GPU programs first.
"""
import argparse
import os
import time

import numpy as np
import torch

from ..common import ROOT, load_queries


def dir_size_gb(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1e9


def report(kind, times, size_gb, extra=""):
    t = np.array(times) * 1000
    print("%s: index %.2f GB%s | per-query ms: mean %.1f, median %.1f, p95 %.1f (n=%d)"
          % (kind, size_gb, extra, t.mean(), np.median(t), np.percentile(t, 95), len(t)))


def run_dense(a, queries):
    from sentence_transformers import SentenceTransformer
    from ..retrieval.search_dense import load_embeddings
    emb_dir = os.path.join(ROOT, a.emb)
    _, emb = load_embeddings(emb_dir)
    model = SentenceTransformer(os.path.join(ROOT, a.model), device="cuda")
    model.max_seq_length = 256
    docs = torch.from_numpy(emb).to("cuda", dtype=torch.float16)
    times = []
    for i, q in enumerate(queries):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qe = model.encode([q], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        s = docs @ qe[0].to(torch.float16)
        torch.topk(s, k=1000)[1].cpu()
        torch.cuda.synchronize()
        if i >= a.warmup:
            times.append(time.perf_counter() - t0)
    report("dense", times, dir_size_gb(emb_dir))


def run_splade(a, queries):
    from ..retrieval.search_splade import load_index
    from ..encoders.splade import Splade
    emb_dir = os.path.join(ROOT, a.emb)
    local = os.path.join(ROOT, a.model)
    enc = Splade(local if os.path.isdir(local) else a.model, max_length=256)
    _, docs = load_index(emb_dir, enc.vocab_size)
    times, nterms = [], []
    for i, q in enumerate(queries):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qv = enc.encode_dense([q])[0].cpu().numpy()
        nz = np.nonzero(qv >= 0.05)[0]
        s = docs[:, nz] @ qv[nz]
        np.argpartition(-s, 1000)[:1000]
        if i >= a.warmup:
            times.append(time.perf_counter() - t0)
            nterms.append(len(nz))
    report("splade", times, dir_size_gb(emb_dir),
           " | avg %.0f non-zero terms/paragraph, %.0f/query"
           % (docs.nnz / docs.shape[0], np.mean(nterms)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["dense", "splade"], required=True)
    p.add_argument("--emb", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    queries = load_queries("study")[1][: a.n + a.warmup]
    print("GPU:", torch.cuda.get_device_name(0))
    (run_dense if a.kind == "dense" else run_splade)(a, queries)


if __name__ == "__main__":
    main()
