"""Encode every paragraph with SPLADE into sparse (CSR) shards on disk. Resumable.

    python -m apps.neural.indexing.encode_splade --out data/neural/splade_zeroshot
    python -m apps.neural.indexing.encode_splade --model models/splade_ft --out data/neural/splade_ft
"""
import argparse
import os
import time

import numpy as np

from ..common import read_paragraphs
from ..encoders.splade import DEFAULT_MODEL, Splade


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="data/neural/splade_zeroshot")
    p.add_argument("--max_length", type=int, default=128, help="paragraph tokens kept; 128 is what the reported indexes were built with")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--shard_size", type=int, default=50_000)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N paragraphs")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids, texts = read_paragraphs()
    if args.limit:
        ids, texts = ids[: args.limit], texts[: args.limit]
    with open(os.path.join(args.out, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids))

    enc = Splade(args.model, max_length=args.max_length)
    n_shards = (len(texts) + args.shard_size - 1) // args.shard_size
    t0 = time.time()
    for s in range(n_shards):
        path = os.path.join(args.out, "shard_%04d.npz" % s)
        if os.path.exists(path):
            print("shard %d/%d exists, skipping" % (s + 1, n_shards))
            continue
        chunk = texts[s * args.shard_size:(s + 1) * args.shard_size]
        indptr, indices, data = enc.encode_sparse(chunk, args.batch_size, args.threshold)
        np.savez(path + ".tmp.npz", indptr=indptr, indices=indices, data=data)
        os.replace(path + ".tmp.npz", path)
        avg = len(indices) / max(len(chunk), 1)
        rate = (s + 1) * args.shard_size / (time.time() - t0)
        print("shard %d/%d done | avg nonzero terms/paragraph %.0f | %.0f paragraphs/s" %
              (s + 1, n_shards, avg, rate), flush=True)
    print("encoding complete:", args.out)


if __name__ == "__main__":
    main()
