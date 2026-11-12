"""Encode every paragraph with a bi-encoder and save embeddings to disk.

Resumable: embeddings are written in shards, and finished shards are skipped
on re-run. Paragraphs are encoded in file order (sentence-transformers sorts
each shard by length internally, which keeps padding low).

    python -m apps.neural.indexing.encode_dense --model sentence-transformers/all-MiniLM-L6-v2 --out data/neural/dense_zeroshot
    python -m apps.neural.indexing.encode_dense --model models/dense_ft --out data/neural/dense_ft
"""
import argparse
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..common import read_paragraphs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="data/neural/dense_zeroshot")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--shard_size", type=int, default=100_000)
    p.add_argument("--limit", type=int, default=0, help="encode only the first N paragraphs (smoke test)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids, texts = read_paragraphs()
    if args.limit:
        ids, texts = ids[: args.limit], texts[: args.limit]
    with open(os.path.join(args.out, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length
    if device == "cuda":
        model.half()

    n_shards = (len(texts) + args.shard_size - 1) // args.shard_size
    for s in range(n_shards):
        path = os.path.join(args.out, "shard_%04d.npy" % s)
        if os.path.exists(path):
            print("shard %d/%d exists, skipping" % (s + 1, n_shards))
            continue
        chunk = texts[s * args.shard_size : (s + 1) * args.shard_size]
        emb = model.encode(chunk, batch_size=args.batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
        np.save(path + ".tmp.npy", emb.astype(np.float16))
        os.replace(path + ".tmp.npy", path)
        print("shard %d/%d done" % (s + 1, n_shards), flush=True)
    print("encoding complete:", args.out)


if __name__ == "__main__":
    main()
