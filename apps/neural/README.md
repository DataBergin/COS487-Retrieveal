# Part 2: neural retrieval (`apps/neural/`)

Dense (all-MiniLM-L6-v2) and sparse (SPLADE, `naver/splade_v2_max`) retrieval over the arXiv
paragraph corpus, fine-tuned on the Train queries only, scored against the Part 1 BM25 baseline.
Run everything from the repository root. The `data/` and `models/` folders are not in git.

## One command

```bash
bash bin/install.sh              # once: conda environment, libraries, Part 1 data and indexes
bash bin/run.sh --neural         # every stage below, in order (hours; needs a CUDA GPU)
bash bin/run.sh --neural train encode   # or only some stages
bash bin/run.sh --neural --dry-run      # print the commands only
```

Stages: `baseline`, `paragraphs`, `train`, `encode`, `search`, `zeroshot`, `evaluate`, `figures`,
`package` (see `pipeline.py`). The commands it runs are the individual scripts shown below.

## Pipeline

```text
python -m apps.neural.corpus.extract_paragraphs                      # data/neural/paragraphs.tsv (2,275,574 rows)

# dense
python -m apps.neural.training.train_dense --out models/dense_ft       # fine-tune on Train queries, writes loss_log.csv
python -m apps.neural.indexing.encode_dense --model models/dense_ft --out data/neural/dense_ft
python -m apps.neural.retrieval.search_dense --emb data/neural/dense_ft --model models/dense_ft --set study
python -m apps.neural.retrieval.search_dense --emb data/neural/dense_ft --model models/dense_ft --set test

# sparse
python -m apps.neural.training.train_splade --out models/splade_ft
python -m apps.neural.indexing.encode_splade --model models/splade_ft --out data/neural/splade_ft
python -m apps.neural.retrieval.search_splade --emb data/neural/splade_ft --model models/splade_ft --set study
python -m apps.neural.retrieval.search_splade --emb data/neural/splade_ft --model models/splade_ft --set test
```

Check each script's `--help` for the exact options; the commands above show the order. Encoding is
sharded and resumable (a re-run skips finished shards).

## Layout

Each subfolder is one job in the pipeline, and `pipeline.py` runs them in order.

```text
apps/neural/
├── common.py        paths, query loading, light text cleaning, paragraph id -> paper id
├── pipeline.py      runs the stages below (used by `bash bin/run.sh --neural`)
├── corpus/          extract_paragraphs.py   build data/neural/paragraphs.tsv
├── encoders/        splade.py               SPLADE encoder: log(1 + ReLU(logits)), max over token positions
├── training/        train_dense.py          contrastive fine-tuning (MultipleNegativesRankingLoss, in-batch negatives)
│                    train_splade.py         contrastive fine-tuning plus a FLOPS sparsity penalty
├── indexing/        encode_dense.py, encode_splade.py   encode every paragraph (sharded, resumable)
├── retrieval/       search_dense.py, search_splade.py   write paragraph and paper run files
├── evaluation/      score.py                nDCG, nDCG@5, MRR, P@5, P@10 (relevant+partial) and MRR, P@5 (relevant only), via ranx
│                    significance.py         paired t-tests between runs, both relevance variants
│                    by_difficulty.py        Easy / Medium / Hard breakdown, counts of queries with nDCG = 0
│                    latency.py              per-query search time and index size
│                    package_results.py      copy run files and qrel into `Retrieval Results/` and check the format
└── reporting/       plot_loss.py, plot_results.py   loss curves and the nDCG@5 bar chart
                     build_annotation.py             0/1/2 annotation page for the Test queries
```

## Design notes

- Train queries are split 90/10 by query (seed 42) into training and validation. The training
  scripts assert that no Study or Test query id is present.
- The best checkpoint is chosen by validation loss.
- A paper's score is the maximum score of its paragraphs, taken from the top 1000 paragraphs.
- SPLADE text is truncated at 128 tokens, dense text at 256.

