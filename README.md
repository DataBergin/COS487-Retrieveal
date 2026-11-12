# COS487 Information Retrieval - Course Project
Code for the course project in COS487 Information Retrieval at the University of Southern Maine,
searching a collection of 34,308 arXiv papers (2,275,574 paragraphs) at the paper and paragraph level.

- **Part 1** (team project, team name Search Party): TF-IDF and BM25 retrieval, evaluation and a search
  interface. Only BM25 is used again in Part 2, as the baseline.
- **Part 2** (JackSearchParty): dense (MiniLM) and sparse
  (SPLADE) neural retrieval with fine-tuning, in `apps/neural/`. See [Part 2: Neural Retrieval](#part-2-neural-retrieval).

### Part 2: Neural Retrieval
Part 2 (dense and SPLADE retrieval, fine-tuning, evaluation) lives in `apps/neural/`; see
`apps/neural/README.md` for the details.
After `bash bin/install.sh` (which also installs the neural libraries), run:

```bash
bash bin/run.sh --neural
```

This runs every stage in order: Part 1 BM25 runs, corpus extraction, fine-tuning, encoding, search,
zero-shot comparison, evaluation and significance tests, figures, and packaging of the
`Retrieval Results/` folder. It takes several hours and needs a CUDA GPU (I used an 8 GB RTX 3050).
To run only some stages, or to preview the commands:

```bash
bash bin/run.sh --neural train encode
bash bin/run.sh --neural --dry-run
```

### Manual Run
Install Python dependencies once from the repository root:

```bash
pip install -r requirements.txt
```

##### Environment Build
```bash
python -m apps.neural.corpus.extract_paragraphs                     
```
# dense
```bash
python -m apps.neural.training.train_dense --out models/dense_ft       # fine-tune on Train queries, writes loss_log.csv
python -m apps.neural.indexing.encode_dense --model models/dense_ft --out data/neural/dense_ft
python -m apps.neural.retrieval.search_dense --emb data/neural/dense_ft --model models/dense_ft --set study
python -m apps.neural.retrieval.search_dense --emb data/neural/dense_ft --model models/dense_ft --set test
```
# sparse
```bash
python -m apps.neural.training.train_splade --out models/splade_ft
python -m apps.neural.indexing.encode_splade --model models/splade_ft --out data/neural/splade_ft
python -m apps.neural.retrieval.search_splade --emb data/neural/splade_ft --model models/splade_ft --set study
python -m apps.neural.retrieval.search_splade --emb data/neural/splade_ft --model models/splade_ft --set test
```



