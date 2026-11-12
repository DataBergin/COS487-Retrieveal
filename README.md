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
python -m apps.processing.preprocess
python -m apps.retrieval.index
```

##### Backend
From the repository root:

```bash
uvicorn apps.prod.backend.main:app --reload --port 8000
```

##### Frontend
In a second terminal:

```bash
cd apps/prod/frontend
npm install
npm run dev
```

##### Evaluation
```bash
python -m apps.evaluation.evaluate
```




