"""SPLADE encoder: text -> sparse vector over the BERT vocabulary.

SPLADE (Formal et al., 2021): run a masked-language-model head over the text,
then for every vocabulary term take the max over token positions of
    log(1 + ReLU(logit))
The log saturates large activations so no term dominates, ReLU (plus the
sparsity regularizer used in training) leaves most terms at zero, and terms
that are not in the text can still get weight (expansion).
"""
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEFAULT_MODEL = "naver/splade_v2_max"  # the checkpoint used in the course's sparsemodel.py


class Splade:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, max_length=256, half=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        if half and self.device == "cuda":
            self.model.half()
        if self.device == "cuda":
            # Cap PyTorch's cache so it cannot fill the whole card; a nearly full GPU makes
            # Windows spill into slow shared memory (observed: ~100 s/shard became ~17 min).
            torch.cuda.set_per_process_memory_fraction(0.65)
        self.max_length = max_length
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def encode_dense(self, texts):
        """Return a (len(texts), vocab) float tensor of SPLADE weights (on device)."""
        b = self.tok(texts, padding=True, truncation=True, max_length=self.max_length,
                     return_tensors="pt").to(self.device)
        logits = self.model(**b).logits
        w = torch.log1p(torch.relu(logits.float())) * b["attention_mask"].unsqueeze(-1)
        return w.max(dim=1).values

    def encode_sparse(self, texts, batch_size=32, threshold=0.05):
        """Return CSR parts (indptr, indices, data) for the texts, pruning weights below threshold.

        Texts are sorted by length inside the call so batches pad little.
        """
        import time
        order = np.argsort([len(t) for t in texts])
        rows = [None] * len(texts)
        t0 = time.time()
        for s in range(0, len(texts), batch_size):
            if (s // batch_size) % 200 == 0 and s:
                print("  batch %d/%d  %.0f paragraphs/s" % (s // batch_size, len(texts) // batch_size,
                                                           s / (time.time() - t0)), flush=True)
            idx = order[s:s + batch_size]
            w = self.encode_dense([texts[i] for i in idx])
            w = torch.where(w >= threshold, w, torch.zeros_like(w))
            nz = w.nonzero(as_tuple=False)
            vals = w[nz[:, 0], nz[:, 1]].cpu().numpy()
            cols = nz[:, 1].cpu().numpy()
            counts = torch.bincount(nz[:, 0], minlength=len(idx)).cpu().numpy()
            off = 0
            for j, i in enumerate(idx):
                c = counts[j]
                rows[i] = (cols[off:off + c].astype(np.int32), vals[off:off + c].astype(np.float16))
                off += c
        indptr = np.zeros(len(texts) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(r[0]) for r in rows])
        indices = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, np.int32)
        data = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, np.float16)
        return indptr, indices, data

    def top_terms(self, text, k=15, threshold=0.05):
        """Human-readable (term, weight) pairs for one text, highest weight first."""
        w = self.encode_dense([text])[0]
        vals, idx = torch.topk(w, k)
        return [(self.tok.convert_ids_to_tokens(int(i)), round(float(v), 2))
                for v, i in zip(vals.cpu(), idx.cpu()) if v >= threshold]
