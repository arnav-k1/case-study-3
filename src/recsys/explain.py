"""'Because you liked X' explanations from the trained model's own item geometry.

- Item k-NN: the rated movie with the highest item-item similarity to the
  recommendation (the neighbour that contributed most, as in Amazon's
  item-to-item lists; Linden et al., 2003).
- ALS / matrix factorization: cosine similarity between item embeddings.
Only movies the user rated >= `like_threshold` are used as anchors.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps


class Explainer:
    def __init__(self, scorer, like_threshold: float = 4.0):
        self.like_threshold = like_threshold
        self.vocab = scorer.items
        self.kind = None
        if hasattr(scorer, "sim_matrix"):
            self.kind = "knn"
            self.sim: sps.csr_array = scorer.sim_matrix.to_scipy().tocsr()
        elif hasattr(scorer, "item_embeddings"):
            self.kind = "mf"
            emb = np.asarray(scorer.item_embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self.emb = emb / np.maximum(norms, 1e-9)
        else:
            raise TypeError(f"no explanation support for {type(scorer).__name__}")

    def _num(self, ids) -> np.ndarray:
        return np.asarray(self.vocab.numbers(np.asarray(ids, dtype=np.int64), missing="negative"))

    def similarities(self, target: int, anchors: list[int]) -> np.ndarray:
        """Similarity of `target` to each anchor (NaN if unknown)."""
        t = self._num([target])[0]
        a = self._num(anchors)
        out = np.full(len(anchors), np.nan, dtype=np.float32)
        if t < 0:
            return out
        ok = a >= 0
        if self.kind == "knn":
            row = self.sim[[t], :].toarray().ravel()
            col = self.sim[:, [t]].toarray().ravel()  # matrix may store only one direction per pair
            s = np.maximum(row, col)
            out[ok] = s[a[ok]]
        else:
            out[ok] = self.emb[a[ok]] @ self.emb[t]
        return out

    def because(self, target: int, history: dict[int, float], top: int = 1) -> list[tuple[int, float]]:
        """Top anchor movies (movieId, similarity) the user liked that best explain `target`."""
        liked = [m for m, r in history.items() if r >= self.like_threshold] or list(history)
        if not liked:
            return []
        sims = self.similarities(target, liked)
        order = np.argsort(-np.nan_to_num(sims, nan=-np.inf))
        return [(int(liked[i]), float(sims[i])) for i in order[:top] if np.isfinite(sims[i]) and sims[i] > 0]
