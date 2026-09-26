"""Hybrid retrieval over the knowledge base: local dense embeddings + BM25.

The small multilingual embedder misses rare terms ("блау-газ", "оперение"), so
dense and lexical scores are min-max normalized and mixed with `lexical_weight`.
"""

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from zeppelin_rag.knowledge import Chunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    """Multilingual ONNX embeddings via fastembed; runs locally, no API key."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts)), dtype=np.float32)


def stems(text: str) -> list[str]:
    """Crude Russian-friendly stemming: first 5 letters of every word longer than 2."""
    return [w[:5] for w in re.findall(r"[a-zа-яё0-9]+", text.lower()) if len(w) > 2]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(stems(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.docs)
        df = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for term in set(stems(query)) & self.idf.keys():
            for i, doc in enumerate(self.docs):
                f = doc[term]
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                    out[i] += self.idf[term] * f * (self.k1 + 1) / (f + norm)
        return out


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _min_max(scores: np.ndarray) -> np.ndarray:
    span = scores.max() - scores.min()
    return (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, lexical_weight: float = 0.5):
        self.chunks = chunks
        self.embedder = embedder
        self.lexical_weight = lexical_weight
        texts = [c.text for c in chunks]
        self._matrix = _normalize(embedder.embed(texts))
        self._bm25 = BM25(texts)

    def search(self, query: str, k: int) -> list[Chunk]:
        dense = self._matrix @ _normalize(self.embedder.embed([query]))[0]
        lexical = self._bm25.scores(query)
        w = self.lexical_weight
        scores = (1 - w) * _min_max(dense) + w * _min_max(lexical)
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]
