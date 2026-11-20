"""Lexical relevance (Okapi BM25) used to rank tool schemas against the user query.

BM25 scores a document by how many query terms it contains, weighting rare terms higher (inverse
document frequency) and damping both repeated terms and long documents. It needs no model download,
which keeps tool selection deterministic and offline.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .text import terms


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs = [Counter(doc) for doc in documents]
        self._lengths = [len(doc) for doc in documents]
        self._average = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._df: Counter[str] = Counter()
        for doc in self._docs:
            self._df.update(doc.keys())

    def idf(self, term: str) -> float:
        n, df = len(self._docs), self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query_terms: Sequence[str]) -> list[float]:
        unique = list(dict.fromkeys(query_terms))
        results = []
        for doc, length in zip(self._docs, self._lengths, strict=True):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * (length / self._average if self._average else 0.0))
            for term in unique:
                frequency = doc.get(term, 0)
                if frequency:
                    score += self.idf(term) * frequency * (self.k1 + 1) / (frequency + norm)
            results.append(score)
        return results


def relevance_scores(query: str, documents: Sequence[str]) -> list[float]:
    """BM25 relevance of each document to ``query``, scaled so the best match is 1.0 (all 0 if none match)."""
    if not documents:
        return []
    raw = BM25([terms(doc) for doc in documents]).scores(terms(query))
    top = max(raw)
    return [value / top if top > 0 else 0.0 for value in raw]
