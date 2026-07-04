"""BM25 sparse retrieval over any chunk-composition strategy.

Self-contained BM25 (Okapi) implementation — no external index dependency —
so the benchmark's sparse leg is fully reproducible and its parameters
(k1, b) are explicit rather than buried in a library default.

Tokenization is deliberately simple (lowercase, alphanumeric word chars,
inch-mark normalisation) and shared with nothing else: BM25 sees exactly
what we say it sees. Chunk-level scores are collapsed to product-level
rankings via max-score (see chunking.collapse_to_products).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from src.ingest.chunking import Chunk, collapse_to_products

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    # normalise inch marks like 21.7'' -> "21.7 inch" is out of scope for the
    # baseline; we simply lowercase and split on non-alphanumerics.
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 with standard parameters k1=1.5, b=0.75."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = list(chunks)

        self._doc_tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)  # term -> chunk idxs

        for idx, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            self._doc_tf.append(tf)
            self._doc_len.append(len(tokens))
            for term in tf:
                self._postings[term].append(idx)

        n = len(self.chunks)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        # BM25+-style floor at 0 via the standard idf formulation with 0.5 smoothing
        self._idf = {
            term: math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for term, post in self._postings.items()
        }

    def search_chunks(self, query: str, top_k: int = 100) -> list[tuple[Chunk, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx in self._postings[term]:
                tf = self._doc_tf[idx][term]
                dl = self._doc_len[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.chunks[i], s) for i, s in ranked]

    def search(self, query: str, top_k: int = 100, chunk_pool: int = 300) -> list[tuple[int, float]]:
        """Product-level ranking: score chunks, collapse to parents, cut to top_k.

        chunk_pool > top_k so that multi-chunk strategies aren't starved after
        deduplication to parents.
        """
        scored = self.search_chunks(query, top_k=chunk_pool)
        return collapse_to_products(scored, limit=top_k)


def build_run(
    index: BM25Index, queries: dict[int, str], top_k: int = 100
) -> dict[int, list[int]]:
    """query_id -> ranked product_ids, the shape evals.metrics.evaluate_run expects."""
    return {
        qid: [pid for pid, _ in index.search(q, top_k=top_k)]
        for qid, q in queries.items()
    }
