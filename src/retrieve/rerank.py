"""Cross-encoder reranking of a candidate product ranking.

A cross-encoder reads (query, document) *together* through one transformer
pass, so it models token-level interaction that bi-encoder retrieval cannot.
It is far too slow to score a whole corpus, hence the standard two-stage
pattern benchmarked here: a cheap retriever produces top-N candidates, the
cross-encoder re-orders only those N.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, strong MS MARCO
baseline). The document text a candidate is scored on comes from a chunking
strategy — by default `name_desc`, deliberately different from whatever the
first stage indexed, so reranking quality isn't confounded with index
composition. That choice is itself reportable.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from evals.golden_set import Product
from src.ingest.chunking import STRATEGIES


class CrossEncoderModel(Protocol):
    name: str
    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int = 64) -> list[float]: ...


class SentenceTransformersCrossEncoder:
    """Wraps sentence_transformers.CrossEncoder. pip install sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        from sentence_transformers import CrossEncoder  # lazy import

        self.name = f"ce:{model_name.split('/')[-1]}"
        self._model = CrossEncoder(model_name, device=device, max_length=512)

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int = 64) -> list[float]:
        return [float(s) for s in self._model.predict(list(pairs), batch_size=batch_size)]


class LexicalOverlapScorer:
    """Toy scorer for mechanics tests only — token overlap ratio. Never report."""

    name = "toy:lexical-overlap"

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int = 0) -> list[float]:
        out = []
        for query, doc in pairs:
            q = set(query.lower().split())
            d = set(doc.lower().split())
            out.append(len(q & d) / len(q) if q else 0.0)
        return out


def _doc_text(product: Product, strategy: str) -> str:
    chunks = list(STRATEGIES[strategy](product))
    return " ".join(c.text for c in chunks)


def rerank_run(
    run: Mapping[int, Sequence[int]],
    queries: Mapping[int, str],
    products: Mapping[int, Product],
    model: CrossEncoderModel,
    rerank_depth: int = 100,
    doc_strategy: str = "name_desc",
    batch_size: int = 64,
) -> dict[int, list[int]]:
    """Rerank the top `rerank_depth` of each query's ranking; keep the tail as-is.

    Keeping the tail (instead of truncating) means recall@k for k <= depth is
    unaffected by candidates the cross-encoder never saw — the rerank ablation
    measures ordering quality, not a hidden recall cut.
    """
    doc_cache: dict[int, str] = {}
    out: dict[int, list[int]] = {}

    for qid, ranking in run.items():
        head = list(ranking[:rerank_depth])
        tail = list(ranking[rerank_depth:])
        if not head:
            out[qid] = tail
            continue
        query = queries[qid]
        pairs = []
        for pid in head:
            if pid not in doc_cache:
                doc_cache[pid] = _doc_text(products[pid], doc_strategy)
            pairs.append((query, doc_cache[pid]))
        scores = model.score(pairs, batch_size=batch_size)
        reordered = [pid for pid, _ in sorted(zip(head, scores), key=lambda x: -x[1])]
        out[qid] = reordered + tail
    return out
