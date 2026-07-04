"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF (Cormack et al., 2009) fuses ranked lists using only ranks, not scores:

    RRF(d) = sum over systems s of  w_s / (rrf_k + rank_s(d))

Rank-based fusion is the standard choice for hybrid sparse+dense retrieval
because BM25 scores and cosine similarities live on incomparable scales;
score-normalised fusion (e.g. min-max) is brittle to outliers. rrf_k=60 is
the conventional constant — larger values flatten the difference between
adjacent ranks. This is the same fusion family Qdrant/OpenSearch/Elastic
expose for hybrid search; here it's ~30 explicit lines instead of a flag.

Operates on product-level runs (query_id -> ranked product_ids), i.e. AFTER
chunk->product collapse, so any retriever combination can be fused.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence


def rrf_fuse_rankings(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[int]:
    """Fuse ranked lists of product_ids for a single query."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match number of rankings")

    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] += weight / (rrf_k + rank)

    fused = sorted(scores, key=lambda pid: -scores[pid])
    return fused[:top_k] if top_k is not None else fused


def rrf_fuse_runs(
    runs: Sequence[Mapping[int, Sequence[int]]],
    weights: Sequence[float] | None = None,
    rrf_k: int = 60,
    top_k: int = 100,
) -> dict[int, list[int]]:
    """Fuse whole runs (query_id -> ranking). Union of query_ids is used;
    a system missing a query simply contributes nothing for it."""
    all_qids: set[int] = set()
    for run in runs:
        all_qids.update(run.keys())

    return {
        qid: rrf_fuse_rankings(
            [run.get(qid, []) for run in runs],
            weights=weights,
            rrf_k=rrf_k,
            top_k=top_k,
        )
        for qid in all_qids
    }
