"""IR evaluation metrics for graded relevance judgments.

WANDS labels map to graded gains: Exact=2, Partial=1, Irrelevant=0.

Two notions of "relevant" for binary metrics (recall@k, MRR):
  - strict: only Exact counts as relevant (default — Partial dominates WANDS,
    so strict mode is the more discriminative signal)
  - lenient: Exact and Partial both count

nDCG uses the graded gains directly, so it needs no such switch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

GAIN = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


@dataclass
class QueryJudgments:
    """Relevance judgments for a single query.

    labels: product_id -> graded gain (0/1/2). Unjudged products are treated
    as gain 0 — standard pooled-judgment assumption, worth stating in the README.
    """

    query_id: int
    query: str
    labels: Mapping[int, int]
    query_class: str | None = None

    _ideal_gains: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._ideal_gains = sorted(self.labels.values(), reverse=True)

    def gain(self, product_id: int) -> int:
        return self.labels.get(product_id, 0)

    def relevant_ids(self, strict: bool = True) -> set[int]:
        threshold = 2 if strict else 1
        return {pid for pid, g in self.labels.items() if g >= threshold}


# ---------------------------------------------------------------------------
# Per-query metrics. `ranking` is the retriever's ordered list of product_ids.
# ---------------------------------------------------------------------------

def recall_at_k(ranking: Sequence[int], judg: QueryJudgments, k: int, strict: bool = True) -> float | None:
    """Fraction of relevant products found in the top k.

    Returns None when the query has no relevant products under the chosen
    mode (undefined recall) — callers must skip, not count as 0, or the
    aggregate is silently deflated.
    """
    relevant = judg.relevant_ids(strict=strict)
    if not relevant:
        return None
    hits = sum(1 for pid in ranking[:k] if pid in relevant)
    return hits / len(relevant)


def mrr(ranking: Sequence[int], judg: QueryJudgments, k: int | None = None, strict: bool = True) -> float | None:
    """Reciprocal rank of the first relevant product (1-indexed)."""
    relevant = judg.relevant_ids(strict=strict)
    if not relevant:
        return None
    cutoff = len(ranking) if k is None else k
    for rank, pid in enumerate(ranking[:cutoff], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[int], judg: QueryJudgments, k: int) -> float | None:
    """Graded nDCG@k with gains Exact=2, Partial=1 and log2 discount."""
    ideal = judg._ideal_gains[:k]
    idcg = sum(g / math.log2(r + 1) for r, g in enumerate(ideal, start=1) if g > 0)
    if idcg == 0:
        return None  # no judged-relevant products at all
    dcg = sum(judg.gain(pid) / math.log2(r + 1) for r, pid in enumerate(ranking[:k], start=1))
    return dcg / idcg


# ---------------------------------------------------------------------------
# Aggregation across queries
# ---------------------------------------------------------------------------

def evaluate_run(
    run: Mapping[int, Sequence[int]],
    judgments: Mapping[int, QueryJudgments],
    ks: Sequence[int] = (5, 10, 20),
    strict: bool = True,
) -> dict[str, float]:
    """Aggregate metrics for a retrieval run.

    run: query_id -> ranked list of product_ids.
    Queries present in `judgments` but missing from `run` count as empty
    rankings (a retriever that returns nothing must not look good).
    Per-metric None values (undefined for that query) are skipped; the
    number of contributing queries is reported per metric as _n_<metric>.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def add(name: str, value: float | None) -> None:
        if value is None:
            return
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1

    for qid, judg in judgments.items():
        ranking = list(run.get(qid, []))
        for k in ks:
            add(f"recall@{k}", recall_at_k(ranking, judg, k, strict=strict))
            add(f"ndcg@{k}", ndcg_at_k(ranking, judg, k))
        add("mrr@10", mrr(ranking, judg, k=10, strict=strict))

    results = {name: sums[name] / counts[name] for name in sums}
    results["n_queries"] = float(len(judgments))
    for name, c in counts.items():
        results[f"_n_{name}"] = float(c)
    return results


def format_results_row(name: str, results: Mapping[str, float], ks: Sequence[int] = (5, 10, 20)) -> str:
    """One markdown table row for the README ablation table."""
    cells = [name]
    for k in ks:
        cells.append(f"{results.get(f'recall@{k}', float('nan')):.3f}")
    for k in ks:
        cells.append(f"{results.get(f'ndcg@{k}', float('nan')):.3f}")
    cells.append(f"{results.get('mrr@10', float('nan')):.3f}")
    return "| " + " | ".join(cells) + " |"
