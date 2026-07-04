"""Golden test set builder for the WANDS product-search benchmark.

Loads the three WANDS TSVs (product.csv, query.csv, label.csv — tab-separated
despite the .csv extension) and exposes:

* ``load_wands()``     -> WandsDataset (products, queries, judgments)
* ``dataset_report()`` -> label-skew statistics worth quoting in the README

Design notes (stated here because they define the benchmark's methodology):

1. Judgments are pooled, not exhaustive: unjudged (query, product) pairs are
   treated as Irrelevant (gain 0). This is the standard assumption and it
   slightly penalises retrievers that surface good-but-unjudged products.
2. "Partial" dominates the labels (~63%), so binary metrics default to
   strict mode (Exact-only relevance) in metrics.py, while nDCG uses the
   full graded scale. Both are reported.
3. Queries with zero Exact judgments cannot contribute to strict recall/MRR;
   metrics.py returns None for them and skips them in aggregation.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .metrics import GAIN, QueryJudgments


@dataclass(frozen=True)
class Product:
    product_id: int
    name: str
    product_class: str
    category_hierarchy: str
    description: str
    features: str  # pipe-delimited "attribute : value" pairs
    average_rating: float | None
    rating_count: float | None
    review_count: float | None

    def feature_pairs(self) -> Iterator[tuple[str, str]]:
        """Yield (attribute, value) pairs from the pipe-delimited features."""
        for part in self.features.split("|"):
            if ":" in part:
                attr, _, val = part.partition(":")
                yield attr.strip(), val.strip()


@dataclass
class WandsDataset:
    products: dict[int, Product]
    judgments: dict[int, QueryJudgments]  # query_id -> judgments

    @property
    def queries(self) -> dict[int, str]:
        return {qid: j.query for qid, j in self.judgments.items()}


def _read_tsv(path: Path) -> Iterator[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f, delimiter="\t")


def _to_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_wands(data_dir: str | Path) -> WandsDataset:
    """Load WANDS from a directory containing product.csv/query.csv/label.csv."""
    data_dir = Path(data_dir)

    products: dict[int, Product] = {}
    for row in _read_tsv(data_dir / "product.csv"):
        pid = int(row["product_id"])
        products[pid] = Product(
            product_id=pid,
            name=row.get("product_name", "") or "",
            product_class=row.get("product_class", "") or "",
            category_hierarchy=row.get("category hierarchy", "") or "",
            description=row.get("product_description", "") or "",
            features=row.get("product_features", "") or "",
            average_rating=_to_float(row.get("average_rating", "")),
            rating_count=_to_float(row.get("rating_count", "")),
            review_count=_to_float(row.get("review_count", "")),
        )

    query_meta: dict[int, tuple[str, str]] = {}
    for row in _read_tsv(data_dir / "query.csv"):
        query_meta[int(row["query_id"])] = (row["query"], row.get("query_class", "") or "")

    labels_by_query: dict[int, dict[int, int]] = {}
    unknown_labels: Counter[str] = Counter()
    for row in _read_tsv(data_dir / "label.csv"):
        label = row["label"].strip()
        if label not in GAIN:
            unknown_labels[label] += 1
            continue
        qid = int(row["query_id"])
        pid = int(row["product_id"])
        labels_by_query.setdefault(qid, {})[pid] = GAIN[label]
    if unknown_labels:
        raise ValueError(f"Unexpected labels in label.csv: {dict(unknown_labels)}")

    judgments = {
        qid: QueryJudgments(
            query_id=qid,
            query=query_meta.get(qid, ("", ""))[0],
            labels=labels,
            query_class=query_meta.get(qid, ("", ""))[1],
        )
        for qid, labels in labels_by_query.items()
    }
    return WandsDataset(products=products, judgments=judgments)


def dataset_report(ds: WandsDataset) -> dict[str, object]:
    """Skew statistics for the README: label distribution, per-query spread."""
    label_counts: Counter[int] = Counter()
    judged_per_query: list[int] = []
    exact_per_query: list[int] = []
    for j in ds.judgments.values():
        gains = list(j.labels.values())
        judged_per_query.append(len(gains))
        exact_per_query.append(sum(1 for g in gains if g == 2))
        label_counts.update(gains)

    judged_per_query.sort()
    n = len(judged_per_query)
    return {
        "n_products": len(ds.products),
        "n_queries": n,
        "n_judgments": sum(judged_per_query),
        "label_distribution": {
            "Exact": label_counts.get(2, 0),
            "Partial": label_counts.get(1, 0),
            "Irrelevant": label_counts.get(0, 0),
        },
        "judged_per_query_min": judged_per_query[0] if n else 0,
        "judged_per_query_median": judged_per_query[n // 2] if n else 0,
        "judged_per_query_max": judged_per_query[-1] if n else 0,
        "queries_with_exact": sum(1 for c in exact_per_query if c > 0),
        "queries_without_exact": sum(1 for c in exact_per_query if c == 0),
    }


if __name__ == "__main__":
    import json
    import sys

    ds = load_wands(sys.argv[1] if len(sys.argv) > 1 else "data")
    print(json.dumps(dataset_report(ds), indent=2))
