"""Document composition strategies for indexing WANDS products.

"What text do we embed / index per product?" is the first ablation axis of
this benchmark. Each strategy turns a Product into one or more Chunks; the
same strategies feed both the BM25 index and the dense (vector) index so the
comparison stays apples-to-apples.

Strategies
----------
name_only        : product name. Deliberately weak baseline.
name_desc        : name + class + description. The "obvious" composition.
full             : name + class + category hierarchy + description + all
                   features flattened to natural-ish text. Maximum context,
                   but risks diluting the embedding with boilerplate
                   attributes (the classic long-document problem).
parent_child     : multiple focused child chunks per product (identity /
                   description / features), each carrying the parent
                   product_id. Retrieval scores children; results are
                   collapsed to parents via max-score dedup. This mirrors
                   the parent-child design from production RAG systems:
                   embed small and precise, return the whole product.

All strategies emit Chunk objects so downstream indexers are agnostic to
which one produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from evals.golden_set import Product

# Features that are mostly boilerplate noise for semantic matching.
_SKIP_FEATURE_PREFIXES = (
    "country of origin",
    "warranty",
    "commercial warranty",
    "supplier intended and approved use",
    "california proposition",
)


@dataclass(frozen=True)
class Chunk:
    """One indexable unit of text belonging to a parent product."""

    chunk_id: str        # e.g. "12345" or "12345:features"
    product_id: int      # parent — what evaluation is scored against
    text: str
    section: str = "full"  # identity | description | features | full


def _clean_features(product: Product, max_pairs: int = 40) -> str:
    parts: list[str] = []
    for attr, val in product.feature_pairs():
        if not val:
            continue
        if any(attr.lower().startswith(p) for p in _SKIP_FEATURE_PREFIXES):
            continue
        parts.append(f"{attr}: {val}")
        if len(parts) >= max_pairs:
            break
    return ". ".join(parts)


def _identity_text(product: Product) -> str:
    bits = [product.name]
    if product.product_class:
        bits.append(product.product_class)
    if product.category_hierarchy:
        bits.append(product.category_hierarchy.replace("/", " > "))
    return ". ".join(b for b in bits if b)


# ---------------------------------------------------------------------------
# Strategies: Product -> Iterator[Chunk]
# ---------------------------------------------------------------------------

def name_only(product: Product) -> Iterator[Chunk]:
    yield Chunk(str(product.product_id), product.product_id, product.name or "")


def name_desc(product: Product) -> Iterator[Chunk]:
    text = ". ".join(
        t for t in (_identity_text(product), product.description) if t
    )
    yield Chunk(str(product.product_id), product.product_id, text)


def full(product: Product) -> Iterator[Chunk]:
    text = ". ".join(
        t
        for t in (
            _identity_text(product),
            product.description,
            _clean_features(product),
        )
        if t
    )
    yield Chunk(str(product.product_id), product.product_id, text)


def parent_child(product: Product) -> Iterator[Chunk]:
    pid = product.product_id
    yield Chunk(f"{pid}:identity", pid, _identity_text(product), "identity")
    if product.description:
        yield Chunk(f"{pid}:description", pid, product.description, "description")
    features = _clean_features(product)
    if features:
        # prefix the name so a features-only chunk still knows what it describes
        yield Chunk(f"{pid}:features", pid, f"{product.name}. {features}", "features")


STRATEGIES: dict[str, Callable[[Product], Iterator[Chunk]]] = {
    "name_only": name_only,
    "name_desc": name_desc,
    "full": full,
    "parent_child": parent_child,
}


def build_chunks(
    products: Iterable[Product], strategy: str
) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise KeyError(f"Unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    fn = STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for product in products:
        chunks.extend(fn(product))
    return chunks


def collapse_to_products(
    scored_chunks: Iterable[tuple[Chunk, float]], limit: int | None = None
) -> list[tuple[int, float]]:
    """Collapse chunk-level scores to a product ranking (max score wins).

    This is the "return the parent" half of parent-child retrieval; it is a
    no-op for single-chunk strategies. Output is sorted best-first.
    """
    best: dict[int, float] = {}
    for chunk, score in scored_chunks:
        if score > best.get(chunk.product_id, float("-inf")):
            best[chunk.product_id] = score
    ranked = sorted(best.items(), key=lambda x: -x[1])
    return ranked[:limit] if limit is not None else ranked
