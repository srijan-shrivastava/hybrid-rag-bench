"""MCP server exposing the hybrid-rag-bench retrieval stack as typed tools.

Design notes
------------
Retrieval strategy is a *tool parameter*, not a server-level config. The
ablation in this repo shows the tradeoff is real and query-dependent
(hybrid+rerank reaches nDCG@10 = 0.784, but costs a cross-encoder pass over
100 candidates), so the calling model is given the choice explicitly rather
than having one point on the quality/latency curve hard-coded for it.

Transport is stdio: this is a single-user local server launched by the client
process. HTTP/SSE would be the choice for a shared remote deployment, where
you would also need auth and per-caller rate limiting — neither is meaningful
over stdio, where the client already owns the process.

Corpus and models load lazily on first tool call, not at import. MCP clients
start servers eagerly at boot; loading a cross-encoder and 43k embeddings on
import would stall client startup for every session, including ones that
never search.
"""

from __future__ import annotations

import sys
import threading
from enum import Enum
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# --------------------------------------------------------------------------
# ADAPTER LAYER -- the only part coupled to module signatures I haven't seen.
# Confirm these four calls against src/retrieve/bm25.py and dense.py.
# --------------------------------------------------------------------------

TOOL_VERSION = "1.0.0"
MAX_TOP_K = 50
RERANK_DEPTH = 100
SYNTHETIC_QID = 0

_state: dict[str, Any] = {}
_lock = threading.Lock()


def _load() -> dict[str, Any]:
    """Load corpus, retrievers and reranker once, on first use."""
    if _state:
        return _state
    with _lock:
        if _state:  # another thread won the race
            return _state

        from evals.golden_set import load_products  # ADAPT: loader name
        from src.retrieve.bm25 import BM25Retriever  # ADAPT
        from src.retrieve.dense import DenseRetriever  # ADAPT
        from src.retrieve.rerank import SentenceTransformersCrossEncoder

        products = load_products()  # ADAPT -> Mapping[int, Product]

        _state["products"] = products
        _state["bm25"] = BM25Retriever(products, strategy="name_desc")  # ADAPT
        _state["dense"] = DenseRetriever(products, strategy="name_desc")  # ADAPT
        _state["reranker"] = SentenceTransformersCrossEncoder()
    return _state


def _bm25_search(query: str, depth: int) -> list[int]:
    """ADAPT: must return a ranked list of product_ids for one query."""
    return list(_load()["bm25"].search(query, top_k=depth))


def _dense_search(query: str, depth: int) -> list[int]:
    """ADAPT: must return a ranked list of product_ids for one query."""
    return list(_load()["dense"].search(query, top_k=depth))


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class RetrievalMethod(str, Enum):
    """Retrieval configurations, ordered by cost.

    bm25            lexical only; cheapest, no model load
    dense           bi-encoder only; better ordering than bm25 on short queries
    hybrid          RRF fusion of bm25 + dense
    hybrid_rerank   hybrid, then cross-encoder over the top candidates (best)
    """

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


mcp = FastMCP("hybrid-rag-bench")


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Structured error return.

    Tools return errors as data rather than raising. An exception crossing the
    protocol boundary gives the model an opaque failure; a typed error payload
    lets it correct the call itself (e.g. clamp top_k and retry).
    """
    return {"ok": False, "error": {"code": code, "message": message, **extra}}


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@mcp.tool()
def search_products(
    query: str = Field(..., min_length=1, max_length=500, description="Free-text product search query."),
    top_k: int = Field(10, ge=1, le=MAX_TOP_K, description=f"Results to return (max {MAX_TOP_K})."),
    method: RetrievalMethod = Field(
        RetrievalMethod.HYBRID_RERANK,
        description="Retrieval config. Default is the highest-quality setting; "
        "use 'bm25' or 'dense' when latency matters more than ranking quality.",
    ),
) -> dict[str, Any]:
    """Search the product corpus and return ranked results.

    Read-only and idempotent: the same query with the same method returns the
    same ranking, so a client may safely retry.
    """
    query = query.strip()
    if not query:
        return _error("empty_query", "Query is empty after trimming whitespace.")

    try:
        state = _load()
        products = state["products"]
        depth = RERANK_DEPTH if method == RetrievalMethod.HYBRID_RERANK else max(top_k, 20)

        if method == RetrievalMethod.BM25:
            ranking = _bm25_search(query, depth)
        elif method == RetrievalMethod.DENSE:
            ranking = _dense_search(query, depth)
        else:
            from src.retrieve.hybrid import rrf_fuse_rankings

            ranking = rrf_fuse_rankings(
                [_bm25_search(query, depth), _dense_search(query, depth)],
                top_k=depth,
            )
            if method == RetrievalMethod.HYBRID_RERANK:
                from src.retrieve.rerank import rerank_run

                reranked = rerank_run(
                    run={SYNTHETIC_QID: ranking},
                    queries={SYNTHETIC_QID: query},
                    products=products,
                    model=state["reranker"],
                    rerank_depth=RERANK_DEPTH,
                    doc_strategy="name_desc",
                )
                ranking = reranked[SYNTHETIC_QID]

        results = []
        for rank, pid in enumerate(ranking[:top_k], start=1):
            product = products.get(pid)
            if product is None:
                continue
            results.append(
                {
                    "rank": rank,
                    "product_id": pid,
                    "name": getattr(product, "name", None),
                    "description": (getattr(product, "description", "") or "")[:400],
                }
            )

        return {
            "ok": True,
            "tool_version": TOOL_VERSION,
            "query": query,
            "method": method.value,
            "candidates_considered": len(ranking),
            "returned": len(results),
            "results": results,
        }
    except Exception as exc:  # noqa: BLE001 -- boundary: never surface a traceback
        return _error("retrieval_failed", f"{type(exc).__name__}: {exc}", method=method.value)


@mcp.tool()
def get_product(
    product_id: int = Field(..., ge=0, description="Product id from a search_products result."),
) -> dict[str, Any]:
    """Fetch the full record for one product id."""
    try:
        product = _load()["products"].get(product_id)
        if product is None:
            return _error("not_found", f"No product with id {product_id}.", product_id=product_id)
        return {
            "ok": True,
            "tool_version": TOOL_VERSION,
            "product": {
                "product_id": product_id,
                "name": getattr(product, "name", None),
                "description": getattr(product, "description", None),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return _error("lookup_failed", f"{type(exc).__name__}: {exc}")


@mcp.tool()
def evaluate_query(
    query_id: int = Field(..., ge=0, description="WANDS query id."),
    method: RetrievalMethod = Field(RetrievalMethod.HYBRID_RERANK, description="Config to evaluate."),
    k: int = Field(10, ge=1, le=MAX_TOP_K, description="Cutoff for metrics."),
) -> dict[str, Any]:
    """Score one WANDS query against human relevance judgments.

    This is the tool that makes reliability checkable rather than assumed: the
    caller can verify how the retrieval config it just used actually performs
    on labelled data, instead of taking the ranking on trust.
    """
    try:
        from evals.golden_set import load_golden_set  # ADAPT
        from evals.metrics import ndcg_at_k, recall_at_k  # ADAPT

        golden = load_golden_set()
        query_text = golden.queries.get(query_id)  # ADAPT
        if query_text is None:
            return _error("unknown_query_id", f"No WANDS query with id {query_id}.", query_id=query_id)

        search = search_products(query=query_text, top_k=k, method=method)
        if not search.get("ok"):
            return search
        retrieved = [r["product_id"] for r in search["results"]]

        judgments = golden.judgments.get(query_id, {})  # ADAPT: pid -> grade
        exact = {pid for pid, grade in judgments.items() if grade == 2}

        return {
            "ok": True,
            "tool_version": TOOL_VERSION,
            "query_id": query_id,
            "query": query_text,
            "method": method.value,
            "k": k,
            "metrics": {
                "recall_at_k": None if not exact else recall_at_k(retrieved, exact, k),
                "ndcg_at_k": ndcg_at_k(retrieved, judgments, k),
                "exact_judgments_available": len(exact),
            },
            "note": (
                "recall is undefined (null) when the query has no Exact judgment; "
                "nDCG uses graded gains Exact=2, Partial=1."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return _error("eval_failed", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    # stdout is the protocol channel; anything printed there corrupts it.
    print("hybrid-rag-bench MCP server starting (stdio)", file=sys.stderr)
    mcp.run(transport="stdio")
