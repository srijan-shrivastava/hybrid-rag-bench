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

Corpus, indexes and models load lazily on first tool call, not at import. MCP
clients start servers eagerly at boot; building a BM25 index and encoding 43k
chunks on import would stall client startup for every session, including ones
that never search. The cost doesn't vanish, it moves — see warmup().

Grounding rules are exposed as a *prompt*, not injected. An MCP server cannot
set its client's system prompt, and shouldn't be able to: a server connected
for retrieval has no business rewriting how the model behaves generally. The
server offers the rules; the user opts in.

Env vars
--------
HYBRID_RAG_BENCH_DATA        WANDS directory (default: "data")
HYBRID_RAG_BENCH_STRATEGY    chunk composition for the first stage
                             (default: "name_desc" — see INDEX_STRATEGY below)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from enum import Enum
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

TOOL_VERSION = "1.0.0"
MAX_TOP_K = 50
RERANK_DEPTH = 100
CHUNK_POOL = 300
SYNTHETIC_QID = 0

DATA_DIR = os.environ.get("HYBRID_RAG_BENCH_DATA", "data")

# Which composition the first stage indexes. name_desc is the balanced choice:
# it is never the worst composition for any of the four methods. If you only
# ever call hybrid_rerank, set this to name_only — that is the best row in the
# benchmark (nDCG@10 = 0.784), because once a cross-encoder reads the
# candidates the first stage only has to surface them, not order them.
INDEX_STRATEGY = os.environ.get("HYBRID_RAG_BENCH_STRATEGY", "name_desc")

# The cross-encoder scores name_desc text regardless of what the first stage
# indexed, matching the benchmark's fairness rule so rerank quality is not
# confounded with index composition.
RERANK_DOC_STRATEGY = "name_desc"

CORPUS_CAVEAT = (
    "Corpus is the WANDS research dataset (Wayfair product search), not a live "
    "storefront. It carries no prices, stock levels or availability."
)

_state: dict[str, Any] = {}
_lock = threading.Lock()


def _load() -> dict[str, Any]:
    """Build corpus, indexes and reranker once, on first use."""
    if "dataset" in _state:
        return _state
    with _lock:
        if "dataset" in _state:  # another thread won the race
            return _state

        from evals.golden_set import load_wands
        from src.ingest.chunking import STRATEGIES
        from src.retrieve.bm25 import BM25Index
        from src.retrieve.dense import DenseIndex, SentenceTransformerEncoder
        from src.retrieve.rerank import SentenceTransformersCrossEncoder

        if INDEX_STRATEGY not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {INDEX_STRATEGY!r}; available: {sorted(STRATEGIES)}"
            )

        dataset = load_wands(DATA_DIR)

        compose = STRATEGIES[INDEX_STRATEGY]
        chunks = [c for product in dataset.products.values() for c in compose(product)]

        _state["n_chunks"] = len(chunks)
        _state["bm25"] = BM25Index(chunks)
        _state["dense"] = DenseIndex.build(
            chunks,
            encoder=SentenceTransformerEncoder(),
            cache_key=INDEX_STRATEGY,  # shares the ablation's embedding cache
        )
        _state["reranker"] = SentenceTransformersCrossEncoder()
        _state["dataset"] = dataset  # set last: it is the "loaded" sentinel
    return _state


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


def _retrieve(query: str, method: RetrievalMethod, depth: int) -> list[int]:
    """Ranked product_ids for one query under the given method."""
    state = _load()

    def bm25() -> list[int]:
        return [pid for pid, _ in state["bm25"].search(query, top_k=depth, chunk_pool=CHUNK_POOL)]

    def dense() -> list[int]:
        return [pid for pid, _ in state["dense"].search(query, top_k=depth, chunk_pool=CHUNK_POOL)]

    if method is RetrievalMethod.BM25:
        return bm25()
    if method is RetrievalMethod.DENSE:
        return dense()

    from src.retrieve.hybrid import rrf_fuse_rankings

    fused = rrf_fuse_rankings([bm25(), dense()], top_k=depth)
    if method is RetrievalMethod.HYBRID:
        return fused

    from src.retrieve.rerank import rerank_run

    reranked = rerank_run(
        run={SYNTHETIC_QID: fused},
        queries={SYNTHETIC_QID: query},
        products=state["dataset"].products,
        model=state["reranker"],
        rerank_depth=RERANK_DEPTH,
        doc_strategy=RERANK_DOC_STRATEGY,
    )
    return reranked[SYNTHETIC_QID]


@mcp.tool()
def warmup() -> dict[str, Any]:
    """Load corpus, indexes and models. Call once before timing anything.

    Cold start is 30-60s: the WANDS TSVs are parsed, a BM25 index is built over
    ~43k chunks, embeddings are loaded or computed, and a cross-encoder is
    pulled into memory. Lazy loading keeps client startup fast but moves that
    cost onto whichever tool is called first, where it looks like a hung
    search and can exceed the client's tool timeout.

    This tool makes the cost explicit and attributable. Idempotent: subsequent
    calls return immediately.
    """
    already = "dataset" in _state
    started = time.perf_counter()
    try:
        state = _load()
    except Exception as exc:  # noqa: BLE001
        return _error("load_failed", f"{type(exc).__name__}: {exc}", data_dir=DATA_DIR)
    return {
        "ok": True,
        "tool_version": TOOL_VERSION,
        "already_loaded": already,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "index_strategy": INDEX_STRATEGY,
        "products": len(state["dataset"].products),
        "chunks_indexed": state["n_chunks"],
        "judged_queries": len(state["dataset"].judgments),
    }


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
    """Search the WANDS product corpus and return a ranked list.

    Read-only and idempotent: the same query with the same method returns the
    same ranking, so a client may safely retry. Results carry no price, stock
    or availability data — the corpus is a research dataset, not a storefront.
    First call may take 30-60s if warmup() has not been called.
    """
    query = query.strip()
    if not query:
        return _error("empty_query", "Query is empty after trimming whitespace.")

    try:
        depth = RERANK_DEPTH if method is RetrievalMethod.HYBRID_RERANK else max(top_k, 20)
        ranking = _retrieve(query, method, depth)
        products = _load()["dataset"].products

        results = []
        for rank, pid in enumerate(ranking[:top_k], start=1):
            product = products.get(pid)
            if product is None:
                continue
            results.append(
                {
                    "rank": rank,
                    "product_id": pid,
                    "name": product.name,
                    "product_class": product.product_class,
                    "description": product.description[:400],
                }
            )

        return {
            "ok": True,
            "tool_version": TOOL_VERSION,
            "query": query,
            "method": method.value,
            "index_strategy": INDEX_STRATEGY,
            "candidates_considered": len(ranking),
            "returned": len(results),
            "results": results,
            "caveat": CORPUS_CAVEAT,
        }
    except Exception as exc:  # noqa: BLE001 -- boundary: never surface a traceback
        return _error("retrieval_failed", f"{type(exc).__name__}: {exc}", method=method.value)


@mcp.tool()
def get_product(
    product_id: int = Field(..., ge=0, description="Product id from a search_products result."),
) -> dict[str, Any]:
    """Fetch the full record for one product id, including parsed features.

    Features come from the dataset's pipe-delimited attribute pairs and are
    incomplete for many products — an absent attribute means unrecorded, not
    absent from the physical product.
    """
    try:
        product = _load()["dataset"].products.get(product_id)
        if product is None:
            return _error("not_found", f"No product with id {product_id}.", product_id=product_id)
        return {
            "ok": True,
            "tool_version": TOOL_VERSION,
            "product": {
                "product_id": product.product_id,
                "name": product.name,
                "product_class": product.product_class,
                "category_hierarchy": product.category_hierarchy,
                "description": product.description,
                "features": dict(product.feature_pairs()),
                "average_rating": product.average_rating,
                "review_count": product.review_count,
            },
            "caveat": CORPUS_CAVEAT,
        }
    except Exception as exc:  # noqa: BLE001
        return _error("lookup_failed", f"{type(exc).__name__}: {exc}")


@mcp.tool()
def evaluate_query(
    query_id: int = Field(..., ge=0, description="WANDS query id."),
    method: RetrievalMethod = Field(RetrievalMethod.HYBRID_RERANK, description="Config to evaluate."),
    k: int = Field(10, ge=1, le=MAX_TOP_K, description="Cutoff for metrics."),
    strict: bool = Field(True, description="Strict relevance counts Exact only; lenient adds Partial."),
) -> dict[str, Any]:
    """Score one WANDS query against human relevance judgments.

    This is the tool that makes reliability checkable rather than assumed: the
    caller can verify how the retrieval config it just used actually performs
    on labelled data, instead of taking the ranking on trust.

    Judgments are pooled, so an unjudged product counts as irrelevant. Metrics
    are therefore a lower bound on true quality.
    """
    try:
        from evals.metrics import mrr, ndcg_at_k, recall_at_k

        judgments = _load()["dataset"].judgments
        judg = judgments.get(query_id)
        if judg is None:
            return _error("unknown_query_id", f"No WANDS query with id {query_id}.", query_id=query_id)

        ranking = _retrieve(judg.query, method, max(RERANK_DEPTH, k))

        return {
            "ok": True,
            "tool_version": TOOL_VERSION,
            "query_id": query_id,
            "query": judg.query,
            "query_class": judg.query_class,
            "method": method.value,
            "k": k,
            "strict": strict,
            "metrics": {
                "recall_at_k": recall_at_k(ranking, judg, k, strict=strict),
                "ndcg_at_k": ndcg_at_k(ranking, judg, k),
                "mrr_at_k": mrr(ranking, judg, k=k, strict=strict),
            },
            "judgments_available": {
                "relevant_under_mode": len(judg.relevant_ids(strict=strict)),
                "total_judged": len(judg.labels),
            },
            "note": (
                "null means the metric is undefined for this query (no relevant "
                "products under the chosen mode) — not zero. nDCG uses graded "
                "gains Exact=2, Partial=1."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return _error("eval_failed", f"{type(exc).__name__}: {exc}")


@mcp.prompt()
def grounded_product_search() -> str:
    """Grounding rules for answering product questions from this corpus.

    Exposed as a prompt rather than injected server-side: MCP servers cannot
    set a client's system prompt, and shouldn't be able to. The user opts in.
    """
    return """You answer product questions using only the hybrid-rag-bench tools.

Rules:
- Every product you mention must come from a search_products result in this
  conversation. Cite it as name (id: N). Never invent a product, id, price,
  material, or dimension.
- If a detail isn't in the tool output, call get_product for it. If it still
  isn't there, say the data doesn't contain it. Do not infer it from the
  product name — names and descriptions disagree often in this corpus.
- If search returns nothing relevant, say so and suggest a reformulation.
  Do not fill the gap from general knowledge.
- If a tool returns ok: false, report the error code. Do not answer as
  though the call succeeded.
- Default to method="hybrid_rerank". Use "bm25" or "dense" only when the
  user asks for speed.
- When the user is testing retrieval quality rather than shopping, use
  evaluate_query so the answer rests on labelled data."""


if __name__ == "__main__":
    # stdout is the protocol channel; anything printed there corrupts it.
    print(
        f"hybrid-rag-bench MCP server starting (stdio) "
        f"data={DATA_DIR} strategy={INDEX_STRATEGY}",
        file=sys.stderr,
    )
    mcp.run(transport="stdio")
