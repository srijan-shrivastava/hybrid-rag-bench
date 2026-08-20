# hybrid-rag-bench

A retrieval ablation benchmark on WANDS (Wayfair product search), plus an MCP
server exposing the same retrieval stack as typed tools.

## Working in this repo

- Reported numbers come from `python -m evals.run_ablation`. Don't quote a
  figure that isn't in the README table or reproducible from that command.
- Negative results stay in. Parent-child chunking underperforming and
  unweighted RRF losing to dense alone are findings, not bugs to fix.
- BM25 and RRF are implemented explicitly rather than via a library flag, so
  every parameter that moves the numbers is visible. Keep it that way.
- Metrics return `None` when undefined for a query (no relevant products under
  the chosen mode). Never coerce that to `0` — it silently deflates aggregates.
- `bge-small-en-v1.5` is the default encoder so the benchmark runs with no API
  keys. Any change must preserve that.

## Using the MCP tools

The server exposes `warmup`, `search_products`, `get_product`, and
`evaluate_query`. Call `warmup` first — cold start is 30–60s while indexes
build and models load, and on a first call it can look like a hung search.

When answering product questions from this corpus:

- Every product mentioned must come from a `search_products` result. Cite as
  name (id: N). Never invent a product, id, price, material, or dimension.
- If a detail isn't in the output, call `get_product`. If it still isn't
  there, say the data doesn't contain it. Do not infer it from the product
  name — names and descriptions disagree often here. Product #25434 is named
  "waiting room chair with wood frame" but described as a salon chair.
- If search returns nothing relevant, say so. Don't fill the gap from general
  knowledge.
- If a tool returns `ok: false`, report the error code rather than answering
  as though it succeeded.
- Default to `method="hybrid_rerank"`. Use `bm25` or `dense` only when speed
  matters more than ranking quality.
- Use `evaluate_query` when the question is about retrieval quality rather
  than about products, so the answer rests on labelled data.

The corpus is a research dataset, not a storefront: no prices, stock, or
availability.
