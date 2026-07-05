# hybrid-rag-bench

A reproducible ablation of hybrid retrieval on real e-commerce search: **BM25 vs dense vs RRF hybrid vs cross-encoder reranking**, crossed with four document-composition strategies, evaluated with graded-relevance metrics on the [WANDS](https://github.com/wayfair/WANDS) dataset (480 queries, ~43k products, 233k human relevance judgments).

Built to answer, with numbers instead of folklore: *which parts of a "modern RAG stack" actually pay for themselves?*

## Results

All configs retrieve top-100 products over the full corpus and are evaluated on all 480 queries. Recall/MRR use strict relevance (Exact only); nDCG uses graded gains (Exact=2, Partial=1). Dense model: `bge-small-en-v1.5` (local, reproducible without API keys). Reranker: `ms-marco-MiniLM-L-6-v2` over top-100 candidates.

| config | R@5 | R@10 | R@20 | nDCG@5 | nDCG@10 | nDCG@20 | MRR@10 |
|---|---|---|---|---|---|---|---|
| bm25 / name_only | 0.206 | 0.247 | 0.317 | 0.673 | 0.670 | 0.666 | 0.613 |
| dense / name_only | 0.205 | 0.261 | 0.342 | 0.749 | 0.742 | 0.733 | 0.672 |
| hybrid / name_only | 0.204 | 0.254 | 0.336 | 0.740 | 0.737 | 0.730 | 0.659 |
| **hybrid+rerank / name_only** | **0.241** | **0.308** | 0.383 | **0.792** | **0.784** | **0.775** | **0.776** |
| bm25 / name_desc | 0.205 | 0.263 | 0.341 | 0.700 | 0.692 | 0.681 | 0.667 |
| dense / name_desc | 0.202 | 0.273 | 0.353 | 0.738 | 0.730 | 0.723 | 0.701 |
| hybrid / name_desc | 0.215 | 0.284 | 0.373 | 0.759 | 0.751 | 0.741 | 0.711 |
| hybrid+rerank / name_desc | 0.238 | 0.308 | **0.397** | 0.766 | 0.759 | 0.750 | 0.767 |
| bm25 / full | 0.220 | 0.273 | 0.355 | 0.692 | 0.688 | 0.680 | 0.696 |
| dense / full | 0.220 | 0.281 | 0.367 | 0.750 | 0.742 | 0.729 | 0.706 |
| hybrid / full | 0.227 | 0.292 | 0.384 | 0.760 | 0.757 | 0.746 | 0.734 |
| hybrid+rerank / full | 0.236 | 0.306 | **0.397** | 0.765 | 0.760 | 0.752 | 0.764 |
| bm25 / parent_child | 0.219 | 0.279 | 0.351 | 0.691 | 0.689 | 0.684 | 0.676 |
| dense / parent_child | 0.220 | 0.275 | 0.360 | 0.729 | 0.721 | 0.707 | 0.711 |
| hybrid / parent_child | 0.221 | 0.285 | 0.382 | 0.752 | 0.751 | 0.745 | 0.712 |
| hybrid+rerank / parent_child | 0.240 | 0.307 | 0.395 | 0.769 | 0.759 | 0.752 | 0.768 |
| *oracle (ceiling)* | *0.415* | *0.514* | *0.634* | *1.000* | *1.000* | *1.000* | *1.000* |

**Reading the table — the oracle row matters.** WANDS judgments are pooled, and many queries have more than 10 Exact matches, so even a perfect retriever caps at R@10 = 0.514. The best real config (0.308) therefore achieves ~60% of *achievable* recall@10, not 31% of a naive 1.0. Benchmarks that omit this ceiling overstate how far from "solved" they are.

## Findings

**1. The cross-encoder reranker is the single highest-value component.** It gives the best nDCG@10 in every composition and lifts MRR@10 to ~0.77 across the board (+0.05–0.12 over its input ranking). If you add exactly one "advanced" stage to BM25+dense, the evidence says: make it the reranker.

**2. With a reranker on top, index composition stops mattering — the cheapest index wins.** The best overall number in the table is `hybrid+rerank / name_only` (nDCG@10 = 0.784): a first stage indexing *only product names* beats richer compositions once a cross-encoder reads the candidates. The first stage only has to get the right products into the top-100; the reranker does the ordering with full attention over query and text. Practical implication: spend your latency/complexity budget on reranking before you spend it on elaborate indexing.

**3. Dense and BM25 find similar products; dense orders them better.** Recall@10 is nearly tied leg-for-leg, but dense wins nDCG by +0.03–0.07 everywhere. On short, keyword-ish e-commerce queries, lexical matching finds candidates fine — semantic similarity mainly improves *ranking*.

**4. Unweighted hybrid fusion is not a free win.** RRF(bm25, dense) beats both legs on name_desc/full/parent_child, but on name_only it *loses* to dense alone (0.737 vs 0.742 nDCG@10): when one leg is clearly stronger, unweighted fusion drags it toward the weaker one. "Hybrid always helps" is folklore; whether it helps depends on the quality gap between legs. (Weighted RRF is the obvious follow-up — `weights` is already a parameter in `src/retrieve/hybrid.py`.)

**5. Parent-child chunking did not help here — and that's a finding, not a failure.** The hypothesis was that splitting products into focused chunks (identity / description / features) would help the dense leg by avoiding embedding dilution. The opposite happened: dense/parent_child (0.721) trails dense/full (0.742). Explanation: these documents already fit comfortably in the encoder's 512-token window, so splitting only fragments context. Parent-child earns its complexity on *long* documents; product cards aren't long. Retrieval technique value is corpus-dependent — measure before adopting.

**Why lexical search misses products at all:** WANDS product #25434 is *named* "waiting room chair with wood frame" but *described* as "a salon chair, barber chair for a hairstylist." For the query "salon chair" (where it is an Exact match), a name-only lexical index can never find it — the matching term exists only in the description. Composition and semantic matching exist to close exactly this gap.

## Run it yourself

```bash
git clone https://github.com/srijan-shrivastava/hybrid-rag-bench && cd hybrid-rag-bench
pip install numpy sentence-transformers
python -m data.download        # fetches WANDS (~96MB) and verifies row counts
python -m evals.run_ablation   # full table; ~30 min on a T4 GPU, longer on CPU
```

Useful flags: `--skip-dense` (lexical-only, no model downloads), `--strategies name_desc parent_child` (subset), `--openai` (adds `text-embedding-3-small` rows; needs `OPENAI_API_KEY`). Embeddings cache under `cache/`, computed rankings under `runs/` — reruns only compute what's missing.

## What's in the box

```
data/download.py          fetch + verify WANDS (row counts checked against published figures)
evals/golden_set.py       loader; dedups 1,467 duplicate judgment pairs (14 conflicting)
evals/metrics.py          recall@k, MRR, nDCG@k with graded gains; per-query undefined-metric handling
evals/run_ablation.py     one command -> the table above
src/ingest/chunking.py    4 document-composition strategies; chunk->product max-score collapse
src/retrieve/bm25.py      explicit Okapi BM25 (k1=1.5, b=0.75, stated tokenizer — no library defaults)
src/retrieve/dense.py     pluggable encoders (local bge / OpenAI) x backends (numpy brute-force / Qdrant)
src/retrieve/hybrid.py    Reciprocal Rank Fusion, ~30 explicit lines
src/retrieve/rerank.py    cross-encoder rerank of top-N; tail preserved so recall isn't silently cut
```

Deliberate choices: BM25 and RRF are implemented explicitly rather than through a search engine's hybrid flag, so every parameter that moves the numbers is visible in this repo. The dense backend defaults to brute-force numpy (exact, zero infra at 43k docs; ANN indexes like Qdrant/HNSW become necessary at ~1M+ vectors or when you need filtering + updates — a `QdrantBackend` is included for that shape).

## Methodology notes

- **Labels:** Exact=2 / Partial=1 / Irrelevant=0. Partial dominates (~63% of judgments), so binary metrics (recall, MRR) use strict Exact-only relevance; nDCG uses the full graded scale. 101/480 queries have no Exact judgment — strict metrics are undefined there and those queries are skipped in those aggregates (not counted as zero).
- **Pooled judgments:** unjudged (query, product) pairs count as irrelevant, which slightly penalizes retrievers that surface good-but-unjudged products. Standard assumption; stated rather than hidden.
- **Dedup:** label.csv contains 233,448 rows but 231,873 unique (query, product) pairs; 14 duplicates conflict and are resolved last-write-wins.
- **Fairness:** every config retrieves from the identical corpus with the identical chunk text per strategy; the reranker reads `name_desc` text regardless of what the first stage indexed, so rerank quality isn't confounded with index composition.

## Roadmap

- **v1.1** — LLM listwise re-ranking as a final stage (quality vs cost per 1k queries); LLM intent extraction + category pre-filtering (does predicted-category filtering help precision more than it hurts recall?)
- **v1.2** — semantic query cache with metadata gating: hit-rate vs false-hit-rate threshold sweep on paraphrased query variants

## License

MIT. WANDS is © Wayfair, MIT-licensed, fetched from the official repository rather than redistributed.
