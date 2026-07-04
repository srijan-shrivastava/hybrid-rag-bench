"""Run the full retrieval ablation and emit the README markdown table.

    python -m evals.run_ablation                 # everything (needs sentence-transformers)
    python -m evals.run_ablation --skip-dense    # lexical-only (no model downloads)
    python -m evals.run_ablation --strategies full parent_child
    python -m evals.run_ablation --openai        # adds text-embedding-3-small rows

Configs produced per chunk strategy:
    bm25            sparse leg alone
    dense           bi-encoder leg alone (local model by default)
    hybrid          RRF fusion of the two
    hybrid+rerank   RRF fused candidates re-ordered by a cross-encoder
plus a single `oracle (ceiling)` row: judged items ranked by gain — the
maximum any retriever could score under pooled judgments.

Each run (query_id -> ranked pids) is cached as JSON under runs/ so
re-running the script only recomputes what's missing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evals.golden_set import load_wands
from evals.metrics import evaluate_run, format_results_row

KS = (5, 10, 20)
HEADER = (
    "| config | R@5 | R@10 | R@20 | nDCG@5 | nDCG@10 | nDCG@20 | MRR@10 |\n"
    "|---|---|---|---|---|---|---|---|"
)


def _cache_path(runs_dir: Path, name: str) -> Path:
    return runs_dir / (name.replace("/", "_").replace(" ", "") + ".json")


def _load_run(runs_dir: Path, name: str) -> dict[int, list[int]] | None:
    path = _cache_path(runs_dir, name)
    if path.exists():
        raw = json.loads(path.read_text())
        return {int(q): pids for q, pids in raw.items()}
    return None


def _save_run(runs_dir: Path, name: str, run: dict[int, list[int]]) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(runs_dir, name).write_text(json.dumps(run))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--runs-dir", default="runs", type=Path)
    parser.add_argument("--strategies", nargs="+",
                        default=["name_only", "name_desc", "full", "parent_child"])
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--openai", action="store_true",
                        help="also run dense with text-embedding-3-small (needs OPENAI_API_KEY)")
    parser.add_argument("--rerank-depth", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    print("loading WANDS ...")
    ds = load_wands(args.data)
    products = list(ds.products.values())
    queries = ds.queries

    from src.ingest.chunking import build_chunks
    from src.retrieve.bm25 import BM25Index
    from src.retrieve.bm25 import build_run as bm25_run
    from src.retrieve.hybrid import rrf_fuse_runs

    encoders = []
    reranker = None
    if not args.skip_dense:
        from src.retrieve.dense import SentenceTransformerEncoder
        encoders.append(SentenceTransformerEncoder())
        if args.openai:
            from src.retrieve.dense import OpenAIEncoder
            encoders.append(OpenAIEncoder())
    if not args.skip_rerank and not args.skip_dense:
        from src.retrieve.rerank import SentenceTransformersCrossEncoder
        reranker = SentenceTransformersCrossEncoder()

    rows: list[str] = []

    def evaluate_and_row(name: str, run: dict[int, list[int]]) -> None:
        res = evaluate_run(run, ds.judgments, ks=KS)
        rows.append(format_results_row(name, res, ks=KS))
        print(rows[-1])

    for strat in args.strategies:
        chunks = build_chunks(products, strat)

        name = f"bm25 / {strat}"
        run_bm25 = _load_run(args.runs_dir, name)
        if run_bm25 is None:
            t0 = time.time()
            run_bm25 = bm25_run(BM25Index(chunks), queries, top_k=args.top_k)
            print(f"[{name}] built in {time.time()-t0:.0f}s")
            _save_run(args.runs_dir, name, run_bm25)
        evaluate_and_row(name, run_bm25)

        for encoder in encoders:
            from src.retrieve.dense import DenseIndex
            from src.retrieve.dense import build_run as dense_run

            name = f"dense[{encoder.name}] / {strat}"
            run_dense = _load_run(args.runs_dir, name)
            if run_dense is None:
                t0 = time.time()
                index = DenseIndex.build(chunks, encoder, cache_key=strat)
                run_dense = dense_run(index, queries, top_k=args.top_k)
                print(f"[{name}] built in {time.time()-t0:.0f}s")
                _save_run(args.runs_dir, name, run_dense)
            evaluate_and_row(name, run_dense)

            name = f"hybrid[{encoder.name}] / {strat}"
            run_hybrid = _load_run(args.runs_dir, name)
            if run_hybrid is None:
                run_hybrid = rrf_fuse_runs([run_bm25, run_dense], top_k=args.top_k)
                _save_run(args.runs_dir, name, run_hybrid)
            evaluate_and_row(name, run_hybrid)

            if reranker is not None:
                from src.retrieve.rerank import rerank_run

                name = f"hybrid+rerank[{encoder.name}] / {strat}"
                run_rr = _load_run(args.runs_dir, name)
                if run_rr is None:
                    t0 = time.time()
                    run_rr = rerank_run(
                        run_hybrid, queries, ds.products, reranker,
                        rerank_depth=args.rerank_depth,
                    )
                    print(f"[{name}] reranked in {time.time()-t0:.0f}s")
                    _save_run(args.runs_dir, name, run_rr)
                evaluate_and_row(name, run_rr)

    oracle = {
        qid: [pid for pid, _ in sorted(j.labels.items(), key=lambda x: -x[1])]
        for qid, j in ds.judgments.items()
    }
    evaluate_and_row("oracle (ceiling)", oracle)

    print("\n\n===== README table =====\n")
    print(HEADER)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
