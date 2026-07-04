"""Dense (vector) retrieval over any chunk-composition strategy.

Design: encoder and backend are both pluggable and orthogonal.

Encoders (text -> unit-normalised vectors):
  * SentenceTransformerEncoder — local, free, reproducible. Default model
    BAAI/bge-small-en-v1.5 (384d). No API key; anyone can rerun the benchmark.
  * OpenAIEncoder — text-embedding-3-small (1536d), the production choice in
    many stacks. Optional: requires OPENAI_API_KEY. Corpus cost ~$0.20.
  Comparing the two IS one of the benchmark's findings (local vs API row pair).

Backends (store vectors, search by cosine):
  * NumpyBackend — in-memory brute force. At ~43k-123k chunks this is
    milliseconds per query and needs zero infrastructure. Default.
  * QdrantBackend — optional, mirrors a production deployment
    (docker-compose up -d qdrant). Same numbers, real infra.

Embeddings are cached to disk (.npy + chunk ids) keyed by
(strategy, encoder-name) so re-running ablations doesn't re-encode.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from src.ingest.chunking import Chunk, collapse_to_products


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class Encoder(Protocol):
    name: str
    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Local encoder via sentence-transformers. pip install sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = f"st:{model_name.split('/')[-1]}"
        self._model = SentenceTransformer(model_name, device=device)
        # bge models recommend a query instruction prefix; document side is raw.
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in model_name.lower()
            else ""
        )

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode([self.query_prefix + t for t in texts])


class OpenAIEncoder:
    """text-embedding-3-small via the OpenAI API. Requires OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI  # lazy import

        self.name = f"openai:{model}"
        self._model = model
        self._client = OpenAI()

    def encode(self, texts: Sequence[str], batch_size: int = 512) -> np.ndarray:
        out: list[list[float]] = []
        texts = [t if t.strip() else " " for t in texts]  # API rejects empty strings
        for i in range(0, len(texts), batch_size):
            resp = self._client.embeddings.create(model=self._model, input=texts[i : i + batch_size])
            out.extend(d.embedding for d in resp.data)
        vecs = np.asarray(out, dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
        return vecs

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(list(texts))


class HashEncoder:
    """Deterministic toy encoder for tests only — token-hash bag-of-words.

    Captures lexical overlap in vector form so retrieval mechanics can be
    verified end-to-end without model downloads. NOT a real embedding model;
    never use for reported numbers.
    """

    def __init__(self, dim: int = 256):
        self.name = f"hash:{dim}"
        self.dim = dim

    def encode(self, texts: Sequence[str], batch_size: int = 0) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
        return vecs

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class NumpyBackend:
    """Brute-force cosine over an in-memory matrix. Exact, zero infra."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None

    def index(self, vectors: np.ndarray) -> None:
        self._matrix = vectors

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        assert self._matrix is not None, "call index() first"
        scores = self._matrix @ query_vec  # unit vectors -> cosine
        top = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


class QdrantBackend:
    """Optional Qdrant backend (docker-compose up -d qdrant). Same interface."""

    def __init__(self, collection: str, url: str = "http://localhost:6333"):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        self._qm = qm
        self._client = QdrantClient(url=url)
        self._collection = collection

    def index(self, vectors: np.ndarray) -> None:
        qm = self._qm
        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=vectors.shape[1], distance=qm.Distance.COSINE),
        )
        batch = 1024
        for start in range(0, len(vectors), batch):
            chunk = vectors[start : start + batch]
            self._client.upsert(
                collection_name=self._collection,
                points=qm.Batch(
                    ids=list(range(start, start + len(chunk))),
                    vectors=chunk.tolist(),
                ),
            )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        hits = self._client.search(
            collection_name=self._collection, query_vector=query_vec.tolist(), limit=top_k
        )
        return [(int(h.id), float(h.score)) for h in hits]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

@dataclass
class DenseIndex:
    chunks: list[Chunk]
    encoder: Encoder
    backend: NumpyBackend | QdrantBackend

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        encoder: Encoder,
        backend: NumpyBackend | QdrantBackend | None = None,
        cache_dir: str | Path | None = "cache",
        cache_key: str | None = None,
    ) -> "DenseIndex":
        chunks = list(chunks)
        vectors = None
        cache_path = None
        if cache_dir is not None:
            safe = (cache_key or "chunks").replace("/", "_")
            enc = encoder.name.replace("/", "_").replace(":", "_")
            cache_path = Path(cache_dir) / f"{safe}__{enc}.npz"
            if cache_path.exists():
                data = np.load(cache_path, allow_pickle=False)
                if list(data["chunk_ids"]) == [c.chunk_id for c in chunks]:
                    vectors = data["vectors"]
        if vectors is None:
            vectors = encoder.encode([c.text for c in chunks])
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    vectors=vectors,
                    chunk_ids=np.array([c.chunk_id for c in chunks]),
                )
        backend = backend or NumpyBackend()
        backend.index(vectors)
        return cls(chunks=chunks, encoder=encoder, backend=backend)

    def search(self, query: str, top_k: int = 100, chunk_pool: int = 300) -> list[tuple[int, float]]:
        qvec = self.encoder.encode_queries([query])[0]
        scored_idx = self.backend.search(qvec, top_k=chunk_pool)
        scored = [(self.chunks[i], s) for i, s in scored_idx]
        return collapse_to_products(scored, limit=top_k)


def build_run(index: DenseIndex, queries: dict[int, str], top_k: int = 100) -> dict[int, list[int]]:
    # encode all queries in one batch, then search
    qids = list(queries)
    qvecs = index.encoder.encode_queries([queries[q] for q in qids])
    run: dict[int, list[int]] = {}
    for qid, qvec in zip(qids, qvecs):
        scored_idx = index.backend.search(qvec, top_k=300)
        scored = [(index.chunks[i], s) for i, s in scored_idx]
        run[qid] = [pid for pid, _ in collapse_to_products(scored, limit=top_k)]
    return run
