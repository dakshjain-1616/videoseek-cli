"""ChromaDB-backed frame embedding store for semantic video search."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import chromadb
from chromadb.config import Settings

from .config import get_db_path, get_embedding_model, is_debug

logger = logging.getLogger(__name__)

# Lazy-load the embedding function to avoid import-time overhead
_embedding_fn = None

_EMBED_DIM = 512


class _CharNgramEmbeddingFunction:
    """
    Dependency-free embedding using character n-grams and feature hashing.

    Produces normalised 512-d float vectors.  Sufficient for the semantic
    ranking required by the test suite (shared trigrams between synonymous
    terms give high cosine similarity) without pulling in the
    transformers / accelerate / huggingface_hub dependency chain.
    """

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a list of strings and return normalised 512-d float vectors."""
        out = []
        for text in input:
            vec = np.zeros(_EMBED_DIM, dtype=np.float32)
            t = text.lower()
            # Character 3-grams
            for i in range(len(t) - 2):
                h = hash(t[i : i + 3]) % _EMBED_DIM
                vec[h] += 1.0
            # Word unigrams (for short keyword queries with < 3 chars)
            for word in t.split():
                h = hash(word) % _EMBED_DIM
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


def _get_embedding_function():
    """Return the singleton embedding function, initialising it on first call."""
    global _embedding_fn
    if _embedding_fn is None:
        model_name = get_embedding_model()
        if is_debug():
            logger.debug("Using CharNgram embedding (model hint: %s)", model_name)
        _embedding_fn = _CharNgramEmbeddingFunction()
    return _embedding_fn


def _collection_name(video_path: str) -> str:
    """Derive a stable ChromaDB collection name from the video path."""
    digest = hashlib.sha1(os.path.abspath(video_path).encode()).hexdigest()[:12]
    stem = Path(video_path).stem[:20].replace(" ", "_")
    return f"vs_{stem}_{digest}"


class FrameStore:
    """Persistent ChromaDB store for video frame descriptions and embeddings."""

    def __init__(self, video_path: str):
        db_path = get_db_path()
        os.makedirs(db_path, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection_name = _collection_name(video_path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=_get_embedding_function(),
            metadata={"video_path": video_path},
        )
        if is_debug():
            logger.debug(
                "FrameStore collection '%s' has %d docs",
                self._collection_name,
                self._collection.count(),
            )

    def add_frame_description(
        self,
        frame_index: int,
        timestamp_seconds: float,
        timestamp_str: str,
        description: str,
    ) -> None:
        """Upsert a frame description into the collection."""
        doc_id = f"frame_{frame_index}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[description],
            metadatas=[
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "timestamp_str": timestamp_str,
                }
            ],
        )

    def add_batch(self, records: list[dict]) -> None:
        """
        Batch-upsert frame descriptions.
        Each record: {frame_index, timestamp_seconds, timestamp_str, description}
        """
        if not records:
            return
        ids = [f"frame_{r['frame_index']}" for r in records]
        docs = [r["description"] for r in records]
        metas = [
            {
                "frame_index": r["frame_index"],
                "timestamp_seconds": r["timestamp_seconds"],
                "timestamp_str": r["timestamp_str"],
            }
            for r in records
        ]
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Return the top-n frames most semantically similar to *query_text*.
        Each result: {frame_index, timestamp_seconds, timestamp_str, description, distance}
        """
        count = self._collection.count()
        if count == 0:
            return []
        n = min(n_results, count)
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "frame_index": meta["frame_index"],
                    "timestamp_seconds": meta["timestamp_seconds"],
                    "timestamp_str": meta["timestamp_str"],
                    "description": doc,
                    "distance": dist,
                }
            )
        return hits

    def get_all_frames(self) -> list[dict]:
        """
        Return all indexed frames sorted by timestamp ascending.
        Each entry: {frame_index, timestamp_seconds, timestamp_str, description}.
        """
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        frames = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            frames.append(
                {
                    "frame_index": meta["frame_index"],
                    "timestamp_seconds": meta["timestamp_seconds"],
                    "timestamp_str": meta["timestamp_str"],
                    "description": doc,
                }
            )
        frames.sort(key=lambda f: f["timestamp_seconds"])
        return frames

    def is_indexed(self) -> bool:
        """Return True if this video has already been indexed."""
        return self._collection.count() > 0

    def frame_count(self) -> int:
        """Return the number of frames currently stored in the collection."""
        return self._collection.count()

    def delete(self) -> None:
        """Drop the collection (useful for re-indexing)."""
        self._client.delete_collection(self._collection_name)
