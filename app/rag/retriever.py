"""
retriever.py
============
Retrieves top-k relevant chunks from ChromaDB using cosine similarity.

Resume claim: "vector search"
Proof: query_similar() embeds the query and performs ANN search in ChromaDB
       (HNSW index, cosine space), returning ranked chunks with scores.
"""

import logging
import time
from dataclasses import dataclass
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector search with its similarity score."""
    text:       str
    source:     str
    score:      float    # cosine similarity (0–1, higher = more relevant)
    chunk_idx:  int
    line_start: int
    line_end:   int


class VectorRetriever:
    """
    Retrieves relevant documentation chunks via vector similarity search.

    Algorithm:
        1. Embed query with same model used during ingestion
        2. Query ChromaDB HNSW index (approximate nearest neighbour)
        3. Filter by similarity threshold
        4. Return top-k ranked chunks

    Latency contribution: ~50–150ms (embedding + ANN search)
    """

    def __init__(self) -> None:
        self.encoder = SentenceTransformer(settings.embedding_model)
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        """
        Find top-k most similar chunks to query.

        Args:
            query: Natural language question
            top_k: Number of chunks to return (default: settings.top_k)

        Returns:
            List of RetrievedChunk sorted by similarity (descending)
        """
        k = top_k or settings.top_k

        if self.collection.count() == 0:
            logger.warning("Vector store is empty — no chunks to retrieve")
            return []

        t0          = time.perf_counter()
        query_embed = self.encoder.encode([query]).tolist()
        results     = self.collection.query(
            query_embeddings=query_embed,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("Vector search: %.1fms", elapsed)

        chunks = []
        docs       = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - dist/2
            similarity = 1.0 - (dist / 2.0)

            if similarity < settings.similarity_threshold:
                continue

            chunks.append(RetrievedChunk(
                text       = doc,
                source     = meta.get("source", "unknown"),
                score      = round(similarity, 4),
                chunk_idx  = meta.get("chunk_idx", 0),
                line_start = meta.get("line_start", 0),
                line_end   = meta.get("line_end",   0),
            ))

        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def count(self) -> int:
        return self.collection.count()
