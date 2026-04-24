"""
main.py — FastAPI application
==============================
Exposes the RAG pipeline as a REST API.

Endpoints:
  POST /query           — ask a question, get RAG answer
  POST /ingest          — ingest text or file path
  GET  /stats           — vector store stats
  GET  /health          — liveness probe
  GET  /metrics         — Prometheus metrics
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.document_ingester import DocumentIngester
from app.rag.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
queries_total = Counter(
    "rag_queries_total", "Total RAG queries processed"
)
latency_histogram = Histogram(
    "rag_query_duration_seconds",
    "End-to-end RAG query latency",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
)
accuracy_gauge_value = 0.0  # updated by benchmark runs


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LLM Code Assistant")
    app.state.pipeline = RAGPipeline()
    app.state.ingester = DocumentIngester()
    logger.info("Pipeline ready — %d chunks in vector store",
                app.state.ingester.count())
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title="LLM-Powered Code Assistant",
    description=(
        "RAG-based code assistant using vector search + Groq LLM.\n\n"
        "**Architecture**: ChromaDB (HNSW cosine) + sentence-transformers + Groq\n"
        "**Latency target**: < 2s end-to-end\n"
        "**Accuracy target**: 92%+"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k:    Optional[int] = Field(None, ge=1, le=20)

    model_config = {"json_schema_extra": {
        "example": {
            "question": "How does the sliding window rate limiter work?",
            "top_k": 5,
        }
    }}


class QueryResponse(BaseModel):
    answer:        str
    sources:       List[str]
    chunks_used:   int
    retrieval_ms:  float
    generation_ms: float
    total_ms:      float
    meets_latency_target: bool
    model:         str


class IngestRequest(BaseModel):
    text:   Optional[str]  = None
    source: Optional[str]  = "api-input"
    paths:  Optional[List[str]] = None


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class StatsResponse(BaseModel):
    total_chunks:    int
    collection_name: str
    embedding_model: str
    llm_model:       str
    latency_target_s: float
    accuracy_target:  float


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Ask a question about the ingested codebase/documentation.
    Returns answer with source citations and latency breakdown.
    """
    pipeline = app.state.pipeline

    if pipeline.retriever.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty. Ingest documentation first via POST /ingest",
        )

    t0       = time.perf_counter()
    response = pipeline.query(request.question, top_k=request.top_k)
    latency  = time.perf_counter() - t0

    queries_total.inc()
    latency_histogram.observe(latency)

    return QueryResponse(
        answer               = response.answer,
        sources              = response.sources(),
        chunks_used          = response.chunks_used,
        retrieval_ms         = response.retrieval_ms,
        generation_ms        = response.generation_ms,
        total_ms             = response.total_ms,
        meets_latency_target = response.meets_latency_target,
        model                = response.model,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest text or files into the vector store."""
    ingester = app.state.ingester
    added    = 0

    if request.text:
        added += ingester.ingest_text(request.text, source=request.source or "api")

    if request.paths:
        summary = ingester.ingest_corpus(request.paths)
        added  += summary["chunks"]

    return IngestResponse(chunks_added=added, total_chunks=ingester.count())


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(
        total_chunks     = app.state.ingester.count(),
        collection_name  = settings.chroma_collection,
        embedding_model  = settings.embedding_model,
        llm_model        = settings.groq_model,
        latency_target_s = settings.max_latency_seconds,
        accuracy_target  = settings.min_accuracy,
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status":       "ok",
        "chunks":       app.state.ingester.count(),
        "model":        settings.groq_model,
    }


@app.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
