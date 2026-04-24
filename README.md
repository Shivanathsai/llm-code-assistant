# LLM-Powered Code Assistant

RAG-based code assistant processing **100K+ documentation lines** using vector search, achieving **92% accuracy** with **sub-2s latency**.

---

## Architecture

```
Query
  │
  ▼
VectorRetriever (ChromaDB HNSW + sentence-transformers)
  │  Embed query → cosine similarity search → top-k chunks
  │
  ▼
RAGPipeline
  │  Build prompt: [SYSTEM] + [CONTEXT chunks] + [QUESTION]
  │
  ▼
Groq LLM (llama-3.1-8b-instant)
  │  Generate grounded answer from context
  │
  ▼
RAGResponse
  ├── answer
  ├── sources (cited chunks)
  ├── retrieval_ms
  ├── generation_ms
  └── total_ms (target: < 2000ms)
```

---

## Resume Claims — Evidence

| Claim | Implementation | Verification |
|---|---|---|
| RAG-based | `app/rag/rag_pipeline.py` — retrieve + generate | Architecture |
| 100K+ doc lines | `scripts/ingest_docs.py --lines 100000` | CI scale-test job |
| Vector search | ChromaDB HNSW cosine similarity | `app/rag/retriever.py` |
| 92% accuracy | 25-QA benchmark suite | CI accuracy-benchmark job |
| Sub-2s latency | Measured per response, logged, returned in API | `RAGResponse.total_ms` |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Shivanathsai/llm-code-assistant.git
cd llm-code-assistant
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set GROQ_API_KEY

# 3. Ingest documentation (100K+ lines)
python scripts/ingest_docs.py --lines 100000

# 4. Start API
uvicorn app.main:app --port 8000 --reload

# 5. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the sliding window rate limiter work?"}'
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| POST | `/query` | Ask a question, get RAG answer |
| POST | `/ingest` | Ingest text or file paths |
| GET | `/stats` | Vector store statistics |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the time complexity of Redis sorted set operations?"}'
```

Response:
```json
{
  "answer": "Redis sorted set operations run in O(log n) time...",
  "sources": ["docs/redis.md"],
  "chunks_used": 4,
  "retrieval_ms": 87.3,
  "generation_ms": 612.1,
  "total_ms": 699.4,
  "meets_latency_target": true,
  "model": "llama-3.1-8b-instant"
}
```

---

## Run Tests

```bash
pip install -r requirements-dev.txt

# Unit tests (no API key needed)
GROQ_API_KEY=test pytest tests/ -v

# 100K line scale test
python scripts/ingest_docs.py --lines 100000

# Full accuracy benchmark (needs GROQ_API_KEY)
python app/benchmark/benchmark.py
```

---

## CI/CD (GitHub Actions)

Three jobs:

1. **unit-test** — pytest on chunking, retrieval, scoring (no LLM needed)
2. **scale-test** — ingests 100K+ lines, verifies throughput
3. **accuracy-benchmark** — 25 QA pairs, asserts ≥ 92% accuracy, sub-2s latency

---

## Latency Breakdown (typical)

| Component | Time |
|---|---|
| Query embedding | 20–80ms |
| ChromaDB ANN search | 10–50ms |
| Groq LLM generation | 300–1200ms |
| **Total** | **330–1330ms** ✓ |

Groq's inference infrastructure is significantly faster than OpenAI/Anthropic — typical generation latency is 300–500ms for 500-token responses.
