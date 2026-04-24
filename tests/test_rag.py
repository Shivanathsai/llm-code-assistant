"""
tests/test_rag.py
=================
Unit tests for the RAG pipeline components.

Tests:
  - Chunking: correct chunk count and overlap
  - Ingestion: text stored in ChromaDB
  - Retrieval: vector search returns relevant results
  - Scoring: benchmark scoring function
  - Latency: retrieval under latency budget
  - 100K lines: ingestion scales to large corpora

Run: pytest tests/ -v
"""

import os
import sys
import time
import tempfile
import shutil

import pytest

sys.path.insert(0, ".")

# Set dummy API key for tests that don't call the LLM
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

from app.config import settings


@pytest.fixture(scope="module")
def temp_chroma_dir():
    """Temporary ChromaDB directory — cleaned up after tests."""
    d = tempfile.mkdtemp(prefix="test_chroma_")
    original = settings.chroma_persist_dir
    settings.__dict__["chroma_persist_dir"] = d
    yield d
    settings.__dict__["chroma_persist_dir"] = original
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def ingester(temp_chroma_dir):
    from app.rag.document_ingester import DocumentIngester
    ing = DocumentIngester()
    ing.reset()
    return ing


@pytest.fixture(scope="module")
def retriever(temp_chroma_dir, ingester):
    from app.rag.retriever import VectorRetriever
    return VectorRetriever()


# ── Chunking tests ────────────────────────────────────────────────────────────

class TestChunking:

    def test_short_text_produces_one_chunk(self, ingester):
        ingester.reset()
        text = "This is a short text with a few words."
        chunks = ingester._chunk_text(text, "test")
        assert len(chunks) >= 1

    def test_long_text_produces_multiple_chunks(self, ingester):
        text = " ".join(["word"] * 2000)  # 2000 words > chunk_size
        chunks = ingester._chunk_text(text, "test")
        assert len(chunks) > 1

    def test_chunk_overlap(self, ingester):
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = ingester._chunk_text(text, "test")
        # With overlap, adjacent chunks should share some words
        if len(chunks) > 1:
            words0 = set(chunks[0].text.split())
            words1 = set(chunks[1].text.split())
            assert len(words0 & words1) > 0, "Adjacent chunks must overlap"

    def test_chunk_has_required_fields(self, ingester):
        chunks = ingester._chunk_text("hello world test text", "src.py")
        for c in chunks:
            assert c.id
            assert c.text
            assert c.source == "src.py"
            assert c.chunk_idx >= 0

    def test_chunk_ids_are_unique(self, ingester):
        text = " ".join([f"word{i}" for i in range(2000)])
        chunks = ingester._chunk_text(text, "test")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "All chunk IDs must be unique"


# ── Ingestion tests ───────────────────────────────────────────────────────────

class TestIngestion:

    def test_ingest_text_returns_chunk_count(self, ingester):
        ingester.reset()
        n = ingester.ingest_text("Python is a programming language with many libraries.", "test")
        assert n >= 1

    def test_ingest_increases_count(self, ingester):
        ingester.reset()
        before = ingester.count()
        ingester.ingest_text("Additional documentation about distributed systems.", "test2")
        assert ingester.count() > before

    def test_ingest_file(self, ingester, tmp_path):
        ingester.reset()
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    '''Return greeting.'''\n    return 'hello'\n")
        n = ingester.ingest_file(str(f))
        assert n >= 1
        assert ingester.count() >= 1

    def test_reset_clears_store(self, ingester):
        ingester.ingest_text("Some text to store", "test")
        ingester.reset()
        assert ingester.count() == 0

    def test_ingest_100k_lines(self, ingester):
        """Resume claim: processes 100K+ documentation lines."""
        ingester.reset()
        # Generate 100K lines of text
        lines = [f"Line {i}: documentation about distributed systems and algorithms." for i in range(100_000)]
        text = "\n".join(lines)
        actual_lines = text.count("\n")
        assert actual_lines >= 99_000, f"Expected 100K lines, got {actual_lines}"

        t0 = time.perf_counter()
        chunks = ingester.ingest_text(text, "100k-test")
        elapsed = time.perf_counter() - t0

        assert chunks > 0, "Must produce at least one chunk"
        assert ingester.count() > 0
        print(f"\n    100K lines ingested: {chunks} chunks in {elapsed:.1f}s")


# ── Retrieval tests ───────────────────────────────────────────────────────────

class TestRetrieval:

    @pytest.fixture(autouse=True)
    def seed(self, ingester, retriever):
        ingester.reset()
        ingester.ingest_text(
            "Redis sorted sets provide O(log n) operations for rate limiting. "
            "ZADD adds elements with scores. ZRANGEBYSCORE retrieves by score range. "
            "Cosine similarity measures angle between vectors. "
            "Kafka topics are divided into partitions for parallel processing. "
            "Docker containers package applications with all dependencies.",
            source="seed"
        )
        # Re-init retriever to pick up new data
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        retriever.collection = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        ).get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("Redis rate limiting")
        assert len(results) >= 1

    def test_retrieve_results_have_scores(self, retriever):
        results = retriever.retrieve("sorted set operations")
        for r in results:
            assert 0 <= r.score <= 1.0

    def test_retrieve_results_sorted_by_score(self, retriever):
        results = retriever.retrieve("Kafka partitions")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_respects_top_k(self, retriever):
        results = retriever.retrieve("distributed systems", top_k=2)
        assert len(results) <= 2

    def test_retrieve_latency_under_budget(self, retriever):
        """Retrieval must be fast (< 500ms) as part of sub-2s total budget."""
        t0 = time.perf_counter()
        retriever.retrieve("vector search cosine similarity")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"Retrieval took {elapsed_ms:.0f}ms, expected < 500ms"

    def test_empty_store_returns_empty_list(self):
        import tempfile, shutil, chromadb
        from chromadb.config import Settings as ChromaSettings
        from app.rag.retriever import VectorRetriever

        d = tempfile.mkdtemp()
        settings.__dict__["chroma_persist_dir"] = d
        r = VectorRetriever()
        results = r.retrieve("anything")
        assert results == []
        shutil.rmtree(d)


# ── Scoring tests ─────────────────────────────────────────────────────────────

class TestBenchmarkScoring:

    def test_all_keywords_present_scores_1(self):
        from app.benchmark.benchmark import score_answer
        answer   = "The sliding window uses Redis sorted sets for O(log n) lookups"
        keywords = ["window", "Redis", "sorted", "log"]
        assert score_answer(answer, keywords) == 1.0

    def test_half_keywords_scores_0_5(self):
        from app.benchmark.benchmark import score_answer
        answer   = "The sliding window uses sorted sets"
        keywords = ["window", "sorted", "Redis", "timestamp"]
        score    = score_answer(answer, keywords)
        assert score == 0.5

    def test_no_keywords_scores_0(self):
        from app.benchmark.benchmark import score_answer
        answer   = "I don't know the answer to this question."
        keywords = ["Redis", "sorted", "log", "ZADD"]
        assert score_answer(answer, keywords) == 0.0

    def test_case_insensitive_matching(self):
        from app.benchmark.benchmark import score_answer
        answer   = "kafka streams processes events in real time"
        keywords = ["Kafka", "Streams", "events"]
        assert score_answer(answer, keywords) == 1.0
