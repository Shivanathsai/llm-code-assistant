"""
Configuration — loaded from environment variables.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Groq LLM
    groq_api_key: str = ""
    groq_model:   str = "llama-3.1-8b-instant"  # fast + accurate for RAG

    # Embeddings (local, no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"   # 384-dim, fast, accurate

    # ChromaDB vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection:  str = "code-docs"

    # RAG parameters
    chunk_size:       int   = 512    # tokens per chunk
    chunk_overlap:    int   = 64     # overlap between chunks
    top_k:            int   = 5      # chunks retrieved per query
    similarity_threshold: float = 0.3

    # Latency target (resume claim: sub-2s)
    max_latency_seconds: float = 2.0

    # Accuracy target (resume claim: 92%)
    min_accuracy: float = 0.92

    # Server
    app_port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
