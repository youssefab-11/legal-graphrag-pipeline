"""Centralized application configuration.

Loads environment variables from .env file and exposes typed settings
for the entire Legal GraphRAG pipeline.

This configuration is optimized for a fully local, free setup using:
- Ollama for LLM inference (llama3.1:8b)
- Ollama for embeddings (nomic-embed-text)
- Neo4j Community Edition as the graph + vector store
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env at project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class Settings:
    """Application settings loaded from environment variables."""

    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Ollama Configuration (Local LLM & Embeddings)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:14b")
    OLLAMA_FALLBACK_LLM_MODEL: str = os.getenv(
        "OLLAMA_FALLBACK_LLM_MODEL", "qwen2.5:7b"
    )
    OLLAMA_EMBEDDING_MODEL: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b"
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "2560"))

    # Scraping Configuration
    QANOON_BASE_URL: str = os.getenv("QANOON_BASE_URL", "https://qanoon.om")
    REQUEST_DELAY_MIN: float = float(os.getenv("REQUEST_DELAY_MIN", "0.2"))
    REQUEST_DELAY_MAX: float = float(os.getenv("REQUEST_DELAY_MAX", "0.7"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))
    USER_AGENT: str = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    )

    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Search Configuration
    TOP_K_CANDIDATES: int = int(os.getenv("TOP_K_CANDIDATES", "15"))
    TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "5"))
    TOPIC_MERGE_THRESHOLD: float = float(os.getenv("TOPIC_MERGE_THRESHOLD", "0.88"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Derived paths
    @property
    def DATA_DIR(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def RAW_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def CHECKPOINT_DIR(self) -> Path:
        return self.DATA_DIR / "checkpoints"

    @property
    def SAMPLE_OUTPUT_DIR(self) -> Path:
        return self.DATA_DIR / "sample_output"

    def ensure_directories(self) -> None:
        """Create required data directories if they don't exist."""
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Factory function returning Settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Global settings instance
settings = get_settings()
