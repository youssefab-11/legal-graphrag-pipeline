"""Qanoon.om scraping and ingestion runner.

Scrapes real Omani legal documents from qanoon.om (Arabic) and decree.om
(English), then ingests them into the GraphRAG pipeline.

Usage:
    # Scrape a small sample (default 10 docs)
    python scrape_qanoon.py

    # Scrape up to 100 documents
    python scrape_qanoon.py --max-documents 100

    # Full database coverage (auto-discovers all listing pages)
    python scrape_qanoon.py --all-docs

    # Only scrape, do not run ingestion
    python scrape_qanoon.py --scrape-only --all-docs
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from src.config.settings import settings
from src.ingestion.document_builder import DocumentBuilder
from src.ingestion.neo4j_client import get_neo4j_client
from src.ingestion.relationship_extractor import RelationshipExtractor
from src.llm_agents.chunker import DocumentChunker
from src.llm_agents.topic_extractor import TopicExtractor
from src.scraper.qanoon_scraper import QanoonScraper
from src.vector_ops.embedder import Embedder

logger = logging.getLogger(__name__)


def run_scrape(
    max_documents: int = 10,
    max_pages: Optional[int] = None,
    all_docs: bool = False,
    scrape_only: bool = False,
    max_workers: int = 5,
) -> List[dict]:
    """Scrape qanoon.om and optionally run full ingestion pipeline."""
    logging.basicConfig(level=settings.LOG_LEVEL)

    if all_docs:
        logger.info("Full-coverage mode enabled. Auto-detecting all listing pages...")

    # 1. Scrape documents
    logger.info("Starting qanoon.om scraper with %d workers...", max_workers)
    scraper = QanoonScraper(
        max_documents=None if all_docs else max_documents,
        delay_min=settings.REQUEST_DELAY_MIN,
        delay_max=settings.REQUEST_DELAY_MAX,
        max_workers=max_workers,
    )
    docs = scraper.run(max_pages=max_pages, all_docs=all_docs)

    if not docs:
        logger.warning("No documents scraped. Exiting.")
        return []

    if scrape_only:
        logger.info("Scrape-only mode: skipping ingestion. Scraped %d documents.", len(docs))
        return docs

    # 2. Setup Neo4j schema
    logger.info("Setting up Neo4j schema...")
    neo4j_client = get_neo4j_client()
    neo4j_client.setup_schema()

    # 3. Ingest into Neo4j
    logger.info("Ingesting %d documents into Neo4j...", len(docs))
    builder = DocumentBuilder()
    builder.ingest_documents(docs)

    # 4. Extract relationships
    logger.info("Extracting AMENDS/REPEALS relationships...")
    rel_extractor = RelationshipExtractor()
    rel_extractor.process_documents(docs)

    # 5. Extract topics
    logger.info("Extracting topics with LLM...")
    topic_extractor = TopicExtractor()
    topic_extractor.process_documents(docs)

    # 6. Chunk documents
    logger.info("Chunking documents...")
    chunker = DocumentChunker()
    chunker.process_documents(docs)

    # 7. Generate embeddings
    logger.info("Generating embeddings...")
    embedder = Embedder()
    embedder.process_all()

    logger.info("Scrape and ingestion pipeline complete!")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape qanoon.om and ingest into GraphRAG pipeline")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=10,
        help="Maximum number of documents to scrape (default: 10). Ignored when --all-docs is set.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of listing pages to crawl. Auto-detect when --all-docs is set.",
    )
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help="Scrape every document discovered on qanoon.om (ignores --max-documents).",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape documents; skip Neo4j ingestion and enrichment.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of concurrent scraping workers (default: 5).",
    )
    args = parser.parse_args()

    run_scrape(
        max_documents=args.max_documents,
        max_pages=args.max_pages,
        all_docs=args.all_docs,
        scrape_only=args.scrape_only,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
