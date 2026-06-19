"""Qanoon.om scraping and ingestion runner.

Scrapes real Omani legal documents from qanoon.om (Arabic) and decree.om
(English), then ingests them into the GraphRAG pipeline.

Usage:
    python scrape_qanoon.py --max-documents 10
"""

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import List

from src.config.settings import settings
from src.ingestion.document_builder import DocumentBuilder
from src.ingestion.neo4j_client import get_neo4j_client
from src.ingestion.relationship_extractor import RelationshipExtractor
from src.llm_agents.chunker import DocumentChunker
from src.llm_agents.topic_extractor import TopicExtractor
from src.scraper.qanoon_scraper import QanoonScraper
from src.vector_ops.embedder import Embedder

logger = logging.getLogger(__name__)


def run_scrape_and_ingest(max_documents: int = 10, max_pages: int = 5) -> List[dict]:
    """Scrape qanoon.om and run full ingestion pipeline."""
    logging.basicConfig(level=settings.LOG_LEVEL)

    # 1. Setup Neo4j schema
    logger.info("Setting up Neo4j schema...")
    neo4j_client = get_neo4j_client()
    neo4j_client.setup_schema()

    # 2. Scrape documents
    logger.info("Scraping up to %d documents from qanoon.om (max %d pages)...", max_documents, max_pages)
    scraper = QanoonScraper(max_documents=max_documents)
    docs = scraper.run(max_pages=max_pages)

    if not docs:
        logger.warning("No documents scraped. Exiting.")
        return []

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
        help="Maximum number of documents to scrape (default: 10)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum number of listing pages to crawl (default: 5)",
    )
    args = parser.parse_args()

    run_scrape_and_ingest(max_documents=args.max_documents, max_pages=args.max_pages)


if __name__ == "__main__":
    main()

