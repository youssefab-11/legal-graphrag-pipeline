"""End-to-end pipeline runner.

Orchestrates sample generation, ingestion, topic extraction, chunking,
embedding, and optional topic merging.

Usage:
    python run_pipeline.py --sample-count 50
"""

import argparse
import logging
from pathlib import Path

from src.config.settings import settings
from src.ingestion.document_builder import DocumentBuilder
from src.ingestion.neo4j_client import get_neo4j_client
from src.ingestion.relationship_extractor import RelationshipExtractor
from src.llm_agents.chunker import DocumentChunker
from src.llm_agents.topic_extractor import TopicExtractor
from src.scraper.sample_generator import SampleGenerator
from src.vector_ops.embedder import Embedder
from src.vector_ops.topic_merger import TopicMerger
from src.vector_ops.index_manager import VectorIndexManager

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_pipeline(sample_count: int = 50, merge_topics: bool = False) -> None:
    """Run the complete pipeline on sample data."""
    setup_logging()

    # 1. Setup Neo4j schema and indexes
    logger.info("=" * 60)
    logger.info("Step 1: Setting up Neo4j schema")
    logger.info("=" * 60)
    neo4j_client = get_neo4j_client()
    neo4j_client.setup_schema()

    # 2. Generate sample documents
    logger.info("=" * 60)
    logger.info("Step 2: Generating %d sample documents", sample_count)
    logger.info("=" * 60)
    generator = SampleGenerator()
    docs = generator.generate(count=sample_count)

    sample_file = settings.SAMPLE_OUTPUT_DIR / "sample_documents.json"

    # 3. Ingest documents into Neo4j
    logger.info("=" * 60)
    logger.info("Step 3: Ingesting documents into Neo4j")
    logger.info("=" * 60)
    builder = DocumentBuilder()
    builder.ingest_documents(docs)

    # 4. Extract cross-reference relationships
    logger.info("=" * 60)
    logger.info("Step 4: Extracting AMENDS / REPEALS relationships")
    logger.info("=" * 60)
    rel_extractor = RelationshipExtractor()
    rel_extractor.process_documents(docs)

    # 5. Extract topics with LLM
    logger.info("=" * 60)
    logger.info("Step 5: Extracting topics with local LLM")
    logger.info("=" * 60)
    topic_extractor = TopicExtractor()
    topic_extractor.process_documents(docs)

    # 6. Chunk documents
    logger.info("=" * 60)
    logger.info("Step 6: Chunking documents")
    logger.info("=" * 60)
    chunker = DocumentChunker()
    chunker.process_documents(docs)

    # 7. Generate embeddings
    logger.info("=" * 60)
    logger.info("Step 7: Generating embeddings")
    logger.info("=" * 60)
    embedder = Embedder()
    embedder.process_all()

    # 8. (Optional) Merge similar topics
    if merge_topics:
        logger.info("=" * 60)
        logger.info("Step 8: Merging similar topics")
        logger.info("=" * 60)
        merger = TopicMerger()
        merger.run()

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("Sample data: %s", sample_file)
    logger.info("Run the search client: python -m src.search.search_client")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal GraphRAG Pipeline Runner")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=50,
        help="Number of sample documents to generate (default: 50)",
    )
    parser.add_argument(
        "--merge-topics",
        action="store_true",
        help="Enable topic merging after extraction",
    )
    args = parser.parse_args()

    run_pipeline(sample_count=args.sample_count, merge_topics=args.merge_topics)


if __name__ == "__main__":
    main()
