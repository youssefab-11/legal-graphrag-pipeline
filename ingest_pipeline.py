"""End-to-end ingestion pipeline for a batch of uningested documents.

Runs all steps automatically:
  1. Build batch of next N uningested documents
  2. Ingest documents into Neo4j
  3. Extract AMENDS/REPEALS relationships
  4. Extract topics via LLM
  5. Chunk documents
  6. Generate embeddings
  7. Merge duplicate topics
  8. Re-run community detection

Usage:
    python ingest_pipeline.py --batch-size 200
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str) -> None:
    """Run a subprocess command and stream output."""
    logger.info("Starting: %s", description)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error("Failed: %s (exit code %d)", description, result.returncode)
        sys.exit(result.returncode)
    logger.info("Completed: %s", description)


def build_batch(batch_size: int, output_path: Path) -> int:
    """Create a JSON batch file with the next N uningested documents."""
    client = get_neo4j_client()
    driver = client.connect()
    ingested_ids = set()
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        result = session.run("MATCH (d:Document) RETURN d.id AS id")
        for record in result:
            ingested_ids.add(record["id"])

    raw_dir = Path("data/raw")
    all_raw = sorted(raw_dir.glob("*.json"))

    batch_paths = []
    for f in all_raw:
        if f.stem not in ingested_ids:
            batch_paths.append(f)
        if len(batch_paths) >= batch_size:
            break

    docs = []
    for p in batch_paths:
        with open(p, "r", encoding="utf-8") as f:
            docs.append(json.load(f))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    logger.info("Wrote %d documents to %s", len(docs), output_path)
    return len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end ingestion pipeline.")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of documents to ingest")
    parser.add_argument("--batch-file", type=str, default="data/ingest_200_next.json", help="Batch file path")
    args = parser.parse_args()

    batch_path = Path(args.batch_file)

    # 1. Build batch
    count = build_batch(args.batch_size, batch_path)
    if count == 0:
        logger.info("No uningested documents found. Exiting.")
        return

    batch_arg = str(batch_path)

    # 2. Run ingestion pipeline steps
    steps = [
        ([sys.executable, "-m", "src.ingestion.document_builder", batch_arg], "Document builder"),
        ([sys.executable, "-m", "src.ingestion.relationship_extractor", batch_arg], "Relationship extractor"),
        ([sys.executable, "-m", "src.llm_agents.topic_extractor", batch_arg], "Topic extractor"),
        ([sys.executable, "-m", "src.llm_agents.chunker", batch_arg], "Chunker"),
        ([sys.executable, "-m", "src.vector_ops.embedder"], "Embedder"),
        ([sys.executable, "-m", "src.vector_ops.topic_merger"], "Topic merger"),
        ([sys.executable, "-m", "src.vector_ops.community_detector"], "Community detector"),
    ]

    for cmd, desc in steps:
        run_command(cmd, desc)

    logger.info("Pipeline complete. Ingested %d documents.", count)


if __name__ == "__main__":
    main()
