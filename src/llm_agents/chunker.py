"""Semantic chunking for legal documents.

Splits document markdown content into coherent text chunks and creates
Chunk nodes linked back to parent Document nodes.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j.exceptions import Neo4jError

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunks legal documents and persists Chunk nodes in Neo4j."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.neo4j = get_neo4j_client()

    def chunk_text(self, text: str) -> List[str]:
        """Split text into semantic chunks.

        Args:
            text: Markdown text.

        Returns:
            List of chunk strings.
        """
        if not text or len(text.strip()) < 50:
            return []
        return self.splitter.split_text(text)

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate chunks for all language versions of a document.

        Args:
            doc: Document dictionary.

        Returns:
            List of chunk records with document_id, language, index, text.
        """
        chunks: List[Dict[str, Any]] = []
        doc_id = doc.get("id")

        language_map = {
            "contentAr": "ar",
            "contentEn": "en",
            "contentFr": "fr",
        }

        for prop_key, lang_code in language_map.items():
            text = doc.get(prop_key, "") or ""
            if not text:
                continue

            split_chunks = self.chunk_text(text)
            for idx, chunk_text in enumerate(split_chunks):
                chunks.append({
                    "document_id": doc_id,
                    "language": lang_code,
                    "index": idx,
                    "text": chunk_text,
                })

        return chunks

    def create_chunk_nodes(self, chunks: List[Dict[str, Any]]) -> int:
        """Persist chunks as Chunk nodes linked to Documents.

        Args:
            chunks: List of chunk records.

        Returns:
            Number of chunks created.
        """
        if not chunks:
            return 0

        cypher = """
        UNWIND $chunks AS chunk
        MATCH (d:Document {id: chunk.document_id})
        CREATE (c:Chunk {
            text: chunk.text,
            language: chunk.language,
            index: chunk.index,
            created_at: datetime()
        })
        CREATE (d)-[:HAS_CHUNK {language: chunk.language}]->(c)
        RETURN count(c) AS count
        """

        try:
            driver = self.neo4j.connect()
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, {"chunks": chunks})
                record = result.single()
                count = record["count"] if record else 0
                logger.info("Created %d Chunk nodes.", count)
                return count
        except Neo4jError as exc:
            logger.error("Failed to create chunk nodes: %s", exc)
            return 0

    def process_document(self, doc: Dict[str, Any]) -> int:
        """Chunk a single document and persist chunks."""
        chunks = self.chunk_document(doc)
        return self.create_chunk_nodes(chunks)

    def process_documents(self, docs: List[Dict[str, Any]]) -> int:
        """Chunk and persist all documents."""
        total = 0
        for doc in docs:
            total += self.process_document(doc)
        logger.info("Chunking complete: %d chunks across %d documents.", total, len(docs))
        return total


def main() -> None:
    """CLI entrypoint for chunking from a JSON file."""
    import sys

    logging.basicConfig(level=settings.LOG_LEVEL)

    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.llm_agents.chunker <path-to-docs.json>")
        return

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    chunker = DocumentChunker()
    chunker.process_documents(data)


if __name__ == "__main__":
    main()
