"""Document node builder and ingestion into Neo4j.

Transforms parsed legal documents into central Document nodes with
language-specific markdown properties, following the exam's simplified
schema principle.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import Session
from neo4j.exceptions import Neo4jError

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class DocumentBuilder:
    """Builds and ingests Document nodes into Neo4j."""

    # Language property keys supported by the pipeline
    LANGUAGE_KEYS = ["contentAr", "contentEn", "contentFr"]

    def __init__(self) -> None:
        self.neo4j = get_neo4j_client()

    @staticmethod
    def _clean_date(date_str: Optional[str]) -> Optional[str]:
        """Normalize date string to ISO format (YYYY-MM-DD) if possible."""
        if not date_str:
            return None
        try:
            parsed = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            try:
                parsed = datetime.strptime(date_str.strip(), "%d/%m/%Y")
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                return date_str.strip() if date_str else None

    @staticmethod
    def _build_properties(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Build Neo4j-compatible property dictionary from parsed document."""
        props: Dict[str, Any] = {
            "id": doc.get("id") or doc.get("source_url"),
            "title": doc.get("title", ""),
            "document_type": doc.get("document_type", ""),
            "number": doc.get("number", ""),
            "issue_date": DocumentBuilder._clean_date(doc.get("issue_date")),
            "issuer": doc.get("issuer", ""),
            "source_url": doc.get("source_url", ""),
        }

        # Store each language's markdown directly as a property on the Document node
        for lang_key in DocumentBuilder.LANGUAGE_KEYS:
            props[lang_key] = doc.get(lang_key, "") or ""

        # Optional fallback raw markdown
        props["raw_markdown"] = doc.get("raw_markdown", "")

        return props

    def ingest_document(self, doc: Dict[str, Any]) -> Optional[str]:
        """Ingest a single Document node into Neo4j.

        Args:
            doc: Parsed document dictionary.

        Returns:
            The document ID if successful, None otherwise.
        """
        props = self._build_properties(doc)

        if not props["id"]:
            logger.error("Document missing required 'id' or 'source_url'. Skipping.")
            return None

        cypher = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.document_type = $document_type,
            d.number = $number,
            d.issue_date = $issue_date,
            d.issuer = $issuer,
            d.source_url = $source_url,
            d.contentAr = $contentAr,
            d.contentEn = $contentEn,
            d.contentFr = $contentFr,
            d.raw_markdown = $raw_markdown,
            d.updated_at = datetime()
        RETURN d.id AS id
        """

        try:
            driver = self.neo4j.connect()
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, props)
                record = result.single()
                if record:
                    logger.info("Ingested Document: %s", record["id"])
                    return record["id"]
        except Neo4jError as exc:
            logger.error("Failed to ingest document %s: %s", props.get("id"), exc)

        return None

    def ingest_documents(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Batch ingest multiple documents.

        Args:
            docs: List of parsed document dictionaries.

        Returns:
            List of successfully ingested document IDs.
        """
        ingested: List[str] = []
        for doc in docs:
            doc_id = self.ingest_document(doc)
            if doc_id:
                ingested.append(doc_id)
        logger.info("Batch ingestion complete: %d/%d documents ingested.", len(ingested), len(docs))
        return ingested

    def ingest_from_json_file(self, file_path: Path) -> List[str]:
        """Ingest documents from a JSON file.

        Args:
            file_path: Path to JSON file containing a single document or list of documents.

        Returns:
            List of successfully ingested document IDs.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("JSON file must contain a document object or a list of documents.")

        return self.ingest_documents(data)


def main() -> None:
    """CLI entrypoint for ingesting sample documents."""
    import sys

    logging.basicConfig(level=settings.LOG_LEVEL)

    client = get_neo4j_client()
    client.setup_schema()

    builder = DocumentBuilder()

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.exists():
            builder.ingest_from_json_file(path)
        else:
            logger.error("File not found: %s", path)
    else:
        logger.info("No input file provided. Schema setup complete.")


if __name__ == "__main__":
    main()
