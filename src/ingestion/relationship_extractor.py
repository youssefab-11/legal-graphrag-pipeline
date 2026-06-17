"""Legal cross-reference relationship extractor.

Detects AMENDS and REPEALS relationships between documents based on
natural language references found in document markdown content.

This implementation uses regex heuristics suitable for the 50–100 document
pilot. A production system would use a fine-tuned NER or LLM-based extractor.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neo4j.exceptions import Neo4jError

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Extracts legal cross-references from document text."""

    # English patterns
    AMENDS_PATTERNS_EN = [
        r"amends\s+(?:Royal Decree|Ministerial Decision|Ministerial Order)\s+(?:No\.?\s*)?(\d+\/\d+)",
        r"amending\s+(?:Royal Decree|Ministerial Decision|Ministerial Order)\s+(?:No\.?\s*)?(\d+\/\d+)",
    ]

    REPEALS_PATTERNS_EN = [
        r"repeals\s+(?:Royal Decree|Ministerial Decision|Ministerial Order)\s+(?:No\.?\s*)?(\d+\/\d+)",
        r"repealing\s+(?:Royal Decree|Ministerial Decision|Ministerial Order)\s+(?:No\.?\s*)?(\d+\/\d+)",
    ]

    # Arabic patterns (simplified; Arabic numerals and "المرسوم" / "القرار")
    AMENDS_PATTERNS_AR = [
        r"(يعدل|تعديل)\s+(?:المرسوم|القرار|القانون)\s+(?:رقم\s*)?(\d+\/\d+)",
    ]

    REPEALS_PATTERNS_AR = [
        r"(يلغي|إلغاء)\s+(?:المرسوم|القرار|القانون)\s+(?:رقم\s*)?(\d+\/\d+)",
    ]

    def __init__(self) -> None:
        self.neo4j = get_neo4j_client()

    @staticmethod
    def _extract_references(text: str, patterns: List[str]) -> List[str]:
        """Find all references matching given regex patterns."""
        references: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Take the last numeric group as the reference number
                groups = [g for g in match.groups() if g]
                if groups:
                    references.append(groups[-1].strip())
        return list(set(references))

    def extract_from_document(self, doc: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Extract (source_id, rel_type, target_reference) tuples from a document.

        Args:
            doc: Document dictionary with contentAr/contentEn fields.

        Returns:
            List of relationship tuples.
        """
        source_id = doc.get("id") or doc.get("source_url", "")
        if not source_id:
            return []

        relationships: List[Tuple[str, str, str]] = []

        for content_key in ["contentEn", "contentAr"]:
            text = doc.get(content_key, "") or ""
            if not text:
                continue

            if content_key == "contentAr":
                amends = self._extract_references(text, self.AMENDS_PATTERNS_AR)
                repeals = self._extract_references(text, self.REPEALS_PATTERNS_AR)
            else:
                amends = self._extract_references(text, self.AMENDS_PATTERNS_EN)
                repeals = self._extract_references(text, self.REPEALS_PATTERNS_EN)

            for ref in amends:
                relationships.append((source_id, "AMENDS", ref))
            for ref in repeals:
                relationships.append((source_id, "REPEALS", ref))

        return relationships

    def create_relationships(self, relationships: List[Tuple[str, str, str]]) -> int:
        """Create relationships in Neo4j.

        Args:
            relationships: List of (source_id, rel_type, target_reference) tuples.

        Returns:
            Number of relationships created.
        """
        if not relationships:
            return 0

        # Separate by relationship type because Neo4j cannot parameterize rel types directly
        amends = [
            {"source_id": src, "target_reference": target_ref}
            for src, rel_type, target_ref in relationships
            if rel_type == "AMENDS"
        ]
        repeals = [
            {"source_id": src, "target_reference": target_ref}
            for src, rel_type, target_ref in relationships
            if rel_type == "REPEALS"
        ]

        cypher_templates = {
            "AMENDS": """
                UNWIND $rels AS rel
                MATCH (source:Document {id: rel.source_id})
                MATCH (target:Document {number: rel.target_reference})
                MERGE (source)-[r:AMENDS]->(target)
                ON CREATE SET r.created_at = datetime()
                RETURN count(r) AS count
            """,
            "REPEALS": """
                UNWIND $rels AS rel
                MATCH (source:Document {id: rel.source_id})
                MATCH (target:Document {number: rel.target_reference})
                MERGE (source)-[r:REPEALS]->(target)
                ON CREATE SET r.created_at = datetime()
                RETURN count(r) AS count
            """,
        }

        total = 0
        driver = self.neo4j.connect()
        for rel_type, rels in [("AMENDS", amends), ("REPEALS", repeals)]:
            if not rels:
                continue
            try:
                with driver.session(database=settings.NEO4J_DATABASE) as session:
                    result = session.run(cypher_templates[rel_type], {"rels": rels})
                    record = result.single()
                    count = record["count"] if record else 0
                    total += count
                    logger.info("Created %d %s relationships.", count, rel_type)
            except Neo4jError as exc:
                logger.error("Failed to create %s relationships: %s", rel_type, exc)

        return total

    def process_documents(self, docs: List[Dict[str, Any]]) -> int:
        """Extract and create relationships for a list of documents."""
        all_relationships: List[Tuple[str, str, str]] = []
        for doc in docs:
            all_relationships.extend(self.extract_from_document(doc))
        return self.create_relationships(all_relationships)


def main() -> None:
    """CLI entrypoint for extracting relationships from a JSON file."""
    import sys

    logging.basicConfig(level=settings.LOG_LEVEL)

    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.ingestion.relationship_extractor <path-to-docs.json>")
        return

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    extractor = RelationshipExtractor()
    extractor.process_documents(data)


if __name__ == "__main__":
    main()
