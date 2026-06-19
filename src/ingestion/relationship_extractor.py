"""Legal cross-reference relationship extractor.

Detects AMENDS and REPEALS relationships between documents based on
natural language references found in document markdown content.

Uses language-specific regex patterns over normalized text. Arabic-Indic
numerals are converted to Western numerals before matching so that
references like ٣٧/٧٣ are captured as 37/73.
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
    """Extracts AMENDS and REPEALS legal cross-references from document text."""

    ARABIC_NUMERALS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    # English patterns
    AMENDS_PATTERNS_EN = [
        r"(?:amends|amending|amendment\s+(?:to|of))\s+(?:the\s+)?(?:Royal Decree|Ministerial Decision|Ministerial Order|Law|Decree)\s+(?:No\.?\s*)?(\d+\s*/\s*\d+)",
        r"(?:Royal Decree|Ministerial Decision|Ministerial Order|Law|Decree)\s+(?:No\.?\s*)?(\d+\s*/\s*\d+)\s+(?:is\s+)?(?:hereby\s+)?amended",
    ]

    REPEALS_PATTERNS_EN = [
        r"(?:repeals|repealing|repeal\s+(?:of|to))\s+(?:the\s+)?(?:Royal Decree|Ministerial Decision|Ministerial Order|Law|Decree)\s+(?:No\.?\s*)?(\d+\s*/\s*\d+)",
        r"(?:Royal Decree|Ministerial Decision|Ministerial Order|Law|Decree)\s+(?:No\.?\s*)?(\d+\s*/\s*\d+)\s+(?:is\s+)?(?:hereby\s+)?repealed",
        r"(?:abolishes|abolishing|abrogates|abrogating)\s+(?:the\s+)?(?:Royal Decree|Ministerial Decision|Ministerial Order|Law|Decree)\s+(?:No\.?\s*)?(\d+\s*/\s*\d+)",
        r"(?:supersedes|superseding|replaces|replacing)\s+(?:the\s+)?(?:Royal Decree|Ministerial Decision|Ministerial Order|Law|Decree)\s+(?:No\.?\s*)?(\d+\s*/\s*\d+)",
    ]

    # Arabic patterns: explicit amendment / repeal verbs followed by law type and number.
    # We intentionally avoid preamble references ("بعد الاطلاع على ... وتعديلاته")
    # because they merely cite prior amendments rather than stating a new one.
    # ال is optional on the law type to match both "القانون" and "قانون السير".
    AMENDS_PATTERNS_AR = [
        # Current law amends target law
        r"(?:يعدل|تعديل|معدل\s+لـ?|معدل\s+ل|تعديلا\s+لـ?|تعديلاً\s+لـ?)\s+(?:ال?مرسوم|ال?قرار|ال?قانون|ال?نظام)(?:\s+[\u0600-\u06FF]+){0,3}\s+(?:رقم\s*)?(\d+\s*/\s*\d+)",
        # Title-style: "بتعديل بعض أحكام القرار رقم X/Y"
        r"بتعديل\s+بعض\s+أحكام\s+(?:ال?مرسوم|ال?قرار|ال?قانون|ال?نظام)(?:\s+[\u0600-\u06FF]+){0,3}\s+رقم\s*(\d+\s*/\s*\d+)",
        r"بتعديل\s+(?:ال?مرسوم|ال?قرار|ال?قانون|ال?نظام)(?:\s+[\u0600-\u06FF]+){0,3}\s+رقم\s*(\d+\s*/\s*\d+)",
        # Amendment is made to target law
        r"(?:يجرى?\s+تعديل|أجريت\s+تعديل|أجري\s+تعديل)\s+(?:على|في)\s+(?:ال?مرسوم|ال?قرار|ال?قانون|ال?نظام)(?:\s+[\u0600-\u06FF]+){0,3}\s+(?:رقم\s*)?(\d+\s*/\s*\d+)",
        # Adding/replacing articles in target law
        r"(?:يضاف|يستبدل|يحذف|تضاف|تستبدل|تحذف)\s+(?:[\u0600-\u06FF\s]+?)\s+(?:ال?مرسوم|ال?قرار|ال?قانون|ال?نظام)(?:\s+[\u0600-\u06FF]+){0,3}\s+رقم\s*(\d+\s*/\s*\d+)",
    ]

    REPEALS_PATTERNS_AR = [
        # Current law repeals target law
        r"(?:يلغى|يلغي|إلغاء|ملغى|منسوخ|يُلغى|يُلغي)\s+(?:ال?مرسوم|ال?قرار|ال?قانون|ال?نظام)(?:\s+[\u0600-\u06FF]+){0,3}\s+(?:رقم\s*)?(\d+\s*/\s*\d+)",
        # Target law is considered repealed
        r"(?:يعتبر|تعتبر)\s+(?:ال?مرسوم|ال?قرار|ال?قانون|ال?نظام)(?:\s+[\u0600-\u06FF]+){0,3}\s+رقم\s*(\d+\s*/\s*\d+)\s+(?:ملغى|منسوخ|ملغياً|ملغيا)",
        # General repeal clauses with explicit numbers
        r"(?:يلغى|يلغي|إلغاء)\s+كل\s+ما\s+يخالف\s+.*?\b(?:القوانين|المراسيم|القرارات)\s+رقم\s*(\d+\s*/\s*\d+)",
        r"(?:يلغى|يلغي|إلغاء)\s+كل\s+ما\s+يتعارض\s+.*?\b(?:القوانين|المراسيم|القرارات)\s+رقم\s*(\d+\s*/\s*\d+)",
    ]

    def __init__(self) -> None:
        self.neo4j = get_neo4j_client()

    @staticmethod
    def _normalize_arabic_numerals(text: str) -> str:
        """Convert Arabic-Indic numerals to Western numerals."""
        return text.translate(RelationshipExtractor.ARABIC_NUMERALS)

    @staticmethod
    def _extract_references(text: str, patterns: List[str]) -> List[str]:
        """Find all references matching given regex patterns."""
        references: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Take the last numeric group as the reference number
                groups = [g for g in match.groups() if g]
                if groups:
                    # Normalize spaces around slash: "37 / 73" -> "37/73"
                    ref = re.sub(r"\s*/\s*", "/", groups[-1].strip())
                    references.append(ref)
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

        # Extract from title as well; Omani laws often state "...بتعديل المرسوم رقم X/Y"
        title = doc.get("title", "") or ""
        content_sources = [
            ("contentAr", self.AMENDS_PATTERNS_AR, self.REPEALS_PATTERNS_AR),
            ("contentEn", self.AMENDS_PATTERNS_EN, self.REPEALS_PATTERNS_EN),
            ("title", self.AMENDS_PATTERNS_AR, self.REPEALS_PATTERNS_AR),
        ]

        for content_key, amend_patterns, repeal_patterns in content_sources:
            text = doc.get(content_key, "") or ""
            if not text:
                continue

            # Normalize Arabic numerals so patterns work on both numeral systems
            text = self._normalize_arabic_numerals(text)

            amends = self._extract_references(text, amend_patterns)
            repeals = self._extract_references(text, repeal_patterns)

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
