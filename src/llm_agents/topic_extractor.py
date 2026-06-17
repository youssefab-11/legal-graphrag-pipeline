"""LLM-driven legal topic extraction.

Uses a local Ollama model to extract key legal themes from document content
and creates Topic nodes linked to Document nodes.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import ollama
from neo4j.exceptions import Neo4jError

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class TopicExtractor:
    """Extracts legal topics from document content using a local LLM."""

    SYSTEM_PROMPT = (
        "You are a legal expert specializing in Omani legislation. "
        "Your task is to read the provided legal text and identify the top 3 to 5 core legal topics or themes. "
        "Return ONLY a valid JSON array of strings, with no markdown, no explanation, and no code fences. "
        "Example output: [\"Taxation\", \"Labour Law\", \"Omanization\"]"
    )

    def __init__(
        self,
        model: str = settings.OLLAMA_LLM_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.client = ollama.Client(host=base_url)
        self.neo4j = get_neo4j_client()

    def _clean_response(self, text: str) -> str:
        """Remove markdown code fences and extra whitespace."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _parse_topics(self, text: str) -> List[str]:
        """Safely parse LLM response into a list of topic strings."""
        cleaned = self._clean_response(text)
        try:
            topics = json.loads(cleaned)
            if isinstance(topics, list):
                return [str(t).strip() for t in topics if str(t).strip()]
            elif isinstance(topics, dict) and "topics" in topics:
                return [str(t).strip() for t in topics["topics"] if str(t).strip()]
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response. Falling back to line split.")

        # Fallback: split by newlines or commas
        fallback = re.split(r"\n|,", cleaned)
        return [t.strip("-\"'[] ") for t in fallback if t.strip("-\"'[] ")]

    def extract(self, content: str) -> List[str]:
        """Extract topics from a single document's markdown content.

        Args:
            content: Markdown text of the document.

        Returns:
            List of topic strings.
        """
        if not content or len(content.strip()) < 50:
            logger.warning("Content too short for topic extraction.")
            return []

        # Truncate to manage context window (llama3.1 8k context is generous)
        truncated = content[:6000]

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract legal topics from the following Omani legislation:\n\n{truncated}"},
        ]

        try:
            response = self.client.chat(model=self.model, messages=messages)
            raw_output = response["message"]["content"]
            topics = self._parse_topics(raw_output)
            logger.info("Extracted topics: %s", topics)
            return topics
        except Exception as exc:
            logger.error("Ollama topic extraction failed: %s", exc)
            return []

    def extract_for_document(self, doc: Dict[str, Any]) -> List[str]:
        """Extract topics using the best available language content."""
        # Prefer English if available, otherwise Arabic
        content = doc.get("contentEn") or doc.get("contentAr") or doc.get("raw_markdown", "")
        return self.extract(content)

    def create_topic_nodes(self, document_id: str, topics: List[str]) -> int:
        """Create Topic nodes and link them to a Document.

        Args:
            document_id: ID of the parent Document.
            topics: List of topic names.

        Returns:
            Number of topics linked.
        """
        if not topics:
            return 0

        cypher = """
        MATCH (d:Document {id: $document_id})
        UNWIND $topics AS topic_name
        MERGE (t:Topic {name: topic_name})
        MERGE (d)-[:HAS_TOPIC]->(t)
        RETURN count(t) AS count
        """

        try:
            driver = self.neo4j.connect()
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, {"document_id": document_id, "topics": topics})
                record = result.single()
                count = record["count"] if record else 0
                logger.info("Linked %d topics to Document %s", count, document_id)
                return count
        except Neo4jError as exc:
            logger.error("Failed to create topic nodes for %s: %s", document_id, exc)
            return 0

    def process_document(self, doc: Dict[str, Any]) -> List[str]:
        """Extract topics and persist them for a single document."""
        doc_id = doc.get("id")
        if not doc_id:
            logger.error("Document has no id. Skipping topic extraction.")
            return []

        topics = self.extract_for_document(doc)
        self.create_topic_nodes(doc_id, topics)
        return topics

    def process_documents(self, docs: List[Dict[str, Any]]) -> int:
        """Extract topics for a list of documents."""
        total = 0
        for doc in docs:
            topics = self.process_document(doc)
            total += len(topics)
        logger.info("Topic extraction complete: %d topics across %d documents.", total, len(docs))
        return total


def main() -> None:
    """CLI entrypoint for topic extraction from a JSON file."""
    import sys

    logging.basicConfig(level=settings.LOG_LEVEL)

    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.llm_agents.topic_extractor <path-to-docs.json>")
        return

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    client = get_neo4j_client()
    client.setup_schema()

    extractor = TopicExtractor()
    extractor.process_documents(data)


if __name__ == "__main__":
    main()
