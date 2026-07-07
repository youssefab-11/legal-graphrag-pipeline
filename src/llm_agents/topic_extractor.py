"""LLM-driven legal topic extraction.

Uses a local Ollama model to extract key legal themes from document content
and creates Topic nodes linked to Document nodes.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j.exceptions import Neo4jError
from tqdm import tqdm

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client
from src.llm_agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


class TopicExtractor:
    """Extracts legal topics from document content using a local or remote LLM."""

    SYSTEM_PROMPT = (
        "You are a legal expert specializing in Omani legislation. "
        "Your task is to read the provided legal text and identify the top 3 to 5 core legal topics or themes. "
        "You do not have access to any tools, functions, or commands. "
        "Do not use tool calls, XML tags, markdown, explanations, or code fences. "
        "Return ONLY a valid JSON array of strings. "
        "Example output: [\"Taxation\", \"Labour Law\", \"Omanization\"]"
    )

    BATCH_SYSTEM_PROMPT = (
        "You are a legal expert specializing in Omani legislation. "
        "For each document below, identify the top 3 to 5 core legal topics or themes. "
        "You do not have access to any tools, functions, or commands. "
        "Do not use tool calls, XML tags, markdown, explanations, or code fences. "
        "Return ONLY a valid JSON object mapping each document ID to an array of topic strings. "
        "Example output: {\"doc-1\": [\"Taxation\", \"Labour Law\"], \"doc-2\": [\"Investment\", \"Foreign Capital\"]}"
    )

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = settings.TOPIC_BATCH_SIZE,
        batch_max_chars: int = settings.TOPIC_BATCH_MAX_CHARS,
    ) -> None:
        self.client = LLMClient(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        self.neo4j = get_neo4j_client()
        self.batch_size = batch_size
        self.batch_max_chars = batch_max_chars

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
            raw_output = self.client.chat(messages=messages)
            topics = self._parse_topics(raw_output)
            logger.info("Extracted topics: %s", topics)
            return topics
        except Exception as exc:
            logger.error("LLM topic extraction failed: %s", exc)
            return []

    def _build_batch_prompt(self, docs: List[Dict[str, Any]]) -> str:
        """Build a prompt containing multiple documents."""
        parts = ["Extract legal topics for each of the following Omani legislation documents:\n"]
        for doc in docs:
            doc_id = doc.get("id", "unknown")
            content = doc.get("contentEn") or doc.get("contentAr") or doc.get("raw_markdown", "")
            truncated = (content or "")[: self.batch_max_chars]
            parts.append(f"\n[Document ID: {doc_id}]\n{truncated}\n")
        return "\n".join(parts)

    def _parse_batch_topics(self, text: str, doc_ids: List[str]) -> Dict[str, List[str]]:
        """Parse a JSON object mapping document IDs to topic arrays."""
        cleaned = self._clean_response(text)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse batch JSON response: %s", exc)
            return {}

        result: Dict[str, List[str]] = {}
        for doc_id in doc_ids:
            topics = parsed.get(doc_id)
            if isinstance(topics, list):
                result[doc_id] = [str(t).strip() for t in topics if str(t).strip()]
            elif isinstance(topics, dict) and "topics" in topics:
                result[doc_id] = [str(t).strip() for t in topics["topics"] if str(t).strip()]
            else:
                result[doc_id] = []
        return result

    def extract_batch(self, docs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract topics for a batch of documents in a single LLM call.

        Args:
            docs: List of document dictionaries.

        Returns:
            Dictionary mapping document ID to list of topic strings.
        """
        if not docs:
            return {}

        doc_ids = [doc.get("id") for doc in docs if doc.get("id")]
        if len(doc_ids) != len(docs):
            logger.warning("Some documents in batch have no id.")

        messages = [
            {"role": "system", "content": self.BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_batch_prompt(docs)},
        ]

        try:
            raw_output = self.client.chat(messages=messages)
            return self._parse_batch_topics(raw_output, doc_ids)
        except Exception as exc:
            logger.error("Batch LLM topic extraction failed: %s", exc)
            return {}

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
        """Extract topics for a list of documents using batching."""
        total = 0
        failed_docs: List[Dict[str, Any]] = []

        # Process documents in batches
        num_batches = (len(docs) + self.batch_size - 1) // self.batch_size
        desc = f"Extracting topics (batches of {self.batch_size})"
        for i in tqdm(range(num_batches), desc=desc, unit="batch"):
            batch = docs[i * self.batch_size : (i + 1) * self.batch_size]
            batch_results = self.extract_batch(batch)

            for doc in batch:
                doc_id = doc.get("id")
                if not doc_id:
                    continue

                topics = batch_results.get(doc_id)
                if topics is None or not topics:
                    failed_docs.append(doc)
                    continue

                self.create_topic_nodes(doc_id, topics)
                total += len(topics)

        # Fallback: individually process any docs that failed in batch mode
        if failed_docs:
            logger.info("Falling back to individual extraction for %d documents.", len(failed_docs))
            for doc in tqdm(failed_docs, desc="Individual fallback", unit="doc"):
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
