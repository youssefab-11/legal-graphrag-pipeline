"""Embedding generation for Topic and Chunk nodes.

Uses Ollama's nomic-embed-text model to compute dense vector representations
locally and stores them as properties on Neo4j nodes.
"""

import logging
from typing import List, Optional

import ollama
from neo4j.exceptions import Neo4jError

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class Embedder:
    """Local embedding generator using Ollama."""

    def __init__(
        self,
        model: str = settings.OLLAMA_EMBEDDING_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.client = ollama.Client(host=base_url)
        self.batch_size = batch_size
        self.neo4j = get_neo4j_client()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for text in texts:
            try:
                response = self.client.embeddings(model=self.model, prompt=text)
                embeddings.append(response["embedding"])
            except Exception as exc:
                logger.error("Embedding failed for text: %s... Error: %s", text[:50], exc)
                embeddings.append([])
        return embeddings

    def embed_nodes(self, label: str, text_property: str = "text") -> int:
        """Compute and store embeddings for all nodes of a given label.

        Args:
            label: Neo4j node label, e.g., 'Topic' or 'Chunk'.
            text_property: Property containing the text to embed.

        Returns:
            Number of nodes updated.
        """
        logger.info("Embedding %s nodes using %s...", label, self.model)

        fetch_query = f"""
        MATCH (n:{label})
        WHERE n.{text_property} IS NOT NULL AND (n.embedding IS NULL OR size(n.embedding) = 0)
        RETURN elementId(n) AS node_id, n.{text_property} AS text
        """

        driver = self.neo4j.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            records = session.run(fetch_query).data()

        if not records:
            logger.info("No %s nodes need embedding.", label)
            return 0

        total_updated = 0
        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]
            texts = [r["text"] for r in batch]
            embeddings = self.embed(texts)

            update_query = f"""
            UNWIND $updates AS update
            MATCH (n:{label})
            WHERE elementId(n) = update.node_id
            SET n.embedding = update.embedding
            """

            updates = [
                {"node_id": r["node_id"], "embedding": emb}
                for r, emb in zip(batch, embeddings)
                if emb
            ]

            if updates:
                try:
                    with driver.session(database=settings.NEO4J_DATABASE) as session:
                        session.run(update_query, {"updates": updates})
                        total_updated += len(updates)
                        logger.info("Embedded %d/%d %s nodes.", total_updated, len(records), label)
                except Neo4jError as exc:
                    logger.error("Failed to store embeddings for %s: %s", label, exc)

        return total_updated

    def process_all(self) -> None:
        """Embed both Topic and Chunk nodes."""
        self.embed_nodes("Topic", text_property="name")
        self.embed_nodes("Chunk", text_property="text")
        logger.info("All embeddings generated and stored.")


def main() -> None:
    """CLI entrypoint for embedding generation."""
    logging.basicConfig(level=settings.LOG_LEVEL)

    client = get_neo4j_client()
    client.setup_schema()

    embedder = Embedder()
    embedder.process_all()


if __name__ == "__main__":
    main()
