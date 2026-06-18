"""Neo4j graph database client and schema manager.

Provides connection handling, schema creation, constraints, vector indexes,
and helper methods for batch ingestion of Document, Topic, and Chunk nodes.
"""

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError, DatabaseError

from src.config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Singleton-style Neo4j client for the Legal GraphRAG pipeline."""

    _instance: Optional["Neo4jClient"] = None

    def __new__(cls) -> "Neo4jClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._driver: Optional[Driver] = None
        return cls._instance

    def connect(self) -> Driver:
        """Initialize and return the Neo4j driver."""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                )
                self._driver.verify_connectivity()
                logger.info("Connected to Neo4j at %s", settings.NEO4J_URI)
            except Neo4jError as exc:
                logger.error("Failed to connect to Neo4j: %s", exc)
                raise
        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    def run_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
        write: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return records as dictionaries.

        Args:
            query: Cypher query string.
            parameters: Query parameters.
            database: Target database name.
            write: If True, uses execute_write; otherwise execute_read.
        """
        driver = self.connect()
        db = database or settings.NEO4J_DATABASE

        def _execute(session: Session) -> List[Dict[str, Any]]:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

        with driver.session(database=db) as session:
            if write:
                return session.execute_write(_execute)
            return session.execute_read(_execute)

    def create_constraints(self) -> None:
        """Create uniqueness constraints for node IDs."""
        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        ]
        for cypher in constraints:
            try:
                self.run_query(cypher)
                logger.info("Constraint created: %s", cypher)
            except DatabaseError as exc:
                logger.warning("Constraint may already exist: %s", exc)

    def create_indexes(self) -> None:
        """Create full-text and vector indexes."""
        # Full-text index for BM25-style keyword search on markdown content
        full_text_query = """
        CREATE FULLTEXT INDEX documentContent IF NOT EXISTS
        FOR (d:Document)
        ON EACH [d.contentAr, d.contentEn]
        """
        try:
            self.run_query(full_text_query)
            logger.info("Full-text index created/verified.")
        except DatabaseError as exc:
            logger.warning("Full-text index issue: %s", exc)

    def create_vector_indexes(self) -> None:
        """Create vector indexes for Topic and Chunk embeddings.

        Assumes embedding dimension 768 for nomic-embed-text.
        Adjust if using a different model.
        """
        vector_indexes = [
            {
                "name": "topic_embeddings",
                "label": "Topic",
                "property": "embedding",
                "dimension": settings.EMBEDDING_DIMENSION,
            },
            {
                "name": "chunk_embeddings",
                "label": "Chunk",
                "property": "embedding",
                "dimension": settings.EMBEDDING_DIMENSION,
            },
        ]

        for idx in vector_indexes:
            query = f"""
            CREATE VECTOR INDEX {idx['name']} IF NOT EXISTS
            FOR (n:{idx['label']})
            ON (n.{idx['property']})
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {idx['dimension']},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            try:
                self.run_query(query)
                logger.info("Vector index created/verified: %s", idx["name"])
            except DatabaseError as exc:
                logger.warning("Vector index issue for %s: %s", idx["name"], exc)

    def setup_schema(self) -> None:
        """Run all schema setup operations."""
        logger.info("Setting up Neo4j schema...")
        self.create_constraints()
        self.create_indexes()
        self.create_vector_indexes()
        logger.info("Schema setup complete.")

    def clear_database(self) -> None:
        """Delete all nodes and relationships. Use with caution."""
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.warning("All nodes and relationships deleted.")


def get_neo4j_client() -> Neo4jClient:
    """Factory function returning the Neo4j client instance."""
    return Neo4jClient()
