"""Vector index manager.

Provides a dedicated interface for creating, verifying, and rebuilding
vector indexes used for Topic and Chunk embeddings.
"""

import logging

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class VectorIndexManager:
    """Manages Neo4j vector indexes."""

    def __init__(self) -> None:
        self.neo4j = get_neo4j_client()

    def setup(self) -> None:
        """Create constraints, full-text, and vector indexes."""
        logger.info("Setting up vector indexes...")
        self.neo4j.setup_schema()
        logger.info("Vector index setup complete.")

    def verify(self) -> dict:
        """Return status of existing indexes."""
        query = "SHOW INDEXES YIELD name, type, state, entityType RETURN name, type, state, entityType"
        driver = self.neo4j.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            records = session.run(query).data()
        return {r["name"]: r for r in records}


def main() -> None:
    """CLI entrypoint for index management."""
    logging.basicConfig(level=settings.LOG_LEVEL)

    manager = VectorIndexManager()
    manager.setup()
    indexes = manager.verify()
    for name, info in indexes.items():
        logger.info("Index: %s | Type: %s | State: %s", name, info["type"], info["state"])


if __name__ == "__main__":
    main()
