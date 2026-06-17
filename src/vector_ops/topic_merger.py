"""Topological topic merging via cosine similarity.

Detects duplicate or highly synonymous Topic nodes by comparing their
embeddings. Merges nodes exceeding a configurable cosine similarity threshold
and consolidates their relationships.
"""

import logging
from typing import List, Tuple

import numpy as np
from neo4j.exceptions import Neo4jError
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class TopicMerger:
    """Merge synonymous Topic nodes based on embedding similarity."""

    def __init__(self, threshold: float = settings.TOPIC_MERGE_THRESHOLD) -> None:
        self.threshold = threshold
        self.neo4j = get_neo4j_client()

    def fetch_topic_embeddings(self) -> List[dict]:
        """Fetch all topics with their embeddings."""
        query = """
        MATCH (t:Topic)
        WHERE t.embedding IS NOT NULL
        RETURN elementId(t) AS node_id, t.name AS name, t.embedding AS embedding
        """
        driver = self.neo4j.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            return session.run(query).data()

    def compute_similarity_pairs(
        self, topics: List[dict]
    ) -> List[Tuple[str, str, float]]:
        """Compute pairwise cosine similarities and return high-sim pairs."""
        if len(topics) < 2:
            return []

        embeddings = np.array([t["embedding"] for t in topics])
        sim_matrix = cosine_similarity(embeddings)

        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                score = float(sim_matrix[i][j])
                if score >= self.threshold:
                    pairs.append((topics[i]["node_id"], topics[j]["node_id"], score))

        # Sort by descending similarity
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def merge_topics(self, keep_id: str, merge_id: str) -> bool:
        """Merge merge_id topic into keep_id topic.

        Transfers all incoming HAS_TOPIC relationships and deletes merge_id.
        """
        cypher = """
        MATCH (keep:Topic) WHERE elementId(keep) = $keep_id
        MATCH (merge:Topic) WHERE elementId(merge) = $merge_id
        WITH keep, merge
        MATCH (d:Document)-[r:HAS_TOPIC]->(merge)
        MERGE (d)-[:HAS_TOPIC]->(keep)
        DELETE r
        WITH keep, merge
        DETACH DELETE merge
        """
        try:
            driver = self.neo4j.connect()
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                session.run(cypher, {"keep_id": keep_id, "merge_id": merge_id})
            logger.info("Merged topic %s into %s", merge_id, keep_id)
            return True
        except Neo4jError as exc:
            logger.error("Failed to merge topics %s -> %s: %s", merge_id, keep_id, exc)
            return False

    def run(self) -> int:
        """Run topic merging and return number of merges performed."""
        topics = self.fetch_topic_embeddings()
        logger.info("Fetched %d topics for merging analysis.", len(topics))

        pairs = self.compute_similarity_pairs(topics)
        logger.info("Found %d topic pairs above threshold %.2f.", len(pairs), self.threshold)

        merged_ids = set()
        merge_count = 0

        for keep_id, merge_id, score in pairs:
            if merge_id in merged_ids:
                continue
            if self.merge_topics(keep_id, merge_id):
                merged_ids.add(merge_id)
                merge_count += 1

        logger.info("Topic merging complete: %d topics merged.", merge_count)
        return merge_count


def main() -> None:
    """CLI entrypoint for topic merging."""
    logging.basicConfig(level=settings.LOG_LEVEL)

    merger = TopicMerger()
    merger.run()


if __name__ == "__main__":
    main()
