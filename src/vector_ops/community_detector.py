"""Louvain community detection on the Document-Topic graph.

Detects legal "sub-fields" by clustering Documents and Topics based on
HAS_TOPIC relationships, then uses a local LLM to generate a human-readable
summary label for each community.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from community import community_louvain
from neo4j.exceptions import Neo4jError

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Runs Louvain community detection on the Document-Topic graph."""

    def __init__(self, resolution: float = 1.0, random_state: int = 42) -> None:
        self.neo4j = get_neo4j_client()
        self.resolution = resolution
        self.random_state = random_state

    def load_graph(self) -> Tuple[nx.Graph, Dict[str, str]]:
        """Load Document-Topic relationships from Neo4j into a NetworkX graph.

        Returns:
            graph: Undirected NetworkX graph with Document and Topic nodes.
            node_types: Mapping from node ID to node type ("Document" or "Topic").
        """
        query = """
        MATCH (d:Document)-[:HAS_TOPIC]->(t:Topic)
        RETURN d.id AS doc_id, t.name AS topic_name
        """
        driver = self.neo4j.connect()
        graph = nx.Graph()
        node_types: Dict[str, str] = {}

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            records = session.run(query).data()

        for record in records:
            doc_id = f"DOC:{record['doc_id']}"
            topic_name = f"TOPIC:{record['topic_name']}"

            graph.add_node(doc_id, node_type="Document")
            graph.add_node(topic_name, node_type="Topic")
            graph.add_edge(doc_id, topic_name)

            node_types[doc_id] = "Document"
            node_types[topic_name] = "Topic"

        logger.info(
            "Loaded graph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph, node_types

    def detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """Run Louvain community detection and return node -> community mapping."""
        partition = community_louvain.best_partition(
            graph,
            resolution=self.resolution,
            random_state=self.random_state,
        )
        num_communities = len(set(partition.values()))
        logger.info("Detected %d communities.", num_communities)
        return partition

    def build_community_summary(
        self,
        graph: nx.Graph,
        partition: Dict[str, int],
        node_types: Dict[str, str],
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate Document and Topic members per community."""
        communities: Dict[int, Dict[str, Any]] = {}
        for node_id, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = {
                    "documents": [],
                    "topics": [],
                }

            node_type = node_types.get(node_id, "Unknown")
            clean_id = node_id.split(":", 1)[1]

            if node_type == "Document":
                communities[comm_id]["documents"].append(clean_id)
            elif node_type == "Topic":
                communities[comm_id]["topics"].append(clean_id)

        return communities

    def generate_summary(self, topics: List[str], sample_docs: List[str]) -> str:
        """Generate a short human-readable label for a community.

        Uses a lightweight heuristic first; can be replaced with LLM calls.
        """
        if not topics:
            return "Mixed legal documents"

        # Simple heuristic: use the 3 most common topics joined together
        top_topics = topics[:5]
        label = ", ".join(top_topics)

        # Truncate if too long
        if len(label) > 100:
            label = label[:97] + "..."

        return label

    def write_communities_to_neo4j(
        self,
        partition: Dict[str, int],
        node_types: Dict[str, str],
        summaries: Dict[int, str],
    ) -> int:
        """Persist community IDs and summaries to Neo4j.

        Creates Community nodes and links members to them.
        """
        driver = self.neo4j.connect()
        communities_created = 0

        # Clear previous community assignments so re-runs do not leave stale links
        cleanup_query = """
        MATCH ()-[r:BELONGS_TO]->()
        DELETE r
        """
        cleanup_communities_query = """
        MATCH (c:Community)
        DELETE c
        """
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run(cleanup_query)
            session.run(cleanup_communities_query)
        logger.info("Cleared previous communities and BELONGS_TO relationships.")

        # Create Community nodes with summaries
        community_cypher = """
        MERGE (c:Community {id: $community_id})
        SET c.summary = $summary,
            c.updated_at = datetime()
        RETURN c.id AS id
        """

        # Link Documents to their Community
        doc_link_cypher = """
        MATCH (d:Document {id: $doc_id})
        MATCH (c:Community {id: $community_id})
        MERGE (d)-[:BELONGS_TO]->(c)
        SET d.community_id = $community_id
        """

        # Link Topics to their Community
        topic_link_cypher = """
        MATCH (t:Topic {name: $topic_name})
        MATCH (c:Community {id: $community_id})
        MERGE (t)-[:BELONGS_TO]->(c)
        SET t.community_id = $community_id
        """

        try:
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                # Create community nodes
                for comm_id, summary in summaries.items():
                    session.run(
                        community_cypher,
                        {"community_id": comm_id, "summary": summary},
                    )
                    communities_created += 1

                # Link members
                for node_id, comm_id in partition.items():
                    node_type = node_types.get(node_id, "Unknown")
                    clean_id = node_id.split(":", 1)[1]

                    if node_type == "Document":
                        session.run(
                            doc_link_cypher,
                            {"doc_id": clean_id, "community_id": comm_id},
                        )
                    elif node_type == "Topic":
                        session.run(
                            topic_link_cypher,
                            {"topic_name": clean_id, "community_id": comm_id},
                        )

            logger.info(
                "Wrote %d communities and member assignments to Neo4j.",
                communities_created,
            )
        except Neo4jError as exc:
            logger.error("Failed to write communities to Neo4j: %s", exc)
            raise

        return communities_created

    def run(self) -> Dict[int, Dict[str, Any]]:
        """Run the full community detection pipeline."""
        graph, node_types = self.load_graph()
        if graph.number_of_nodes() == 0:
            logger.warning("Graph is empty. No communities to detect.")
            return {}

        partition = self.detect_communities(graph)
        community_data = self.build_community_summary(graph, partition, node_types)

        summaries: Dict[int, str] = {}
        for comm_id, data in community_data.items():
            summaries[comm_id] = self.generate_summary(
                data["topics"], data["documents"]
            )
            logger.info(
                "Community %d: %d documents, %d topics, summary: %s",
                comm_id,
                len(data["documents"]),
                len(data["topics"]),
                summaries[comm_id],
            )

        self.write_communities_to_neo4j(partition, node_types, summaries)

        # Attach summaries to community data for return
        for comm_id, data in community_data.items():
            data["summary"] = summaries[comm_id]

        return community_data


def main() -> None:
    """CLI entrypoint for community detection."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=settings.LOG_LEVEL)
    detector = CommunityDetector()
    communities = detector.run()

    print(f"\nDetected {len(communities)} communities:")
    for comm_id, data in communities.items():
        print(
            f"  Community {comm_id}: {len(data['documents'])} docs, "
            f"{len(data['topics'])} topics — {data['summary']}"
        )


if __name__ == "__main__":
    main()
