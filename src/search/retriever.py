"""Hybrid retriever combining BM25, vector search, and graph expansion.

Stage 1: Candidate generation via weighted BM25 + dense vector scores.
Stage 2: Topological context expansion using graph relationships.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from neo4j.exceptions import Neo4jError

from src.config.settings import settings
from src.ingestion.neo4j_client import get_neo4j_client
from src.vector_ops.embedder import Embedder

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves relevant chunks using hybrid search and graph expansion."""

    def __init__(
        self,
        top_k: int = settings.TOP_K_CANDIDATES,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.neo4j = get_neo4j_client()
        self.embedder = Embedder()

    def vector_search_chunks(self, query_embedding: List[float], top_k: int) -> List[dict]:
        """Vector similarity search over Chunk nodes."""
        query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
        YIELD node AS chunk, score
        MATCH (d:Document)-[:HAS_CHUNK]->(chunk)
        RETURN elementId(chunk) AS chunk_id,
               chunk.text AS text,
               chunk.language AS language,
               d.id AS document_id,
               d.title AS title,
               score
        """
        try:
            driver = self.neo4j.connect()
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                return session.run(query, {"top_k": top_k, "embedding": query_embedding}).data()
        except Neo4jError as exc:
            logger.error("Vector search failed: %s", exc)
            return []

    def vector_search_topics(self, query_embedding: List[float], top_k: int) -> List[dict]:
        """Vector similarity search over Topic nodes to find related documents."""
        query = """
        CALL db.index.vector.queryNodes('topic_embeddings', $top_k, $embedding)
        YIELD node AS topic, score
        MATCH (d:Document)-[:HAS_TOPIC]->(topic)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        RETURN elementId(c) AS chunk_id,
               c.text AS text,
               c.language AS language,
               d.id AS document_id,
               d.title AS title,
               score,
               topic.name AS matched_topic
        LIMIT $top_k
        """
        try:
            driver = self.neo4j.connect()
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                return session.run(query, {"top_k": top_k, "embedding": query_embedding}).data()
        except Neo4jError as exc:
            logger.error("Topic vector search failed: %s", exc)
            return []

    def bm25_search(self, query: str, top_k: int) -> List[dict]:
        """BM25 keyword search over Document markdown content.

        Uses in-memory scoring over available chunks since Neo4j Community
        full-text scoring is limited. This is a practical hybrid fallback.
        """
        try:
            driver = self.neo4j.connect()
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                # Fetch all chunks for BM25 scoring
                records = session.run("""
                    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                    RETURN elementId(c) AS chunk_id, c.text AS text,
                           c.language AS language, d.id AS document_id, d.title AS title
                """).data()
        except Neo4jError as exc:
            logger.error("BM25 fetch failed: %s", exc)
            return []

        if not records:
            return []

        tokenized_corpus = [r["text"].lower().split() for r in records]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        for idx, record in enumerate(records):
            record["score"] = float(scores[idx])

        records.sort(key=lambda x: x["score"], reverse=True)
        return records[:top_k]

    def combine_candidates(
        self,
        vector_chunk_results: List[dict],
        vector_topic_results: List[dict],
        bm25_results: List[dict],
    ) -> List[dict]:
        """Merge and rank candidates from multiple sources."""
        candidate_map: Dict[str, dict] = {}

        def normalize_scores(results: List[dict], source: str, weight: float) -> None:
            if not results:
                return
            scores = [r.get("score", 0.0) for r in results]
            max_score = max(scores) if scores else 1.0
            for r in results:
                chunk_id = r.get("chunk_id")
                if not chunk_id:
                    continue
                norm_score = (r.get("score", 0.0) / max_score) * weight if max_score > 0 else 0.0
                if chunk_id not in candidate_map:
                    candidate_map[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": r.get("text", ""),
                        "language": r.get("language", ""),
                        "document_id": r.get("document_id", ""),
                        "title": r.get("title", ""),
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                        "sources": [],
                    }
                candidate = candidate_map[chunk_id]
                if source == "vector_chunk" or source == "vector_topic":
                    candidate["vector_score"] = max(candidate["vector_score"], norm_score)
                elif source == "bm25":
                    candidate["bm25_score"] = max(candidate["bm25_score"], norm_score)
                candidate["sources"].append(source)

        normalize_scores(vector_chunk_results, "vector_chunk", self.vector_weight)
        normalize_scores(vector_topic_results, "vector_topic", self.vector_weight)
        normalize_scores(bm25_results, "bm25", self.bm25_weight)

        for c in candidate_map.values():
            c["combined_score"] = c["vector_score"] + c["bm25_score"]

        ranked = sorted(candidate_map.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked[: self.top_k]

    def graph_expand(self, chunk_id: str) -> Dict[str, Any]:
        """Expand a chunk with graph context: document metadata, topics, relationships."""
        query = """
        MATCH (c:Chunk) WHERE elementId(c) = $chunk_id
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        OPTIONAL MATCH (d)-[:AMENDS|REPEALS]->(related:Document)
        RETURN d.id AS document_id,
               d.title AS title,
               d.document_type AS document_type,
               d.number AS number,
               d.issue_date AS issue_date,
               d.issuer AS issuer,
               collect(DISTINCT t.name) AS topics,
               collect(DISTINCT related.id) AS related_documents
        """
        try:
            driver = self.neo4j.connect()
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(query, {"chunk_id": chunk_id}).data()
                return result[0] if result else {}
        except Neo4jError as exc:
            logger.error("Graph expansion failed: %s", exc)
            return {}

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Run full hybrid retrieval with graph expansion.

        Args:
            query: User query string.

        Returns:
            List of expanded candidate dictionaries.
        """
        logger.info("Retrieving candidates for query: %s", query)

        # Generate query embedding
        query_embedding = self.embedder.embed([query])[0]

        # Stage 1: Candidate generation
        vector_chunks = self.vector_search_chunks(query_embedding, self.top_k)
        vector_topics = self.vector_search_topics(query_embedding, self.top_k)
        bm25_results = self.bm25_search(query, self.top_k)

        candidates = self.combine_candidates(vector_chunks, vector_topics, bm25_results)
        logger.info("Combined candidates: %d", len(candidates))

        # Stage 2: Graph expansion
        for candidate in candidates:
            context = self.graph_expand(candidate["chunk_id"])
            candidate["context"] = context
            candidate["expanded_text"] = self._build_expanded_text(candidate, context)

        return candidates

    @staticmethod
    def _build_expanded_text(candidate: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build a single text string from chunk + graph context."""
        parts = [
            f"Document: {context.get('title', candidate.get('title', ''))}",
            f"Type: {context.get('document_type', '')} | Number: {context.get('number', '')}",
            f"Topics: {', '.join(context.get('topics', []))}",
            f"Content: {candidate.get('text', '')}",
        ]
        if context.get("related_documents"):
            parts.append(f"Related documents: {', '.join(context['related_documents'])}")
        return "\n".join(parts)
