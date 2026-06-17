"""Cross-Encoder reranker for hybrid retrieval results.

Scores expanded candidate texts against the user query using a local
cross-encoder model, producing a refined top-N ranking.
"""

import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from src.config.settings import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Local cross-encoder reranker for final candidate scoring."""

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            logger.info("Loading cross-encoder model: %s", self.model_name)
            self.model = CrossEncoder(self.model_name)
        except Exception as exc:
            logger.error("Failed to load cross-encoder model: %s", exc)
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = settings.TOP_K_FINAL,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: Original user query.
            candidates: List of expanded candidate dictionaries.
            top_n: Number of top candidates to return.

        Returns:
            Reranked list of top-N candidates.
        """
        if not self.model or not candidates:
            logger.warning("Cross-encoder unavailable. Returning candidates as-is.")
            return candidates[:top_n]

        pairs = [(query, c.get("expanded_text", c.get("text", ""))) for c in candidates]
        try:
            scores = self.model.predict(pairs)
            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return ranked[:top_n]
        except Exception as exc:
            logger.error("Reranking failed: %s", exc)
            return candidates[:top_n]
