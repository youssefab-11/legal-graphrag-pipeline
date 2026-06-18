"""Interactive search CLI for the Legal GraphRAG pipeline.

Performs hybrid retrieval, cross-encoder reranking, and LLM synthesis
to answer legal questions over the Omani legislation knowledge graph.
"""

import logging
from typing import Any, Dict, List

import ollama

from src.config.settings import settings
from src.search.retriever import HybridRetriever
from src.search.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class SearchClient:
    """CLI search client for hybrid GraphRAG queries."""

    def __init__(
        self,
        llm_model: str = settings.OLLAMA_LLM_MODEL,
        top_k_candidates: int = settings.TOP_K_CANDIDATES,
        top_k_final: int = settings.TOP_K_FINAL,
    ) -> None:
        self.llm_model = llm_model
        self.top_k_final = top_k_final
        self.retriever = HybridRetriever(top_k=top_k_candidates)
        self.reranker = CrossEncoderReranker()
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    def synthesize(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate a final answer using the local LLM and retrieved context.

        Args:
            query: User question.
            contexts: Top-N reranked candidate contexts.

        Returns:
            Generated answer string.
        """
        context_texts = []
        for idx, ctx in enumerate(contexts, 1):
            context_texts.append(
                f"[{idx}] {ctx.get('expanded_text', ctx.get('text', ''))}"
            )

        full_context = "\n\n".join(context_texts)

        system_prompt = (
            "You are a legal assistant specializing in Omani legislation. "
            "Use ONLY the provided legal context to answer the question. "
            "If the context does not contain enough information, say so. "
            "Be concise, accurate, and cite relevant document numbers when possible."
        )

        user_prompt = (
            f"Question: {query}\n\n"
            f"Legal Context:\n{full_context}\n\n"
            "Answer the question based on the legal context above."
        )

        try:
            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response["message"]["content"]
        except Exception as exc:
            logger.error("LLM synthesis failed: %s", exc, exc_info=True)
            return f"Sorry, I could not generate an answer at this time. Error: {exc}"

    def search(self, query: str) -> Dict[str, Any]:
        """Execute a full search: retrieve, rerank, synthesize.

        Args:
            query: User question.

        Returns:
            Dictionary with query, candidates, reranked contexts, and answer.
        """
        candidates = self.retriever.retrieve(query)
        top_contexts = self.reranker.rerank(query, candidates, top_n=self.top_k_final)
        answer = self.synthesize(query, top_contexts)

        return {
            "query": query,
            "candidates_found": len(candidates),
            "top_contexts": top_contexts,
            "answer": answer,
        }

    def run_interactive(self) -> None:
        """Run the interactive CLI loop."""
        print("\n" + "=" * 60)
        print("  Legal GraphRAG Search Client")
        print("  Type your question or 'exit' to quit.")
        print("=" * 60 + "\n")

        while True:
            query = input("Question: ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Goodbye.")
                break

            result = self.search(query)

            print("\n" + "-" * 60)
            print(f"Candidates found: {result['candidates_found']}")
            print(f"Top contexts used: {len(result['top_contexts'])}")
            print("\nGraph Context:")
            for idx, ctx in enumerate(result["top_contexts"], 1):
                title = ctx.get("context", {}).get("title", "Unknown")
                doc_type = ctx.get("context", {}).get("document_type", "")
                print(f"  [{idx}] {title} ({doc_type})")

            print("\nAnswer:\n", result["answer"])
            print("-" * 60 + "\n")


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=settings.LOG_LEVEL)

    client = SearchClient()
    client.run_interactive()


if __name__ == "__main__":
    main()
