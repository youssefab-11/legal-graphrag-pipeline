"""Environment connectivity checker.

Verifies that Neo4j and Ollama are reachable and that required models are
available before running the full pipeline.
"""

import logging

import ollama
from neo4j import GraphDatabase

from src.config.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL)


def check_neo4j() -> bool:
    """Verify Neo4j connectivity."""
    logger.info("Checking Neo4j connection at %s...", settings.NEO4J_URI)
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        driver.verify_connectivity()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 AS ok")
            record = result.single()
            assert record["ok"] == 1
        driver.close()
        logger.info("✅ Neo4j connection successful.")
        return True
    except Exception as exc:
        logger.error("❌ Neo4j connection failed: %s", exc)
        return False


def check_ollama() -> bool:
    """Verify Ollama connectivity and required models."""
    logger.info("Checking Ollama connection at %s...", settings.OLLAMA_BASE_URL)
    try:
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        models_response = client.list()
        available_models = {m["model"] for m in models_response.get("models", [])}

        logger.info("Available Ollama models: %s", sorted(available_models))

        required_models = [
            settings.OLLAMA_LLM_MODEL,
            settings.OLLAMA_EMBEDDING_MODEL,
        ]

        missing = [m for m in required_models if m not in available_models]
        if missing:
            logger.error("❌ Missing Ollama models: %s", missing)
            logger.info("Run the following commands:")
            for m in missing:
                logger.info("  ollama pull %s", m)
            return False

        logger.info("✅ Ollama connection successful. Required models available.")
        return True
    except Exception as exc:
        logger.error("❌ Ollama connection failed: %s", exc)
        logger.info("Make sure Ollama is running: ollama serve")
        return False


def main() -> None:
    logger.info("=" * 60)
    logger.info("Legal GraphRAG Pipeline - Setup Check")
    logger.info("=" * 60)

    neo4j_ok = check_neo4j()
    ollama_ok = check_ollama()

    if neo4j_ok and ollama_ok:
        logger.info("=" * 60)
        logger.info("✅ All systems ready. You can run the pipeline now.")
        logger.info("   python run_pipeline.py")
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("⚠️  Please fix the issues above before running the pipeline.")
        logger.warning("=" * 60)


if __name__ == "__main__":
    main()
