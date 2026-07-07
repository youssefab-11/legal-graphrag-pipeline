"""Check current Neo4j state."""
from src.ingestion.neo4j_client import get_neo4j_client
from src.config.settings import settings

client = get_neo4j_client()
driver = client.connect()
with driver.session(database=settings.NEO4J_DATABASE) as session:
    for label in ("Document", "Topic", "Chunk", "Community"):
        result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
        print(f"{label}: {result.single()['c']}")
    for rel in ("AMENDS", "REPEALS", "HAS_TOPIC", "HAS_CHUNK", "BELONGS_TO"):
        result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
        print(f"{rel}: {result.single()['c']}")
    # Check docs without topics
    result = session.run("MATCH (d:Document) WHERE NOT (d)-[:HAS_TOPIC]->() RETURN count(d) AS c")
    print(f"Documents WITHOUT topics: {result.single()['c']}")
    # Check topics without embeddings
    result = session.run("MATCH (t:Topic) WHERE t.embedding IS NULL RETURN count(t) AS c")
    print(f"Topics WITHOUT embeddings: {result.single()['c']}")
    # Check chunks without embeddings
    result = session.run("MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c) AS c")
    print(f"Chunks WITHOUT embeddings: {result.single()['c']}")
driver.close()
