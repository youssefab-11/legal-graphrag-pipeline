"""List AMENDS and REPEALS relationships in Neo4j."""
from src.ingestion.neo4j_client import get_neo4j_client
from src.config.settings import settings


def main() -> None:
    driver = get_neo4j_client().connect()
    with driver.session(database=settings.NEO4J_DATABASE) as s:
        repeals = s.run(
            "MATCH (s:Document)-[r:REPEALS]->(t:Document) "
            "RETURN s.id AS source, t.id AS target, t.number AS target_number"
        ).data()
        print("REPEALS relationships:")
        for r in repeals:
            print(f"  {r['source']} -> {r['target']} (number: {r['target_number']})")

        amends = s.run(
            "MATCH (s:Document)-[r:AMENDS]->(t:Document) "
            "RETURN s.id AS source, t.id AS target, t.number AS target_number"
        ).data()
        print("AMENDS relationships:")
        for r in amends:
            print(f"  {r['source']} -> {r['target']} (number: {r['target_number']})")


if __name__ == "__main__":
    main()
