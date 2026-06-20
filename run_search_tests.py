"""Run search tests over the current Neo4j knowledge graph."""
import json
from src.search.search_client import SearchClient

QUESTIONS = [
    "ما هي قوانين الجمعيات في سلطنة عمان؟",
    "ما هي عقوبة القتل في القانون العماني؟",
    "ما هي شروط استخراج رخصة القيادة في سلطنة عمان؟",
    "ما هي قوانين الاستثمار الأجنبي في سلطنة عمان؟",
    "What are the labor laws in Oman?",
]


def main() -> None:
    client = SearchClient()
    results = []

    for q in QUESTIONS:
        print(f"Processing: {q[:60]}...")
        result = client.search(q)
        results.append(
            {
                "question": q,
                "candidates_found": result["candidates_found"],
                "top_contexts": len(result["top_contexts"]),
                "context_titles": [
                    ctx.get("context", {}).get("title", "Unknown")
                    for ctx in result["top_contexts"]
                ],
                "answer": result["answer"],
            }
        )

    with open("search_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Tests complete. Results saved to search_test_results.json")


if __name__ == "__main__":
    main()
