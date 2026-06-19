from src.search.search_client import SearchClient

client = SearchClient()
questions = [
    "ما هي قوانين الجمعيات في سلطنة عمان؟",
    "ما هي حقوق المؤلف في عمان؟",
    "ما هي قوانين الحجر الصحي للحيوانات؟",
]
for q in questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    print('='*60)
    result = client.search(q)
    print("Candidates found:", result['candidates_found'])
    print("Top contexts used:", len(result['top_contexts']))
    for idx, ctx in enumerate(result['top_contexts'], 1):
        title = ctx.get('context', {}).get('title', 'Unknown')
        doc_type = ctx.get('context', {}).get('document_type', '')
        print(f"  [{idx}] {title} ({doc_type})")
    print("\nAnswer:", result['answer'])
