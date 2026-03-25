"""
Run once to seed Qdrant with credit event corpus.
Usage: python data/seed_qdrant.py
"""
import sys
sys.path.insert(0, '.')

from engine.evidence import (
    get_qdrant_client, get_embedding_model,
    setup_collection, seed_events, retrieve_evidence
)

if __name__ == "__main__":
    print("Connecting to Qdrant...")
    client = get_qdrant_client()
    model  = get_embedding_model()

    setup_collection(client)
    seed_events(client, model)

    # Test retrieval
    print("\n=== TEST RETRIEVAL ===")
    query = "DHFL bond yield spike 25% April 2019 extreme spread widening default"
    results = retrieve_evidence(query, client, model, top_k=3)
    for r in results:
        print(f"\n[score={r['score']}] {r['title']} ({r['date']})")
        print(f"  {r['text'][:120]}...")