import chromadb
import time
import uuid

# ============================================================
# MODULE 3 — SEMANTIC MEMORY (ChromaDB)
# ============================================================
# Stores approved rules and context as vector embeddings so the
# system can semantically retrieve relevant constraints later.
#
# FIX: Old code used a hardcoded 'dynamic_rule_01' ID — this
# caused a crash on the 2nd run ("duplicate ID" error).
# Now IDs are unique timestamps + UUIDs.
# ============================================================


def setup_memory():
    print("\n[Module 3] Initializing Semantic Memory (ChromaDB)...")
    client = chromadb.PersistentClient(path="./chroma_storage")
    collection = client.get_or_create_collection(name="neurosymbolic_rules")
    return collection


def store_knowledge(collection, doc_id: str, text: str):
    """
    Store a rule/knowledge chunk with a unique ID.
    Uses timestamp + uuid suffix to avoid duplicate key collisions.
    """
    unique_id = f"{doc_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    collection.add(
        documents=[text],
        ids=[unique_id]
    )
    print(f"💾 Ingested into Memory [{unique_id}]: {text[:60]}...")
    return unique_id


def store_all_rules(collection, structured_rules: list) -> list:
    """Store each structured rule individually. Returns list of stored IDs."""
    stored_ids = []
    for i, rule in enumerate(structured_rules):
        rule_text = rule.get("display", rule.get("original", str(rule)))
        stored_id = store_knowledge(collection, f"rule_{i:02d}", rule_text)
        stored_ids.append(stored_id)
    return stored_ids


def retrieve_context(collection, query_text: str, n_results: int = 3):
    """Retrieve the most semantically relevant stored rules/context."""
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        if results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            print(f"🧠 Retrieved {len(docs)} rule(s) from Memory.")
            return docs
    except Exception as e:
        print(f"⚠️  Memory retrieval issue: {e}")
    return []
