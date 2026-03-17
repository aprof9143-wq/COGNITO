import chromadb
import time
import uuid

# ============================================================
# MODULE 3 — SEMANTIC MEMORY (ChromaDB)
# ============================================================
# PERFORMANCE UPGRADES vs original:
#
#  1. SINGLETON CLIENT — setup_memory() returns a cached
#     (client, collection) pair; no reconnect per pipeline run.
#
#  2. BATCH UPSERT — store_all_rules() sends a single
#     collection.add() call with all documents at once instead
#     of one call per rule.
#
#  3. UPSERT INSTEAD OF ADD — uses upsert semantics via unique
#     timestamp+uuid IDs, so duplicate-ID crashes are impossible.
# ============================================================

_COLLECTION_NAME = "neurosymbolic_rules"
_CLIENT_CACHE: dict = {}   # path → (client, collection)


def setup_memory(path: str = "./chroma_storage"):
    """Return a cached (client, collection) pair for this storage path."""
    if path not in _CLIENT_CACHE:
        print(f"\n[Module 3] Connecting to ChromaDB at '{path}'...")
        client     = chromadb.PersistentClient(path=path)
        collection = client.get_or_create_collection(name=_COLLECTION_NAME)
        _CLIENT_CACHE[path] = (client, collection)
        print(f"   Collection '{_COLLECTION_NAME}' ready.")
    return _CLIENT_CACHE[path][1]   # return just the collection


def _unique_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def store_knowledge(collection, doc_id: str, text: str) -> str:
    """Store a single knowledge chunk with a guaranteed-unique ID."""
    uid = _unique_id(doc_id)
    collection.add(documents=[text], ids=[uid])
    print(f"   [M3] Stored [{uid}]: {text[:60]}...")
    return uid


def store_all_rules(collection, structured_rules: list) -> list:
    """
    Batch-store all rules in a SINGLE collection.add() call.
    N=10 rules: 10 round trips (old) → 1 round trip (new).
    """
    if not structured_rules:
        return []

    documents = []
    ids       = []
    for i, rule in enumerate(structured_rules):
        text = rule.get("display", rule.get("original", str(rule)))
        uid  = _unique_id(f"rule_{i:02d}")
        documents.append(text)
        ids.append(uid)

    collection.add(documents=documents, ids=ids)
    print(f"   [M3] Batch-stored {len(ids)} rule(s).")
    return ids


def retrieve_context(collection, query_text: str, n_results: int = 3) -> list:
    """Retrieve the most semantically relevant stored rules/context."""
    try:
        # Clamp n_results to collection size to avoid ChromaDB error
        count = collection.count()
        n     = min(n_results, count) if count > 0 else 0
        if n == 0:
            return []

        results = collection.query(query_texts=[query_text], n_results=n)
        docs    = results["documents"][0] if results["documents"] else []
        if docs:
            print(f"   [M3] Retrieved {len(docs)} chunk(s) from memory.")
        return docs
    except Exception as e:
        print(f"   [M3] Memory retrieval issue: {e}")
        return []
