import time
import uuid
import json
import hashlib
from datetime import datetime

# ============================================================
# MODULE 3 — SYMBOLIC MEMORY (Qdrant)
# ============================================================
# Uses Qdrant as the vector database for symbolic references.
# Supports both local in-memory mode and remote Qdrant server.
#
# Each stored entry is a "symbolic reference" — a rich record
# capturing not just the text but the full symbolic structure:
# rule type, operator, threshold, scope, confidence, timestamp.
#
# This metadata powers the Second Brain visualisation in app.py.
#
# Collections:
#   - "rules"   : structured constraint records
#   - "sources" : research context chunks
#   - "audit"   : audit result records per run
# ============================================================

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue, ScrollRequest,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Embedding dimension for text-embedding-3-small
_EMBED_DIM   = 1536
_COLLECTIONS = ["rules", "sources", "audit"]

# ── Client singleton ──────────────────────────────────────────────────────────
_CLIENT: "QdrantClient | None" = None


def _get_client(url: str = None, api_key: str = None) -> "QdrantClient":
    """Return singleton Qdrant client. Defaults to in-memory if no URL given."""
    global _CLIENT
    if _CLIENT is None:
        if url:
            _CLIENT = QdrantClient(url=url, api_key=api_key or None)
            print(f"[M3] Connected to Qdrant at {url}")
        else:
            _CLIENT = QdrantClient(":memory:")
            print("[M3] Using in-memory Qdrant (no URL configured)")
    return _CLIENT


def setup_memory(url: str = None, openai_api_key: str = None,
                 qdrant_api_key: str = None):
    """
    Initialise Qdrant collections. Returns the client.
    Call once per session before storing anything.
    """
    if not QDRANT_AVAILABLE:
        raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")

    client = _get_client(url=url, api_key=qdrant_api_key)

    existing = {c.name for c in client.get_collections().collections}
    for name in _COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )
            print(f"[M3] Created collection '{name}'")

    return client


def _embed(text: str, openai_api_key: str) -> list:
    """Embed text with OpenAI text-embedding-3-small. Returns float list."""
    import openai
    client   = openai.OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000],
    )
    return response.data[0].embedding


def _point_id() -> str:
    """Generate a UUID-based point ID."""
    return str(uuid.uuid4())


# ── Store functions ───────────────────────────────────────────────────────────

def store_rule(client, rule: dict, openai_api_key: str) -> str:
    """
    Store a structured rule as a symbolic reference in the 'rules' collection.
    The payload carries full symbolic metadata for Second Brain visualisation.
    """
    text  = rule.get("display", rule.get("original", str(rule)))
    pid   = _point_id()
    ts    = datetime.utcnow().isoformat()

    vector  = _embed(text, openai_api_key)
    payload = {
        "text"            : text,
        "original"        : rule.get("original", text),
        "variable"        : rule.get("variable", ""),
        "constraint_type" : rule.get("constraint_type", "boolean"),
        "operator"        : rule.get("operator", ""),
        "threshold"       : rule.get("threshold"),
        "threshold_low"   : rule.get("threshold_low"),
        "threshold_high"  : rule.get("threshold_high"),
        "unit"            : rule.get("unit", ""),
        "scope"           : rule.get("scope", "always"),
        "rule_nature"     : rule.get("rule_nature", "constraint"),
        "stored_at"       : ts,
        "record_type"     : "rule",
    }

    client.upsert(collection_name="rules",
                  points=[PointStruct(id=pid, vector=vector, payload=payload)])
    print(f"   [M3] Stored rule [{pid[:8]}…]: {text[:55]}")
    return pid


def store_all_rules(client, structured_rules: list, openai_api_key: str) -> list:
    """Batch-embed and store all rules. Returns list of point IDs."""
    if not structured_rules:
        return []

    texts    = [r.get("display", r.get("original", str(r))) for r in structured_rules]
    vectors  = _batch_embed(texts, openai_api_key)
    ts       = datetime.utcnow().isoformat()
    points   = []
    ids      = []

    for rule, vec, text in zip(structured_rules, vectors, texts):
        pid = _point_id()
        ids.append(pid)
        points.append(PointStruct(
            id      = pid,
            vector  = vec,
            payload = {
                "text"            : text,
                "original"        : rule.get("original", text),
                "variable"        : rule.get("variable", ""),
                "constraint_type" : rule.get("constraint_type", "boolean"),
                "operator"        : rule.get("operator", ""),
                "threshold"       : rule.get("threshold"),
                "threshold_low"   : rule.get("threshold_low"),
                "threshold_high"  : rule.get("threshold_high"),
                "unit"            : rule.get("unit", ""),
                "scope"           : rule.get("scope", "always"),
                "rule_nature"     : rule.get("rule_nature", "constraint"),
                "stored_at"       : ts,
                "record_type"     : "rule",
            }
        ))

    client.upsert(collection_name="rules", points=points)
    print(f"   [M3] Batch-stored {len(points)} rule(s) in Qdrant.")
    return ids


def store_source(client, source: dict, openai_api_key: str) -> str:
    """Store a research source chunk in the 'sources' collection."""
    text  = source.get("context", "")[:2000]
    pid   = _point_id()
    ts    = datetime.utcnow().isoformat()

    vector  = _embed(text, openai_api_key)
    payload = {
        "text"        : text,
        "title"       : source.get("title", ""),
        "reference"   : source.get("reference", ""),
        "source_name" : source.get("source_name", ""),
        "stored_at"   : ts,
        "record_type" : "source",
    }
    client.upsert(collection_name="sources",
                  points=[PointStruct(id=pid, vector=vector, payload=payload)])
    return pid


def store_audit_result(client, audit_result: dict, openai_api_key: str,
                        run_id: str = "") -> str:
    """Store an audit result record in the 'audit' collection."""
    text = (f"{audit_result.get('rule_display','')} — "
            f"{'PASS' if audit_result.get('satisfies') else 'FAIL'}: "
            f"{audit_result.get('explanation','')}")
    pid  = _point_id()
    ts   = datetime.utcnow().isoformat()

    vector  = _embed(text, openai_api_key)
    payload = {
        "text"                 : text,
        "rule_display"         : audit_result.get("rule_display", ""),
        "satisfies"            : audit_result.get("satisfies", False),
        "scope"                : audit_result.get("scope", ""),
        "extracted_value_raw"  : audit_result.get("extracted_value_raw", ""),
        "extracted_value_num"  : audit_result.get("extracted_value_num"),
        "explanation"          : audit_result.get("explanation", ""),
        "symbolic_check_used"  : audit_result.get("symbolic_check_used", False),
        "premise_confidence"   : audit_result.get("premise_confidence", 1.0),
        "conclusion_confidence": audit_result.get("conclusion_confidence", 0.0),
        "domain_warning"       : audit_result.get("domain_warning", ""),
        "run_id"               : run_id,
        "stored_at"            : ts,
        "record_type"          : "audit",
    }
    client.upsert(collection_name="audit",
                  points=[PointStruct(id=pid, vector=vector, payload=payload)])
    return pid


# ── Retrieve functions ────────────────────────────────────────────────────────

def retrieve_context(client, query_text: str, openai_api_key: str,
                     n_results: int = 4) -> list:
    """Semantic search across 'rules' and 'sources'. Returns text snippets."""
    try:
        vec   = _embed(query_text, openai_api_key)
        texts = []
        for col in ("rules", "sources"):
            count = client.get_collection(col).points_count
            if count == 0:
                continue
            hits = client.search(
                collection_name=col,
                query_vector=vec,
                limit=min(n_results, count),
            )
            texts.extend(h.payload.get("text", "") for h in hits)
        print(f"   [M3] Retrieved {len(texts)} chunk(s) from Qdrant.")
        return texts[:n_results]
    except Exception as e:
        print(f"   [M3] Retrieval error: {e}")
        return []


# ── Second Brain inspection ───────────────────────────────────────────────────

def get_all_records(client, collection: str, limit: int = 200) -> list:
    """
    Fetch all stored points from a collection for Second Brain visualisation.
    Returns list of payload dicts (no vectors).
    """
    try:
        result = client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [pt.payload for pt in result[0]]
    except Exception as e:
        print(f"   [M3] scroll error ({collection}): {e}")
        return []


def get_collection_stats(client) -> dict:
    """Return point counts per collection for the dashboard."""
    stats = {}
    for name in _COLLECTIONS:
        try:
            stats[name] = client.get_collection(name).points_count
        except Exception:
            stats[name] = 0
    return stats


def store_knowledge(client, doc_id: str, text: str,
                    openai_api_key: str = None) -> str:
    """Legacy-compatible single-doc store into 'sources'."""
    src = {"context": text, "title": doc_id,
           "reference": "", "source_name": doc_id}
    if openai_api_key:
        return store_source(client, src, openai_api_key)
    return doc_id   # no-op if no key


# ── Internal helpers ──────────────────────────────────────────────────────────

def _batch_embed(texts: list, openai_api_key: str) -> list:
    """Embed a list of texts in one API call."""
    import openai
    client   = openai.OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[t[:8000] for t in texts],
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
