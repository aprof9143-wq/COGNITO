import openai
import wikipedia
import urllib.request
import urllib.parse
import json
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# MODULE 4 — AGENTIC ROUTER: MULTI-SOURCE RESEARCH
# ============================================================
# PERFORMANCE UPGRADES vs original:
#
#  1. CONCURRENT SOURCES — Wikipedia and DuckDuckGo are fetched
#     IN PARALLEL via ThreadPoolExecutor.
#     2 serial fetches (~4-8s) → ~2-4s concurrent.
#
#  2. SINGLE TOPIC EXTRACTION — _extract_search_topic() is
#     called ONCE and shared between both source searches,
#     eliminating a redundant LLM call.
#
#  3. SHARED CLIENT — same cached OpenAI client as m2 so no
#     extra TLS handshake cost.
#
#  4. TOPIC CACHE — same query string skips re-extraction.
# ============================================================

_MODEL       = "gpt-5-mini-2025-08-07"
_TOPIC_CACHE : dict = {}   # query_hash → search_topic string
_CLIENT_CACHE: dict = {}   # api_key_hash → openai.OpenAI


def _get_client(api_key: str) -> openai.OpenAI:
    key = (api_key or "").strip()
    if not key:
        raise ValueError(
            "OpenAI API key is empty. Paste your sk- key in the UI "
            "or set OPENAI_API_KEY in your environment."
        )
    h = hashlib.md5(key.encode()).hexdigest()
    if h not in _CLIENT_CACHE:
        _CLIENT_CACHE[h] = openai.OpenAI(api_key=key)
    return _CLIENT_CACHE[h]


def _gpt(api_key: str, prompt: str, max_completion_tokens: int = 256) -> str:
    client   = _get_client(api_key)
    response = client.chat.completions.create(
        model=_MODEL,
        max_completion_tokens=max_completion_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError(
            f"GPT returned no content "
            f"(finish_reason='{response.choices[0].finish_reason}')."
        )
    return content


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def research_all_sources(query_text: str, api_key: str = None) -> list:
    """
    Search Wikipedia AND DuckDuckGo CONCURRENTLY.
    Returns list of source result dicts: {source_name, context, reference, title}.
    """
    # Extract search topic ONCE, share between both sources
    search_topic = _extract_search_topic(query_text, api_key)
    print(f"\n[Module 4] Search topic: '{search_topic}'")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        wiki_future = pool.submit(_wiki_fetch, search_topic, query_text, api_key)
        ddg_future  = pool.submit(_ddg_fetch,  search_topic)

        wiki_result = wiki_future.result()
        ddg_result  = ddg_future.result()

    elapsed = time.perf_counter() - t0
    print(f"   [M4] Both sources fetched in {elapsed:.2f}s (concurrent)")

    results = []

    if wiki_result["context"] not in _FAILED_CONTEXTS:
        wiki_result["source_name"] = "Wikipedia"
        results.append(wiki_result)
    else:
        print("   [M4] Wikipedia: no usable result.")

    if ddg_result["context"] not in _FAILED_CONTEXTS:
        if not _same_topic(ddg_result, wiki_result):
            ddg_result["source_name"] = "DuckDuckGo"
            results.append(ddg_result)
        else:
            print("   [M4] DuckDuckGo: same topic as Wikipedia — skipping duplicate.")
    else:
        print("   [M4] DuckDuckGo: no usable result.")

    return results


# ── SOURCE 1: WIKIPEDIA ──────────────────────────────────────────────────────

def _wiki_fetch(search_query: str, original_query: str, api_key: str = None) -> dict:
    """Fetch best Wikipedia result for search_query."""
    print(f"   [M4] Wikipedia: '{search_query}'")
    try:
        candidates = wikipedia.search(search_query, results=5)
        if not candidates:
            return _failed("No Wikipedia results found.")

        for title in candidates:
            try:
                if not _is_relevant(title, search_query):
                    continue
                page    = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(title, sentences=5, auto_suggest=False)
                if api_key and not _llm_domain_check(page.title, summary,
                                                      search_query, api_key):
                    continue
                print(f"   Wikipedia: {page.url}")
                return {"context": summary, "reference": page.url, "title": page.title}

            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page    = wikipedia.page(e.options[0], auto_suggest=False)
                    summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                    if api_key and not _llm_domain_check(page.title, summary,
                                                          search_query, api_key):
                        continue
                    return {"context": summary, "reference": page.url, "title": page.title}
                except Exception:
                    continue
            except Exception:
                continue

        return _failed("No relevant Wikipedia results found.")
    except Exception as e:
        print(f"   Wikipedia failed: {e}")
        return _failed("Search failed.")


# Kept as public alias for backward compatibility
def wiki_search_fallback(query_text: str, api_key: str = None) -> dict:
    topic = _extract_search_topic(query_text, api_key)
    return _wiki_fetch(topic, query_text, api_key)


# ── SOURCE 2: DUCKDUCKGO INSTANT ANSWER ──────────────────────────────────────

def _ddg_fetch(search_query: str) -> dict:
    """Fetch DuckDuckGo Instant Answer for search_query."""
    print(f"   [M4] DuckDuckGo: '{search_query}'")
    try:
        encoded = urllib.parse.quote_plus(search_query)
        url     = (f"https://api.duckduckgo.com/?q={encoded}"
                   f"&format=json&no_redirect=1&no_html=1&skip_disambig=1")
        req = urllib.request.Request(url, headers={"User-Agent": "NeurosymbolicAI/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        abstract     = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        abstract_src = data.get("AbstractSource", "DuckDuckGo")

        if abstract and len(abstract) > 80:
            print(f"   DuckDuckGo abstract ({abstract_src}): {abstract_url or 'N/A'}")
            return {
                "context"  : abstract,
                "reference": abstract_url or f"https://duckduckgo.com/?q={encoded}",
                "title"    : data.get("Heading", search_query),
            }

        # Fallback: RelatedTopics snippets
        snippets = []
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict):
                t = item.get("Text", "").strip()
                if t and len(t) > 40:
                    snippets.append(t)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        t = sub.get("Text", "").strip()
                        if t and len(t) > 40:
                            snippets.append(t)

        if snippets:
            print("   DuckDuckGo: related topics fallback")
            return {
                "context"  : " ".join(snippets[:4]),
                "reference": f"https://duckduckgo.com/?q={encoded}",
                "title"    : search_query,
            }

        return _failed("DuckDuckGo returned no usable content.")
    except Exception as e:
        print(f"   DuckDuckGo failed: {e}")
        return _failed("Search failed.")


# Kept as public alias
def duckduckgo_search(query_text: str, api_key: str = None) -> dict:
    topic = _extract_search_topic(query_text, api_key)
    return _ddg_fetch(topic)


# ── SHARED HELPERS ────────────────────────────────────────────────────────────

_FAILED_CONTEXTS = {
    "Search failed.",
    "No Wikipedia results found.",
    "No relevant Wikipedia results found.",
    "DuckDuckGo returned no usable content.",
}


def _failed(reason: str) -> dict:
    return {"context": reason, "reference": "None", "title": "None"}


def _extract_search_topic(query_text: str, api_key: str = None) -> str:
    """Extract a concise 3-5 word search topic. Cached per query."""
    cache_key = hashlib.md5(query_text.strip().lower().encode()).hexdigest()
    if cache_key in _TOPIC_CACHE:
        return _TOPIC_CACHE[cache_key]

    if api_key:
        try:
            prompt = (
                f'Extract the core subject of this request as a short search query (3-5 words max).\n'
                f'Focus on TOPIC only — ignore personal details (age, weight, name etc.).\n'
                f'Return ONLY the search query string, no explanation, no quotes.\n\n'
                f'Request: "{query_text}"\n\n'
                f'Examples:\n'
                f'  "diet plan for bulking, 22yo 82kg" → muscle hypertrophy bulking nutrition\n'
                f'  "study plan to score 1500 SAT" → SAT college admission test preparation\n\n'
                f'Search query:'
            )
            result = _gpt(api_key, prompt, max_completion_tokens=32)
            result = result.replace('"', "").replace("'", "").replace("\n", " ").strip()[:60]
            _TOPIC_CACHE[cache_key] = result
            return result
        except Exception:
            pass

    result = _simple_clean(query_text)
    _TOPIC_CACHE[cache_key] = result
    return result


def _simple_clean(query_text: str) -> str:
    filler = [
        "i want to", "i want you to", "i need to", "i would like to",
        "help me", "can you", "please", "create", "make", "build",
        "generate", "write", "give me", "provide", "show me",
        "i am", "i weigh", "years old", "kg", "lbs",
    ]
    q = query_text.lower().strip()
    for phrase in filler:
        q = q.replace(phrase, " ")
    words = [w for w in q.split() if len(w) > 2][:6]
    return " ".join(words)


def _llm_domain_check(article_title: str, summary: str,
                       query: str, api_key: str) -> bool:
    """Ask GPT whether a Wikipedia article is relevant to the search query."""
    try:
        prompt = (
            f'Is the Wikipedia article "{article_title}" relevant to "{query}"?\n'
            f'Summary: "{summary[:300]}"\n'
            f'Answer ONLY "yes" or "no".'
        )
        answer = _gpt(api_key, prompt, max_completion_tokens=10)
        return answer.strip().lower().startswith("yes")
    except Exception:
        return True


def _is_relevant(article_title: str, search_query: str) -> bool:
    """Token-overlap relevance check (no LLM needed)."""
    stop = {"the","a","an","of","in","on","at","to","for","and","or",
            "with","by","from","as","is","are","its","it","that","this","was","were"}

    def tok(text):
        return [t for t in re.split(r"[\s\-_/]+", text.lower())
                if t not in stop and len(t) > 2]

    qt = set(tok(search_query))
    tt = set(tok(article_title))
    if not qt:
        return True
    if qt & tt:
        return True
    for qw in qt:
        for tw in tt:
            if len(qw) >= 4 and len(tw) >= 4:
                if tw.startswith(qw) or qw.startswith(tw):
                    return True
    return False


def _same_topic(a: dict, b: dict) -> bool:
    stop = {"the","a","an","of","in","on","at","to","for","and"}
    wa = {w for w in a.get("title","").lower().split() if w not in stop and len(w) > 2}
    wb = {w for w in b.get("title","").lower().split() if w not in stop and len(w) > 2}
    return len(wa & wb) >= 2
