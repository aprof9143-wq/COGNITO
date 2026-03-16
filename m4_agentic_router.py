import wikipedia
import google.generativeai as genai
from google import genai
import urllib.request
import urllib.parse
import json
import re

# ============================================================
# MODULE 4 — AGENTIC ROUTER: MULTI-SOURCE RESEARCH
# ============================================================
# Sources used (all free, no API key required):
#
#   1. Wikipedia  — depth on well-known topics, good for
#                   established numerical facts and definitions
#
#   2. DuckDuckGo Instant Answer API — broader coverage,
#                   handles queries Wikipedia misses (e.g.
#                   niche engineering topics), returns clean
#                   topic abstracts in JSON, no key needed.
#                   Endpoint: https://api.duckduckgo.com/
#
# Both sources are searched independently. Their contexts are
# returned separately so the caller can derive rules from each
# and aggregate them, with no cap on the number of rules.
# ============================================================


# ── PUBLIC ENTRY POINTS ──────────────────────────────────────────────────────

def research_all_sources(query_text: str, api_key: str = None) -> list:
    """
    Search ALL configured sources for the query and return a list of
    source result dicts, each with keys: source_name, context, reference, title.

    Returns results only for sources that found something useful.
    """
    results = []

    # ── Source 1: Wikipedia ──────────────────────────────────────────────────
    wiki_result = wiki_search_fallback(query_text, api_key=api_key)
    if wiki_result["context"] not in _FAILED_CONTEXTS:
        wiki_result["source_name"] = "Wikipedia"
        results.append(wiki_result)
    else:
        print("   [Module 4] Wikipedia: no usable result.")

    # ── Source 2: DuckDuckGo Instant Answer ──────────────────────────────────
    ddg_result = duckduckgo_search(query_text, api_key=api_key)
    if ddg_result["context"] not in _FAILED_CONTEXTS:
        # Deduplicate: skip DDG if it returned the same topic as Wikipedia
        if not _same_topic(ddg_result, wiki_result):
            ddg_result["source_name"] = "DuckDuckGo"
            results.append(ddg_result)
        else:
            print("   [Module 4] DuckDuckGo: returned same topic as Wikipedia — skipping duplicate.")
    else:
        print("   [Module 4] DuckDuckGo: no usable result.")

    return results


# ── SOURCE 1: WIKIPEDIA ──────────────────────────────────────────────────────

def wiki_search_fallback(query_text: str, api_key: str = None) -> dict:
    """
    Search Wikipedia for the most relevant article about query_text.
    Uses LLM to extract a clean search topic first if api_key provided.
    Returns {"context": str, "reference": url, "title": str}
    """
    search_query = _extract_search_topic(query_text, api_key, source_hint="Wikipedia")
    print(f"\n[Module 4] 🌐 Wikipedia search: '{search_query}'")

    try:
        search_results = wikipedia.search(search_query, results=5)
        if not search_results:
            return _failed("No Wikipedia results found.")

        for title_candidate in search_results:
            try:
                if not _is_relevant(title_candidate, search_query):
                    print(f"   [Module 4] Wikipedia: skipping off-topic result '{title_candidate}'")
                    continue

                page    = wikipedia.page(title_candidate, auto_suggest=False)
                summary = wikipedia.summary(title_candidate, sentences=5, auto_suggest=False)

                # ── LLM domain-relevance gate ──────────────────────────────
                # Word overlap can pass wrong-domain articles (e.g. "Spacecraft
                # design" for "3D printed torque adapter" — both contain "design").
                # Ask the LLM: is this article actually relevant to the query?
                if api_key and not _llm_domain_check(
                    article_title=page.title,
                    article_summary=summary,
                    search_query=search_query,
                    api_key=api_key
                ):
                    print(f"   [Module 4] Wikipedia: LLM domain gate rejected '{page.title}'")
                    continue

                print(f"   ✅ Wikipedia: {page.url}")
                return {"context": summary, "reference": page.url, "title": page.title}
            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page    = wikipedia.page(e.options[0], auto_suggest=False)
                    summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                    if api_key and not _llm_domain_check(
                        page.title, summary, search_query, api_key
                    ):
                        print(f"   [Module 4] Wikipedia: LLM domain gate rejected '{page.title}'")
                        continue
                    print(f"   ✅ Wikipedia: {page.url}")
                    return {"context": summary, "reference": page.url, "title": page.title}
                except Exception:
                    continue
            except Exception:
                continue

        print("   [Module 4] Wikipedia: no relevant article found after filtering.")
        return _failed("No relevant Wikipedia results found.")

    except Exception as e:
        print(f"   ❌ Wikipedia search failed: {e}")
        return _failed("Search failed.")


# ── SOURCE 2: DUCKDUCKGO INSTANT ANSWER ──────────────────────────────────────

def duckduckgo_search(query_text: str, api_key: str = None) -> dict:
    """
    Query the DuckDuckGo Instant Answer API for a topic summary.
    Completely free, no API key required.
    Returns {"context": str, "reference": url, "title": str}

    DDG Instant Answer returns:
      - AbstractText: a topic summary paragraph (best for rule derivation)
      - AbstractSource: source name (e.g. "Wikipedia", "Britannica")
      - AbstractURL: canonical source URL
      - RelatedTopics: list of related subtopics (used as fallback)
    """
    search_query = _extract_search_topic(query_text, api_key, source_hint="DuckDuckGo")
    print(f"\n[Module 4] 🦆 DuckDuckGo search: '{search_query}'")

    try:
        encoded_query = urllib.parse.quote_plus(search_query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1&no_html=1&skip_disambig=1"

        req = urllib.request.Request(url, headers={"User-Agent": "NeurosymbolicAI/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)

        # ── Try AbstractText first (best quality) ─────────────────────────
        abstract = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        abstract_source = data.get("AbstractSource", "DuckDuckGo")

        if abstract and len(abstract) > 80:
            title = data.get("Heading", search_query)
            print(f"   ✅ DuckDuckGo abstract from {abstract_source}: {abstract_url or 'N/A'}")
            return {
                "context"   : abstract,
                "reference" : abstract_url or f"https://duckduckgo.com/?q={encoded_query}",
                "title"     : title
            }

        # ── Fallback: try RelatedTopics snippets ──────────────────────────
        related = data.get("RelatedTopics", [])
        snippets = []
        for item in related:
            if isinstance(item, dict):
                text = item.get("Text", "").strip()
                if text and len(text) > 40:
                    snippets.append(text)
            # Some items are sub-lists (topic groups)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        text = sub.get("Text", "").strip()
                        if text and len(text) > 40:
                            snippets.append(text)

        if snippets:
            combined = " ".join(snippets[:4])  # Use up to 4 snippets
            print(f"   ✅ DuckDuckGo: using {len(snippets[:4])} related topic snippets")
            return {
                "context"   : combined,
                "reference" : f"https://duckduckgo.com/?q={encoded_query}",
                "title"     : data.get("Heading", search_query)
            }

        print("   [Module 4] DuckDuckGo: API returned no usable abstract or snippets.")
        return _failed("DuckDuckGo returned no usable content.")

    except Exception as e:
        err_str = str(e)
        # JSON parse error usually means DDG returned empty/HTML body for niche query.
        # Retry once with a broader 2-word query before giving up.
        if "Expecting value" in err_str or "JSONDecodeError" in err_str:
            try:
                broad_query = " ".join(search_query.split()[:2])
                print(f"   [Module 4] DuckDuckGo: retrying with broader query '{broad_query}'...")
                encoded_broad = urllib.parse.quote_plus(broad_query)
                retry_url = (f"https://api.duckduckgo.com/?q={encoded_broad}"
                             f"&format=json&no_redirect=1&no_html=1&skip_disambig=1")
                req2 = urllib.request.Request(retry_url,
                                              headers={"User-Agent": "NeurosymbolicAI/1.0"})
                with urllib.request.urlopen(req2, timeout=8) as r2:
                    raw2 = r2.read().decode("utf-8")
                data2 = json.loads(raw2)
                abstract2 = data2.get("AbstractText", "").strip()
                if abstract2 and len(abstract2) > 80:
                    title2 = data2.get("Heading", broad_query)
                    url2   = data2.get("AbstractURL", "") or f"https://duckduckgo.com/?q={encoded_broad}"
                    print(f"   ✅ DuckDuckGo (broad retry): {url2}")
                    return {"context": abstract2, "reference": url2, "title": title2}
            except Exception:
                pass
        print(f"   ❌ DuckDuckGo search failed: {e}")
        return _failed("Search failed.")


# ── SHARED HELPERS ────────────────────────────────────────────────────────────

_FAILED_CONTEXTS = {
    "Search failed.",
    "No Wikipedia results found.",
    "No relevant Wikipedia results found.",
    "DuckDuckGo returned no usable content.",
}


def _failed(reason: str) -> dict:
    return {"context": reason, "reference": "None", "title": "None"}


def _llm_domain_check(article_title: str, article_summary: str,
                      search_query: str, api_key: str) -> bool:
    """
    Ask the LLM whether a Wikipedia article's domain genuinely matches
    the search query. Returns True if relevant, False if wrong domain.

    This catches cases where word-overlap passes but the domain is wrong,
    e.g. "Spacecraft design" for "3D printed torque adapter" — both contain
    "design" but the domains are completely different.

    We keep the prompt minimal and the answer binary to minimise latency.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""Is the Wikipedia article titled "{article_title}" genuinely relevant to answering questions about "{search_query}"?

Article summary: "{article_summary[:300]}"

Answer with ONLY "yes" or "no". No explanation."""
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        return answer.startswith("yes")
    except Exception:
        return True  # On failure, default to accepting the article


def _extract_search_topic(query_text: str, api_key: str = None,
                           source_hint: str = "") -> str:
    """
    Use LLM to extract a precise 3-5 word search topic from the user query.
    Falls back to simple keyword extraction if no API key is provided.
    """
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""
Extract the core subject of this user request as a short search query (3-5 words max).
Focus on the TOPIC, not the user's personal details (age, weight, name etc.).
This query will be used on {source_hint or "a search engine"}.
Return ONLY the search query string — no explanation, no punctuation, no quotes.

User request: "{query_text}"

Examples:
  "i want a diet plan for bulking i am 22 and weigh 82kg" → muscle hypertrophy bulking nutrition
  "make me a study plan to score 1500 on the SAT" → SAT college admission test
  "design safety logic for a two-wheel balancing robot" → inverted pendulum balancing robot control
  "create a ramadan diet plan for fat loss" → Ramadan fasting nutrition
  "PID controller tuning guide for industrial temperature" → PID controller tuning methods

Search query:"""
            result = model.generate_content(prompt).text.strip()
            result = result.replace('"', '').replace("'", '').replace("\n", ' ').strip()
            return result[:60]
        except Exception:
            pass

    return _simple_clean(query_text)


def _simple_clean(query_text: str) -> str:
    """Fallback topic extractor without LLM."""
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


def _is_relevant(article_title: str, search_query: str,
                  accept_if_no_filter: bool = True) -> bool:
    """
    Relevance check: does the article title share at least one meaningful
    word/stem/acronym with the search query?

    Three-pass strategy:
      Pass 1 — exact word overlap              e.g. "safety" == "safety"
      Pass 2 — prefix containment              e.g. "safe" is prefix of "safety"
                                               (avoids false positives like
                                               interlock vs interface)
      Pass 3 — acronym expansion               e.g. "PLC" matches
                                               "Programmable Logic Controller"
    """
    stop_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for",
                  "and", "or", "with", "by", "from", "as", "is", "are",
                  "its", "it", "that", "this", "was", "were"}

    def _tokenise(text):
        tokens = re.split(r"[\s\-_/]+", text.lower())
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    query_tokens = _tokenise(search_query)
    title_tokens = _tokenise(article_title)

    if not query_tokens:
        return accept_if_no_filter

    qset = set(query_tokens)
    tset = set(title_tokens)

    # Pass 1: exact word match
    if qset & tset:
        return True

    # Pass 2: prefix containment — one word must fully contain the other as
    #         a prefix (e.g. "safe" is a prefix of "safety"), min 4 chars.
    #         Uses startswith so "interlock" never matches "interface".
    for qw in qset:
        if len(qw) < 4:
            continue
        for tw in tset:
            if len(tw) < 4:
                continue
            if tw.startswith(qw) or qw.startswith(tw):
                return True

    # Pass 3: acronym expansion — if a query word looks like an acronym
    #         (2-5 uppercase chars), check if it matches the initials of
    #         the title words. E.g. "PLC" matches "Programmable Logic Controller".
    raw_query_words  = search_query.split()
    raw_title_words  = article_title.split()
    title_initials   = "".join(w[0].upper() for w in raw_title_words if w[0].isalpha())

    for rw in raw_query_words:
        clean = re.sub(r"[^A-Z]", "", rw)          # keep only uppercase letters
        if 2 <= len(clean) <= 5 and clean == clean: # looks like an acronym
            if clean in title_initials:
                return True

    return False


def _same_topic(result_a: dict, result_b: dict) -> bool:
    """
    Check if two source results cover the same topic to avoid
    ingesting duplicate context.
    """
    title_a = result_a.get("title", "").lower().strip()
    title_b = result_b.get("title", "").lower().strip()
    if not title_a or not title_b or title_b == "none":
        return False
    # Same topic if titles share 2+ meaningful words
    stop_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and"}
    words_a = {w for w in title_a.split() if w not in stop_words and len(w) > 2}
    words_b = {w for w in title_b.split() if w not in stop_words and len(w) > 2}
    return len(words_a & words_b) >= 2
