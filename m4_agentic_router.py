import anthropic
import wikipedia
import urllib.request
import urllib.parse
import json
import re

# ============================================================
# MODULE 4 — AGENTIC ROUTER: MULTI-SOURCE RESEARCH
# ============================================================
# Rewritten to use Anthropic Claude instead of Google Gemini.
# Sources: Wikipedia + DuckDuckGo Instant Answer (both free).
# ============================================================

_MODEL = "claude-sonnet-4-20250514"


def _claude(api_key: str, prompt: str, max_tokens: int = 256) -> str:
    """Thin wrapper: single-turn Claude call, returns response text."""
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── PUBLIC ENTRY POINTS ──────────────────────────────────────────────────────

def research_all_sources(query_text: str, api_key: str = None) -> list:
    """
    Search ALL configured sources and return a list of source result dicts,
    each with keys: source_name, context, reference, title.
    """
    results = []

    # Source 1: Wikipedia
    wiki_result = wiki_search_fallback(query_text, api_key=api_key)
    if wiki_result["context"] not in _FAILED_CONTEXTS:
        wiki_result["source_name"] = "Wikipedia"
        results.append(wiki_result)
    else:
        print("   [Module 4] Wikipedia: no usable result.")

    # Source 2: DuckDuckGo Instant Answer
    ddg_result = duckduckgo_search(query_text, api_key=api_key)
    if ddg_result["context"] not in _FAILED_CONTEXTS:
        if not _same_topic(ddg_result, wiki_result):
            ddg_result["source_name"] = "DuckDuckGo"
            results.append(ddg_result)
        else:
            print("   [Module 4] DuckDuckGo: same topic as Wikipedia — skipping duplicate.")
    else:
        print("   [Module 4] DuckDuckGo: no usable result.")

    return results


# ── SOURCE 1: WIKIPEDIA ──────────────────────────────────────────────────────

def wiki_search_fallback(query_text: str, api_key: str = None) -> dict:
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

                if api_key and not _llm_domain_check(page.title, summary, search_query, api_key):
                    print(f"   [Module 4] Wikipedia: LLM domain gate rejected '{page.title}'")
                    continue

                print(f"   ✅ Wikipedia: {page.url}")
                return {"context": summary, "reference": page.url, "title": page.title}

            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page    = wikipedia.page(e.options[0], auto_suggest=False)
                    summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                    if api_key and not _llm_domain_check(page.title, summary, search_query, api_key):
                        continue
                    print(f"   ✅ Wikipedia: {page.url}")
                    return {"context": summary, "reference": page.url, "title": page.title}
                except Exception:
                    continue
            except Exception:
                continue

        return _failed("No relevant Wikipedia results found.")

    except Exception as e:
        print(f"   ❌ Wikipedia search failed: {e}")
        return _failed("Search failed.")


# ── SOURCE 2: DUCKDUCKGO INSTANT ANSWER ──────────────────────────────────────

def duckduckgo_search(query_text: str, api_key: str = None) -> dict:
    search_query = _extract_search_topic(query_text, api_key, source_hint="DuckDuckGo")
    print(f"\n[Module 4] 🦆 DuckDuckGo search: '{search_query}'")

    try:
        encoded_query = urllib.parse.quote_plus(search_query)
        url = (f"https://api.duckduckgo.com/?q={encoded_query}"
               f"&format=json&no_redirect=1&no_html=1&skip_disambig=1")

        req = urllib.request.Request(url, headers={"User-Agent": "NeurosymbolicAI/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))

        abstract     = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        abstract_src = data.get("AbstractSource", "DuckDuckGo")

        if abstract and len(abstract) > 80:
            title = data.get("Heading", search_query)
            print(f"   ✅ DuckDuckGo abstract from {abstract_src}: {abstract_url or 'N/A'}")
            return {
                "context"   : abstract,
                "reference" : abstract_url or f"https://duckduckgo.com/?q={encoded_query}",
                "title"     : title,
            }

        # Fallback: RelatedTopics snippets
        related  = data.get("RelatedTopics", [])
        snippets = []
        for item in related:
            if isinstance(item, dict):
                text = item.get("Text", "").strip()
                if text and len(text) > 40:
                    snippets.append(text)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        text = sub.get("Text", "").strip()
                        if text and len(text) > 40:
                            snippets.append(text)

        if snippets:
            combined = " ".join(snippets[:4])
            print(f"   ✅ DuckDuckGo (related topics fallback)")
            return {
                "context"   : combined,
                "reference" : f"https://duckduckgo.com/?q={encoded_query}",
                "title"     : search_query,
            }

        return _failed("DuckDuckGo returned no usable content.")

    except Exception as e:
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
    """Ask Claude whether a Wikipedia article is relevant to the search query."""
    try:
        prompt = (
            f'Is the Wikipedia article titled "{article_title}" genuinely relevant to '
            f'answering questions about "{search_query}"?\n\n'
            f'Article summary: "{article_summary[:300]}"\n\n'
            f'Answer with ONLY "yes" or "no". No explanation.'
        )
        answer = _claude(api_key, prompt, max_tokens=10)
        return answer.strip().lower().startswith("yes")
    except Exception:
        return True  # default accept on failure


def _extract_search_topic(query_text: str, api_key: str = None,
                           source_hint: str = "") -> str:
    """Extract a concise 3-5 word search topic from the user query."""
    if api_key:
        try:
            prompt = f"""Extract the core subject of this user request as a short search query (3-5 words max).
Focus on the TOPIC, not the user's personal details (age, weight, name etc.).
This query will be used on {source_hint or "a search engine"}.
Return ONLY the search query string — no explanation, no punctuation, no quotes.

User request: "{query_text}"

Examples:
  "i want a diet plan for bulking i am 22 and weigh 82kg" → muscle hypertrophy bulking nutrition
  "make me a study plan to score 1500 on the SAT" → SAT college admission test
  "design safety logic for a two-wheel balancing robot" → inverted pendulum balancing robot control

Search query:"""
            result = _claude(api_key, prompt, max_tokens=32)
            result = result.replace('"', "").replace("'", "").replace("\n", " ").strip()
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
    """Check whether an article title shares meaningful words with the search query."""
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

    if qset & tset:
        return True

    for qw in qset:
        if len(qw) < 4:
            continue
        for tw in tset:
            if len(tw) < 4:
                continue
            if tw.startswith(qw) or qw.startswith(tw):
                return True

    raw_title_words = article_title.split()
    title_initials  = "".join(w[0].upper() for w in raw_title_words if w[0].isalpha())
    for rw in search_query.split():
        clean = re.sub(r"[^A-Z]", "", rw)
        if 2 <= len(clean) <= 5 and clean == clean:
            if clean in title_initials:
                return True

    return False


def _same_topic(result_a: dict, result_b: dict) -> bool:
    """Check if two source results cover the same topic."""
    title_a = result_a.get("title", "").lower().strip()
    title_b = result_b.get("title", "").lower().strip()
    if not title_a or not title_b or title_b == "none":
        return False
    stop_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and"}
    words_a = {w for w in title_a.split() if w not in stop_words and len(w) > 2}
    words_b = {w for w in title_b.split() if w not in stop_words and len(w) > 2}
    return len(words_a & words_b) >= 2
