import openai
import json
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# MODULE 2 — LLM PARSER & STRUCTURED CONSTRAINT AUDITOR
# ============================================================
# PERFORMANCE UPGRADES vs original:
#
#  1. SHARED CLIENT — openai.OpenAI() created once per api_key,
#     cached in _CLIENT_CACHE. Eliminates repeated TLS handshake
#     overhead on every call.
#
#  2. PARALLEL RULE PARSING — parse_rules_parallel() sends all
#     N rule-parse requests concurrently via ThreadPoolExecutor.
#     N=5 rules: ~5s serial → ~1.2s parallel.
#
#  3. BATCHED AUDIT — structured_audit() sends ONE prompt
#     containing ALL rules at once instead of N separate calls.
#     N=5 rules: ~5 API round trips → 1 round trip.
#     Falls back to parallel individual calls if batch parse fails.
#
#  4. RESULT CACHE — identical rules are cached by content hash
#     so re-running the same rule set hits zero API calls.
# ============================================================

_MODEL       = "gpt-5-mini-2025-08-07"
_MAX_WORKERS = 8          # parallel threads for rule parsing
_AUDIT_BATCH = 12         # max rules per single batch-audit call

# ── Shared client cache (keyed by api_key hash) ──────────────────────────────
_CLIENT_CACHE: dict = {}
_PARSE_CACHE:  dict = {}   # rule_text_hash → parsed constraint dict


def _get_client(api_key: str) -> openai.OpenAI:
    """Return a cached OpenAI client for this key."""
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


def _gpt(api_key: str, prompt: str, max_completion_tokens: int = 4096) -> str:
    """Single-turn GPT call using cached client."""
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


# ── Rule parsing ──────────────────────────────────────────────────────────────

_PARSE_SCHEMA = """{
    "original"        : "<the original rule text>",
    "rule_nature"     : "<constraint | observation>",
    "variable"        : "<snake_case identifier>",
    "constraint_type" : "<numerical_upper_bound | numerical_lower_bound | numerical_range | boolean | categorical>",
    "operator"        : "<  < | <= | > | >= | == | in_range | contains | excludes>",
    "threshold"       : <primary numeric threshold as float, or null>,
    "threshold_low"   : <lower bound for in_range, or null>,
    "threshold_high"  : <upper bound for in_range, or null>,
    "unit"            : "<unit string or empty string>",
    "display"         : "<concise human-readable form>",
    "scope"           : "<always | initial | final | maximum | minimum | conditional | context_only>",
    "scope_hint"      : "<which occurrence to check>",
    "extraction_hint" : "<how to find this value in generated text>"
}"""


def parse_rule_to_constraint(rule_text: str, api_key: str) -> dict:
    """Parse a single natural-language rule → structured constraint dict.
    Results are cached by rule text hash."""
    cache_key = hashlib.md5(rule_text.strip().encode()).hexdigest()
    if cache_key in _PARSE_CACHE:
        print(f"   [M2 cache hit] {rule_text[:50]}")
        return _PARSE_CACHE[cache_key]

    prompt = f"""You are a formal logic compiler. Convert this rule into a JSON constraint.

RULE: "{rule_text}"

Return ONLY raw JSON (no markdown, no backticks) with this exact schema:
{_PARSE_SCHEMA}

RULE_NATURE: 'constraint' = must be enforced | 'observation' = input context fact
SCOPE: always|initial|final|maximum|minimum|conditional|context_only"""

    try:
        raw    = _gpt(api_key, prompt, max_completion_tokens=512)
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        _PARSE_CACHE[cache_key] = result
        return result
    except Exception:
        slug = re.sub(r"[^a-z0-9]+", "_", rule_text.lower().strip())[:40].strip("_") or "rule"
        fallback = {
            "original": rule_text, "variable": slug,
            "constraint_type": "boolean", "operator": "==",
            "threshold": None, "threshold_low": None, "threshold_high": None,
            "unit": "", "display": rule_text, "scope": "always",
            "rule_nature": "constraint",
            "scope_hint": f"Verify: {rule_text}",
            "extraction_hint": f"Check whether text satisfies: {rule_text}",
        }
        _PARSE_CACHE[cache_key] = fallback
        return fallback


def parse_rules_parallel(rule_texts: list, api_key: str) -> list:
    """Parse ALL rules concurrently. N rules in ~time-of-1 rule."""
    if not rule_texts:
        return []

    results   = [None] * len(rule_texts)
    uncached  = [(i, r) for i, r in enumerate(rule_texts)
                 if hashlib.md5(r.strip().encode()).hexdigest() not in _PARSE_CACHE]

    # Fill cached ones instantly
    for i, r in enumerate(rule_texts):
        h = hashlib.md5(r.strip().encode()).hexdigest()
        if h in _PARSE_CACHE:
            results[i] = _PARSE_CACHE[h]

    if not uncached:
        return results

    t0 = time.perf_counter()
    workers = min(_MAX_WORKERS, len(uncached))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(parse_rule_to_constraint, rule, api_key): idx
            for idx, rule in uncached
        }
        for future in as_completed(futures):
            idx         = futures[future]
            results[idx] = future.result()

    elapsed = time.perf_counter() - t0
    print(f"   [M2] Parsed {len(uncached)} rule(s) in parallel: {elapsed:.2f}s")
    return results


# ── Batched audit ─────────────────────────────────────────────────────────────

def _build_scope_instruction(rule: dict) -> str:
    scope      = rule.get("scope", "always")
    scope_hint = rule.get("scope_hint", "")
    nature     = rule.get("rule_nature", "constraint")
    if scope == "context_only" or nature == "observation":
        return (f"SCOPE=OBSERVATION: Given fact, not hard constraint. "
                f"Check output USES this value as starting context. {scope_hint}")
    scope_map = {
        "initial"     : "SCOPE=INITIAL: Check ONLY first/starting value.",
        "final"       : "SCOPE=FINAL: Check ONLY last/final value.",
        "maximum"     : "SCOPE=MAXIMUM: Find and check the single highest value.",
        "minimum"     : "SCOPE=MINIMUM: Find and check the single lowest value.",
        "conditional" : f"SCOPE=CONDITIONAL: Only applies when condition met. {scope_hint}",
        "always"      : "SCOPE=ALL: Check every occurrence; any violation = fail.",
    }
    base = scope_map.get(scope, "SCOPE=ALL: Check every occurrence.")
    return f"{base} {scope_hint}".strip()


def structured_audit(draft_text: str, structured_rules: list, api_key: str) -> list:
    """
    Audit ALL rules in a SINGLE batched GPT call.
    Falls back to parallel individual calls if the batch JSON is malformed.

    Speed: N=5 rules → 5 API calls (old) → 1 API call (new).
    """
    if not structured_rules:
        return []

    # Split into chunks to respect context limits
    all_results = []
    for chunk_start in range(0, len(structured_rules), _AUDIT_BATCH):
        chunk = structured_rules[chunk_start:chunk_start + _AUDIT_BATCH]
        chunk_results = _batch_audit_chunk(draft_text, chunk,
                                           chunk_start, api_key)
        all_results.extend(chunk_results)
    return all_results


def _batch_audit_chunk(draft_text: str, rules: list,
                        offset: int, api_key: str) -> list:
    """Send one batch-audit prompt for a chunk of rules."""

    rules_block = ""
    for i, rule in enumerate(rules):
        idx = offset + i
        scope_instr = _build_scope_instruction(rule)
        unit = rule.get("unit", "").strip() or "same unit as threshold"
        rules_block += f"""
RULE_{idx}:
  original:    "{rule.get('original', '')}"
  display:     {rule.get('display', rule.get('original', ''))}
  unit:        {unit}
  {scope_instr}
  hint:        {rule.get('extraction_hint', '')}
"""

    prompt = f"""You are a strict constraint auditor for a neurosymbolic AI system.

DRAFT TEXT:
\"\"\"{draft_text[:5000]}\"\"\"

RULES TO AUDIT:
{rules_block}

For EACH rule above, extract the relevant value from the draft and check compliance.

UNIT NORMALIZATION: If the draft uses different units than the constraint, convert before comparing.
  Example: "45 minutes" with unit "hours" → extracted_value_num = 0.75

Return ONLY a raw JSON array (no markdown, no backticks) with one object per rule, in order:
[
  {{
    "rule_index"           : <int — matches RULE_N index above>,
    "extracted_value_raw"  : "<exact phrase found, e.g. 'initial Kp: 0.1'>",
    "extracted_value_num"  : <float in constraint unit, or null if non-numerical>,
    "unit_conversion_note" : "<e.g. 'converted 45 min to 0.75 hours' or 'none'>",
    "scope_note"           : "<which occurrence you checked and why>",
    "satisfies"            : <true or false>,
    "explanation"          : "<one sentence>"
  }},
  ...
]

RULES:
- Respect SCOPE — wrong occurrence = verification error
- Normalise units BEFORE comparing
- If variable not found → satisfies=false
- Strict: rule says < 10 and value == 10 → VIOLATION"""

    try:
        t0  = time.perf_counter()
        raw = _gpt(api_key, prompt, max_completion_tokens=4096)
        elapsed = time.perf_counter() - t0
        print(f"   [M2] Batch audit {len(rules)} rule(s) in {elapsed:.2f}s")

        clean    = raw.strip().replace("```json", "").replace("```", "").strip()
        llm_list = json.loads(clean)

        # Build a lookup by rule_index
        by_index = {item["rule_index"]: item for item in llm_list}

        return [_build_result(offset + i, rule, by_index.get(offset + i))
                for i, rule in enumerate(rules)]

    except Exception as e:
        print(f"   [M2] Batch audit failed ({e}), falling back to parallel individual calls.")
        return _parallel_audit_fallback(draft_text, rules, offset, api_key)


def _parallel_audit_fallback(draft_text: str, rules: list,
                              offset: int, api_key: str) -> list:
    """Fallback: audit rules in parallel individual calls."""
    results = [None] * len(rules)
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(rules))) as pool:
        futures = {
            pool.submit(_single_audit, draft_text, rule, offset + i, api_key): i
            for i, rule in enumerate(rules)
        }
        for future in as_completed(futures):
            i          = futures[future]
            results[i] = future.result()
    return results


def _single_audit(draft_text: str, rule: dict, idx: int, api_key: str) -> dict:
    """Audit a single rule (used in fallback path)."""
    scope_instr = _build_scope_instruction(rule)
    unit        = rule.get("unit", "").strip() or "same unit as threshold"
    rule_display = rule.get("display", rule.get("original", ""))

    prompt = f"""You are a strict constraint auditor.

RULE: "{rule.get('original', '')}"
CONSTRAINT: {rule_display}
UNIT: {unit}
{scope_instr}
HINT: {rule.get('extraction_hint', '')}

DRAFT:
\"\"\"{draft_text[:4000]}\"\"\"

Return ONLY raw JSON:
{{
  "extracted_value_raw": "<exact phrase>",
  "extracted_value_num": <float or null>,
  "unit_conversion_note": "<note>",
  "scope_note": "<which occurrence>",
  "satisfies": <true|false>,
  "explanation": "<one sentence>"
}}"""

    try:
        raw     = _gpt(api_key, prompt, max_completion_tokens=512)
        clean   = raw.strip().replace("```json","").replace("```","").strip()
        llm_res = json.loads(clean)
    except Exception:
        llm_res = {
            "extracted_value_raw": "EXTRACTION FAILED",
            "extracted_value_num": None,
            "satisfies": False,
            "explanation": "GPT extraction failed — treating as violation.",
        }
    return _build_result(idx, rule, llm_res)


def _build_result(idx: int, rule: dict, llm_res: dict | None) -> dict:
    """Merge LLM audit result with symbolic checks and domain validation."""
    if llm_res is None:
        llm_res = {
            "extracted_value_raw": "NO RESULT",
            "extracted_value_num": None,
            "satisfies": False,
            "explanation": "No audit result returned.",
        }

    rule_display = rule.get("display", rule.get("original", ""))
    scope        = rule.get("scope", "always")

    num_val = llm_res.get("extracted_value_num")
    op      = rule.get("operator")
    thresh  = rule.get("threshold")
    t_low   = rule.get("threshold_low")
    t_high  = rule.get("threshold_high")

    # ── Symbolic double-check ────────────────────────────────────────────────
    symbolic_override = None
    if num_val is not None and op is not None:
        try:
            num_val = float(num_val)
            if   op == "<"        and thresh  is not None: symbolic_override = num_val <  float(thresh)
            elif op == "<="       and thresh  is not None: symbolic_override = num_val <= float(thresh)
            elif op == ">"        and thresh  is not None: symbolic_override = num_val >  float(thresh)
            elif op == ">="       and thresh  is not None: symbolic_override = num_val >= float(thresh)
            elif op == "=="       and thresh  is not None: symbolic_override = num_val == float(thresh)
            elif op == "in_range" and t_low is not None and t_high is not None:
                symbolic_override = float(t_low) <= num_val <= float(t_high)
        except (TypeError, ValueError):
            symbolic_override = None

    final_satisfies = (symbolic_override if symbolic_override is not None
                       else llm_res.get("satisfies", False))

    # ── Domain validity check ────────────────────────────────────────────────
    domain_warning = ""
    if num_val is not None:
        unit_tokens = set(re.split(r"[_\s]+", rule.get("unit", "").lower()))
        var_tokens  = set(re.split(r"[_\s]+", rule.get("variable", "").lower()))
        all_tokens  = unit_tokens | var_tokens
        if all_tokens & {"proportion", "ratio", "fraction"} and num_val > 1.0:
            domain_warning  = (f"DOMAIN VIOLATION: '{rule.get('variable','')}' is a proportion "
                               f"but extracted value {num_val} > 1.0 is impossible.")
            final_satisfies = False
        elif all_tokens & {"probability", "prob"} and (num_val > 1.0 or num_val < 0.0):
            domain_warning  = (f"DOMAIN VIOLATION: '{rule.get('variable','')}' is a probability "
                               f"but {num_val} is outside [0,1].")
            final_satisfies = False
        elif (all_tokens & {"percent", "confidence"} or "%" in all_tokens):
            if num_val > 100.0 or num_val < 0.0:
                domain_warning  = (f"DOMAIN VIOLATION: '{rule.get('variable','')}' is a percentage "
                                   f"but {num_val} is outside [0,100].")
                final_satisfies = False

    status = "PASS" if final_satisfies else "FAIL"
    method = "(symbolic)" if symbolic_override is not None else "(semantic)"
    print(f"   R{idx+1} {status} {method} — {rule_display[:60]}")
    if domain_warning:
        print(f"        DOMAIN: {domain_warning}")

    return {
        "rule_id"              : idx,
        "rule_display"         : rule_display,
        "original_rule"        : rule.get("original", ""),
        "scope"                : scope,
        "extracted_value_raw"  : llm_res.get("extracted_value_raw", "N/A"),
        "extracted_value_num"  : num_val,
        "unit_conversion_note" : llm_res.get("unit_conversion_note", ""),
        "scope_note"           : llm_res.get("scope_note", ""),
        "domain_warning"       : domain_warning,
        "satisfies"            : final_satisfies,
        "symbolic_check_used"  : symbolic_override is not None,
        "premise_confidence"   : 1.0,
        "conclusion_confidence": 1.0 if final_satisfies else 0.05,
        "explanation"          : llm_res.get("explanation", "No explanation."),
    }


def audit_results_to_ltn_entities(audit_results: list) -> dict:
    """Convert audit results into the entity format that m1_ltn_core expects."""
    return {
        "entities": [
            {
                "name"                 : r["rule_display"],
                "premise_confidence"   : r["premise_confidence"],
                "conclusion_confidence": r["conclusion_confidence"],
            }
            for r in audit_results
        ]
    }


# ── Legacy wrapper ────────────────────────────────────────────────────────────
def extract_universal_facts(document_text, active_rule, api_key):
    prompt = (f'GOVERNING RULE: "{active_rule}"\n'
              f'DOCUMENT: "{document_text}"\n'
              'For each entity/claim assess premise_confidence and conclusion_confidence (0-1).\n'
              'Return ONLY raw JSON: {"entities":[{"name":"...","premise_confidence":0.0,"conclusion_confidence":0.0}]}')
    try:
        raw   = _gpt(api_key, prompt, max_completion_tokens=512)
        clean = raw.strip().replace("```json","").replace("```","")
        return json.loads(clean)
    except Exception:
        return None
