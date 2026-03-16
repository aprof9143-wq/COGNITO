import anthropic
import json
import re

# ============================================================
# MODULE 2 — LLM PARSER & STRUCTURED CONSTRAINT AUDITOR
# ============================================================
# Rewritten to use Anthropic Claude instead of Google Gemini.
# All genai.configure / GenerativeModel calls replaced with
# anthropic.Anthropic client + client.messages.create().
# ============================================================

_MODEL = "claude-sonnet-4-20250514"


def _claude(api_key: str, prompt: str, max_tokens: int = 1024) -> str:
    """Thin wrapper: send a single-turn prompt to Claude, return response text."""
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def parse_rule_to_constraint(rule_text: str, api_key: str) -> dict:
    """
    Parse a natural-language rule into a structured constraint dict.
    """
    prompt = f"""
You are a formal logic compiler. Convert the natural-language rule below into a strict
structured JSON constraint object.

RULE: "{rule_text}"

Return ONLY raw JSON (no markdown, no backticks, no explanation) in this exact schema:
{{
    "original"        : "<the original rule text>",
    "rule_nature"     : "<one of: constraint | observation>",
    "variable"        : "<snake_case identifier for what is being constrained or observed>",
    "constraint_type" : "<one of: numerical_upper_bound | numerical_lower_bound | numerical_range | boolean | categorical>",
    "operator"        : "<one of: < | <= | > | >= | == | in_range | contains | excludes>",
    "threshold"       : <primary numeric threshold as float, or null if not applicable>,
    "threshold_low"   : <lower bound for in_range, or null>,
    "threshold_high"  : <upper bound for in_range, or null>,
    "unit"            : "<unit string, e.g. hours, points, calories, or empty string>",
    "display"         : "<concise human-readable form, e.g. study_time_per_day < 10 hours/day>",
    "scope"           : "<one of: always | initial | final | maximum | minimum | conditional | context_only>",
    "scope_hint"      : "<precise instruction to the auditor on WHICH occurrence to check>",
    "extraction_hint" : "<how to find this value in generated text, combined with scope>"
}}

RULE_NATURE — this is the most important field, choose carefully:
- 'constraint'  : a rule that MUST be enforced — the output MUST satisfy it everywhere it applies
- 'observation' : a current real-world fact/reading provided as INPUT CONTEXT

SCOPE RULES — choose carefully:
- 'always'       : the constraint applies to every occurrence
- 'initial'      : only the FIRST/STARTING value matters
- 'final'        : only the LAST/FINAL value matters
- 'maximum'      : check the single highest value mentioned
- 'minimum'      : check the single lowest value mentioned
- 'conditional'  : the rule only applies when a condition is met
- 'context_only' : for observations — the value is given context to USE, not enforce everywhere
"""
    try:
        raw = _claude(api_key, prompt, max_tokens=1024)
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        slug = re.sub(r"[^a-z0-9]+", "_", rule_text.lower().strip())[:40].strip("_") or "rule_satisfied"
        return {
            "original"        : rule_text,
            "variable"        : slug,
            "constraint_type" : "boolean",
            "operator"        : "==",
            "threshold"       : None,
            "threshold_low"   : None,
            "threshold_high"  : None,
            "unit"            : "",
            "display"         : rule_text,
            "scope"           : "always",
            "rule_nature"     : "constraint",
            "scope_hint"      : f"Verify the output satisfies: {rule_text}",
            "extraction_hint" : f"Check whether the text satisfies: {rule_text}",
        }


def structured_audit(draft_text: str, structured_rules: list, api_key: str) -> list:
    """
    For each structured rule, extract the relevant value from the draft and
    check whether the constraint is satisfied using symbolic + LLM verification.
    Returns a list of audit result dicts.
    """
    audit_results = []

    for idx, rule in enumerate(structured_rules):
        rule_display    = rule.get("display", rule.get("original", ""))
        constraint_unit = rule.get("unit", "").strip() or "the same unit as the constraint threshold"
        scope           = rule.get("scope", "always")
        scope_hint      = rule.get("scope_hint", "")
        rule_nature     = rule.get("rule_nature", "constraint")

        print(f"\n[Module 2] Auditing Rule {idx+1}: {rule_display}")

        # Build scope instruction
        if scope == "context_only" or rule_nature == "observation":
            scope_instruction = (
                f"SCOPE — OBSERVATION/CONTEXT: This is a GIVEN FACT, not a hard constraint. "
                f"The value ({rule.get('threshold', 'as stated')}) is provided as INPUT CONTEXT. "
                f"Check that the output USES this value appropriately as a starting condition. "
                f"Do NOT fail just because other parts of the output use different values. {scope_hint}"
            )
        elif scope == "initial":
            scope_instruction = (
                f"SCOPE — INITIAL VALUE ONLY: This rule applies ONLY to the first/starting value. "
                f"Do NOT check later, adjusted, or final values. {scope_hint}"
            )
        elif scope == "final":
            scope_instruction = (
                f"SCOPE — FINAL VALUE ONLY: This rule applies ONLY to the last/final/settled value. "
                f"Ignore starting or intermediate values. {scope_hint}"
            )
        elif scope == "maximum":
            scope_instruction = (
                f"SCOPE — MAXIMUM VALUE: Find the single highest value mentioned for this variable. "
                f"That is the value to check. {scope_hint}"
            )
        elif scope == "minimum":
            scope_instruction = (
                f"SCOPE — MINIMUM VALUE: Find the single lowest value mentioned for this variable. "
                f"That is the value to check. {scope_hint}"
            )
        elif scope == "conditional":
            scope_instruction = (
                f"SCOPE — CONDITIONAL: This rule only applies under a specific condition. {scope_hint}"
            )
        else:  # always
            scope_instruction = (
                f"SCOPE — ALL OCCURRENCES: Check every occurrence of this variable. "
                f"If any occurrence violates the constraint, set satisfies=false. {scope_hint}"
            )

        extraction_prompt = f"""
You are a strict constraint auditor for a neurosymbolic AI system.

RULE BEING CHECKED: "{rule.get('original', '')}"
FORMAL CONSTRAINT:  {rule_display}
CONSTRAINT UNIT:    {constraint_unit}
{scope_instruction}
HINT:               {rule.get('extraction_hint', '')}

DRAFT TEXT:
\"\"\"
{draft_text[:4000]}
\"\"\"

Your job:
1. Using the SCOPE instruction above, find the CORRECT occurrence of this variable in the draft.
2. CRITICAL — UNIT NORMALIZATION: If the draft expresses the value in a DIFFERENT unit
   than the constraint, you MUST convert it before returning extracted_value_num.
   Examples: "45 minutes" with constraint unit "hours" -> extracted_value_num = 0.75
   extracted_value_raw = what you literally found (e.g. "Kp: 0.1 (initial)")
   extracted_value_num = converted to {constraint_unit}
3. Determine whether the scoped, converted value satisfies the formal constraint.

Return ONLY raw JSON (no markdown, no backticks):
{{
    "extracted_value_raw"  : "<exact phrase + context, e.g. 'initial Kp: 0.1'>",
    "extracted_value_num"  : <float in constraint unit ({constraint_unit}), or null if non-numerical>,
    "unit_conversion_note" : "<e.g. 'converted 45 min to 0.75 hours' or 'no conversion needed'>",
    "scope_note"           : "<which occurrence you checked and why>",
    "satisfies"            : <true or false>,
    "explanation"          : "<one sentence explaining the result>"
}}

IMPORTANT RULES:
- Respect the SCOPE — checking the wrong occurrence is a verification error.
- Always normalise units BEFORE comparing.
- If the variable is never mentioned, set satisfies=false.
- Be strict. If rule says < 10 and converted value equals 10, that is a VIOLATION.
"""
        try:
            raw = _claude(api_key, extraction_prompt, max_tokens=512)
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            llm_result = json.loads(clean)
        except Exception:
            llm_result = {
                "extracted_value_raw" : "EXTRACTION FAILED",
                "extracted_value_num" : None,
                "satisfies"           : False,
                "explanation"         : "Claude extraction failed — treating as violation.",
            }

        # ── Symbolic double-check for numerical constraints ───────────────────
        symbolic_override = None
        num_val = llm_result.get("extracted_value_num")
        op      = rule.get("operator")
        thresh  = rule.get("threshold")
        t_low   = rule.get("threshold_low")
        t_high  = rule.get("threshold_high")

        if num_val is not None and op is not None:
            try:
                num_val = float(num_val)
                if   op == "<"        and thresh is not None: symbolic_override = num_val < float(thresh)
                elif op == "<="       and thresh is not None: symbolic_override = num_val <= float(thresh)
                elif op == ">"        and thresh is not None: symbolic_override = num_val > float(thresh)
                elif op == ">="       and thresh is not None: symbolic_override = num_val >= float(thresh)
                elif op == "=="       and thresh is not None: symbolic_override = num_val == float(thresh)
                elif op == "in_range" and t_low is not None and t_high is not None:
                    symbolic_override = float(t_low) <= num_val <= float(t_high)
            except (TypeError, ValueError):
                symbolic_override = None

        final_satisfies = symbolic_override if symbolic_override is not None else llm_result.get("satisfies", False)

        # ── Domain validity check ─────────────────────────────────────────────
        domain_warning = ""
        if num_val is not None:
            unit_tokens = set(re.split(r"[_\s]+", rule.get("unit", "").lower()))
            var_tokens  = set(re.split(r"[_\s]+", rule.get("variable", "").lower()))
            all_tokens  = unit_tokens | var_tokens

            if all_tokens & {"proportion", "ratio", "fraction"}:
                if num_val > 1.0:
                    domain_warning  = (f"⚠️  DOMAIN VIOLATION: '{rule.get('variable','')}' is a proportion "
                                       f"but extracted value {num_val} > 1.0 is impossible.")
                    final_satisfies = False
            elif all_tokens & {"probability", "prob"}:
                if num_val > 1.0 or num_val < 0.0:
                    domain_warning  = (f"⚠️  DOMAIN VIOLATION: '{rule.get('variable','')}' is a probability "
                                       f"but {num_val} is outside [0,1].")
                    final_satisfies = False
            elif all_tokens & {"percent", "confidence"} or "%" in all_tokens:
                if num_val > 100.0 or num_val < 0.0:
                    domain_warning  = (f"⚠️  DOMAIN VIOLATION: '{rule.get('variable','')}' is a percentage "
                                       f"but {num_val} is outside [0,100].")
                    final_satisfies = False

        premise_confidence    = 1.0
        conclusion_confidence = 1.0 if final_satisfies else 0.05

        result = {
            "rule_id"               : idx,
            "rule_display"          : rule_display,
            "original_rule"         : rule.get("original", ""),
            "scope"                 : scope,
            "extracted_value_raw"   : llm_result.get("extracted_value_raw", "N/A"),
            "extracted_value_num"   : num_val,
            "unit_conversion_note"  : llm_result.get("unit_conversion_note", ""),
            "scope_note"            : llm_result.get("scope_note", ""),
            "domain_warning"        : domain_warning,
            "satisfies"             : final_satisfies,
            "symbolic_check_used"   : symbolic_override is not None,
            "premise_confidence"    : premise_confidence,
            "conclusion_confidence" : conclusion_confidence,
            "explanation"           : llm_result.get("explanation", "No explanation."),
        }

        status = "✅ PASS" if final_satisfies else "❌ FAIL"
        method = "(symbolic)" if symbolic_override is not None else "(semantic LLM)"
        print(f"           ├─ Scope     : {scope.upper()}")
        print(f"           ├─ Extracted : {llm_result.get('extracted_value_raw', 'N/A')}")
        print(f"           ├─ Check     : {rule_display} {method}")
        if domain_warning:
            print(f"           ├─ ⚠️  DOMAIN : {domain_warning}")
        print(f"           └─ Result    : {status}  — {llm_result.get('explanation', '')}")

        audit_results.append(result)

    return audit_results


def audit_results_to_ltn_entities(audit_results: list) -> dict:
    """Convert audit results into the entity format that m1_ltn_core expects."""
    return {
        "entities": [
            {
                "name"                  : r["rule_display"],
                "premise_confidence"    : r["premise_confidence"],
                "conclusion_confidence" : r["conclusion_confidence"],
            }
            for r in audit_results
        ]
    }


# ── Legacy wrapper (kept for backward compatibility) ─────────────────────────
def extract_universal_facts(document_text, active_rule, api_key):
    prompt = f"""
    GOVERNING RULE: "{active_rule}"
    DOCUMENT: "{document_text}"
    For each entity/claim, assess premise_confidence and conclusion_confidence (0-1).
    Return ONLY raw JSON: {{"entities": [{{"name": "...", "premise_confidence": 0.0, "conclusion_confidence": 0.0}}]}}
    """
    try:
        raw   = _claude(api_key, prompt, max_tokens=512)
        clean = raw.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except Exception:
        return None
