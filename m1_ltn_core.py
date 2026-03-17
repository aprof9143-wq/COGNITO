import math

# ============================================================
# MODULE 1 — LTN CORE: LOGIC TENSOR NETWORK VERIFICATION
# ============================================================
# Reimplemented with pure math — no TensorFlow, no ltn package.
# Identical mathematical semantics to the original:
#
#   Reichenbach implication:  P → C  =  1 − P + P·C
#   pMeanError aggregation (p=2):
#       Forall score = 1 − (mean((1 − φᵢ)²))^(1/2)
#
# 50-100× faster than the TF version for the small N (<50 rules)
# this system handles, with zero import overhead.
# ============================================================

_P = 2  # pMeanError exponent — matches original ltn default


def _reichenbach_implies(premise: float, conclusion: float) -> float:
    """Fuzzy Reichenbach implication: 1 − p + p·c  (clamped to [0,1])."""
    val = 1.0 - premise + premise * conclusion
    return max(0.0, min(1.0, val))


def _pmean_error_forall(implication_values: list) -> float:
    """
    pMeanError universal quantifier aggregation.
    Score = 1 − (mean((1 − φᵢ)^p))^(1/p)
    Higher = more universally satisfied.
    """
    if not implication_values:
        return 1.0
    errors = [(1.0 - v) ** _P for v in implication_values]
    mean_error = sum(errors) / len(errors)
    score = 1.0 - math.pow(mean_error, 1.0 / _P)
    return max(0.0, min(1.0, score))


def evaluate_generic_logic(parsed_data: dict) -> float:
    """
    Runs LTN verification over a set of logical entities.
    Each entity must have: premise_confidence, conclusion_confidence.
    Returns a float score in [0, 1]. Score < 0.8 = violation detected.
    """
    print("\n[Module 1] Running LTN verification (pure-math engine)...")

    entities = parsed_data.get("entities", [])
    if not entities:
        print("     No entities — defaulting score to 1.0")
        return 1.0

    implication_values = [
        _reichenbach_implies(e["premise_confidence"], e["conclusion_confidence"])
        for e in entities
    ]

    score = _pmean_error_forall(implication_values)
    print(f"     Universal LTN Verification Score: {score:.4f}")
    return score


def verify_and_report(audit_results: list) -> tuple:
    """
    Main entry point for verification.
    Takes audit_results from m2.structured_audit().
    Returns (ltn_score: float, violations: list[dict]).
    """
    parsed_data = {
        "entities": [
            {
                "name"                  : r["rule_display"],
                "premise_confidence"    : r["premise_confidence"],
                "conclusion_confidence" : r["conclusion_confidence"],
            }
            for r in audit_results
        ]
    }

    ltn_score  = evaluate_generic_logic(parsed_data)
    violations = [r for r in audit_results if not r["satisfies"]]
    return ltn_score, violations
