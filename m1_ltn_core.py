import ltn
import tensorflow as tf

# ============================================================
# MODULE 1 — LTN CORE: LOGIC TENSOR NETWORK VERIFICATION
# ============================================================
# The LTN evaluates whether generated content obeys all logical
# constraints by running a fuzzy-logic formula over the audit
# results produced by Module 2.
#
# Key formula: ∀ rule: Premise(rule) → Conclusion(rule)
# = For every rule, IF it applies THEN it must be satisfied.
#
# When m2 produces real confidence scores (1.0 for pass, 0.05 for fail),
# any violation causes a significant score drop, making failures visible.
# ============================================================


def evaluate_generic_logic(parsed_data: dict) -> float:
    """
    Runs LTN verification over a set of logical entities.
    Each entity must have: premise_confidence, conclusion_confidence.
    Returns a float score in [0, 1]. Score < 0.8 = violation detected.
    """
    print("\n[Module 1] Initializing LTN Engine...")

    data_matrix = []
    for entity in parsed_data['entities']:
        data_matrix.append([
            entity['premise_confidence'],
            entity['conclusion_confidence']
        ])

    packet_tensor = tf.constant(data_matrix, dtype=tf.float32)
    entities = ltn.Variable('entities', packet_tensor)

    class ExtractPremise(tf.keras.Model):
        def call(self, inputs):
            return inputs[:, 0:1]

    class ExtractConclusion(tf.keras.Model):
        def call(self, inputs):
            return inputs[:, 1:2]

    is_premise_met     = ltn.Predicate(ExtractPremise())
    is_conclusion_valid = ltn.Predicate(ExtractConclusion())

    Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
    Forall  = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")

    formula = Forall(
        entities,
        Implies(is_premise_met(entities), is_conclusion_valid(entities))
    )

    score = float(formula.tensor.numpy())
    print(f"⚖️  Universal LTN Verification Score: {score:.4f}")
    return score


def verify_and_report(audit_results: list) -> tuple:
    """
    Main entry point for verification.
    Takes audit_results from m2.structured_audit().
    Returns (ltn_score: float, violations: list[dict]).

    violations is a list of rules that FAILED, with details for the rewrite prompt.
    """
    # Build entity matrix from audit results
    parsed_data = {
        "entities": [
            {
                "name"                  : r["rule_display"],
                "premise_confidence"    : r["premise_confidence"],
                "conclusion_confidence" : r["conclusion_confidence"]
            }
            for r in audit_results
        ]
    }

    ltn_score  = evaluate_generic_logic(parsed_data)
    violations = [r for r in audit_results if not r["satisfies"]]

    return ltn_score, violations