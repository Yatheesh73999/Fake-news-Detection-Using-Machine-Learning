"""Evidence scoring utilities.

Combines similarity, stance, and source credibility into a final evidence score.
"""

# Configurable scoring weights.
SIMILARITY_WEIGHT = 0.3
CREDIBILITY_WEIGHT = 0.3
STANCE_WEIGHT = 0.4

# Configurable value mappings used in weighted scoring.
CREDIBILITY_SCORES = {
    "HIGH": 1.0,
    "MEDIUM": 0.6,
    "LOW": 0.2,
}

STANCE_SCORES = {
    "SUPPORTS": 1.0,
    "NEUTRAL": 0.5,
    "REFUTES": 0.2,
}


def score_evidence(
    similarity: float, stance: str, credibility: str, direct_match: bool = False
) -> dict:
    """Compute final evidence score and classify evidence strength.

    Args:
        similarity: Semantic similarity score in [0, 1].
        stance: One of SUPPORTS, REFUTES, NEUTRAL.
        credibility: One of HIGH, MEDIUM, LOW.
    """
    stance_norm = (stance or "").upper()
    credibility_norm = (credibility or "").upper()

    # Clamp similarity into a safe [0, 1] range.
    similarity_clamped = float(max(0.0, min(1.0, similarity)))

    credibility_weight = CREDIBILITY_SCORES.get(credibility_norm, CREDIBILITY_SCORES["LOW"])
    stance_weight = STANCE_SCORES.get(stance_norm, STANCE_SCORES["NEUTRAL"])

    # Weighted evidence score (stance/credibility prioritized over similarity).
    score = (
        (similarity_clamped * SIMILARITY_WEIGHT)
        + (stance_weight * STANCE_WEIGHT)
        + (credibility_weight * CREDIBILITY_WEIGHT)
    )

    # Strength classification is rule-based, never similarity-only.
    if credibility_norm == "LOW":
        evidence_strength = "WEAK"
    elif direct_match and stance_norm == "SUPPORTS":
        evidence_strength = "STRONG"
    elif stance_norm == "SUPPORTS" and credibility_norm == "HIGH":
        evidence_strength = "STRONG"
    elif stance_norm == "REFUTES" and credibility_norm == "HIGH":
        evidence_strength = "COUNTER"
    elif stance_norm == "NEUTRAL":
        evidence_strength = "WEAK"
    else:
        evidence_strength = "WEAK"

    # Safeguard: clear direct matches should not be dragged down by moderate similarity.
    if direct_match and evidence_strength == "STRONG":
        score = max(score, 0.7)

    print(
        "DEBUG evidence scoring -> "
        f"stance={stance_norm}, credibility={credibility_norm}, "
        f"similarity={similarity_clamped:.4f}, final_strength={evidence_strength}, "
        f"direct_match={direct_match}"
    )

    return {
        "evidence_strength": evidence_strength,
        "score": round(float(score), 4),
    }


if __name__ == "__main__":
    # Unit-test style examples (simple assertions).
    case_1 = score_evidence(similarity=0.85, stance="SUPPORTS", credibility="HIGH")
    assert case_1["evidence_strength"] == "STRONG"

    case_2 = score_evidence(similarity=0.80, stance="REFUTES", credibility="HIGH")
    assert case_2["evidence_strength"] == "COUNTER"

    case_3 = score_evidence(similarity=0.90, stance="SUPPORTS", credibility="LOW")
    assert case_3["evidence_strength"] == "WEAK"

    case_4 = score_evidence(similarity=0.50, stance="NEUTRAL", credibility="MEDIUM")
    assert case_4["evidence_strength"] == "WEAK"

    print("All evidence_scorer tests passed.")
