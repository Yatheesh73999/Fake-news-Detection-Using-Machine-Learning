"""Decision engine for final claim verdict generation."""


def determine_verdict(similarity_scores: list[float]) -> dict:
    """Determine verdict and confidence from article similarity scores."""
    strong_evidence_count = sum(1 for score in similarity_scores if score > 0.75)
    weak_evidence_count = sum(1 for score in similarity_scores if 0.55 <= score <= 0.75)
    confidence = max(similarity_scores) if similarity_scores else 0.0

    if strong_evidence_count >= 2:
        verdict = "Likely True"
    elif strong_evidence_count == 1 and confidence >= 0.75:
        # Avoid returning Unverified when there is a clear high-confidence support signal.
        verdict = "Likely True"
    elif confidence >= 0.8 and (strong_evidence_count >= 1 or weak_evidence_count >= 3):
        # Extra safeguard: multiple strong/weak supportive signals with high confidence.
        verdict = "Likely True"
    elif strong_evidence_count == 1:
        verdict = "Unverified Claim"
    elif strong_evidence_count == 0 and weak_evidence_count >= 2:
        verdict = "Unverified Claim"
    else:
        verdict = "Likely False"

    return {
        "verdict": verdict,
        "confidence": confidence,
    }
