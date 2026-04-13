"""Stance detection between a claim and an evidence sentence.

This module is intentionally modular so the decision logic can later be
replaced by an LLM/API-backed classifier.
"""

import re

from semantic_similarity import compute_similarity
from utils.text_cleaning import clean_text


CONTRADICTION_CUES = {"not", "never", "false", "denied"}
AGREEMENT_CUES = {"confirmed", "won", "announced"}
STOPWORDS = {
    "the",
    "is",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "at",
    "from",
    "that",
    "this",
    "it",
}


def _semantic_stance_score(claim: str, evidence: str) -> float:
    """Compute semantic alignment score between claim and evidence."""
    return compute_similarity(claim, evidence)


def _has_any_cue(text: str, cues: set[str]) -> bool:
    """Check if text contains any cue words/phrases."""
    normalized = clean_text(text).lower()
    return any(cue in normalized for cue in cues)


def _keyword_tokens(text: str) -> set[str]:
    """Tokenize text into meaningful terms for factual overlap checks."""
    normalized = clean_text(text).lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", normalized)
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


def has_direct_factual_match(claim: str, evidence: str) -> bool:
    """Detect explicit factual match between claim and evidence text."""
    claim_tokens = _keyword_tokens(claim)
    evidence_tokens = _keyword_tokens(evidence)
    if not claim_tokens or not evidence_tokens:
        return False

    overlap = claim_tokens.intersection(evidence_tokens)
    overlap_ratio = len(overlap) / len(claim_tokens)
    has_contradiction = _has_any_cue(evidence, CONTRADICTION_CUES)
    return overlap_ratio >= 0.7 and not has_contradiction


def _rule_override(evidence: str) -> tuple[str, float, str] | None:
    """Apply high-priority lexical overrides when strong cues are present."""
    has_contradiction = _has_any_cue(evidence, CONTRADICTION_CUES)
    has_agreement = _has_any_cue(evidence, AGREEMENT_CUES)

    if has_contradiction and not has_agreement:
        return "REFUTES", 0.82, "Contradiction cues detected in evidence."
    if has_agreement and not has_contradiction:
        return "SUPPORTS", 0.82, "Strong agreement cues detected in evidence."
    return None


def detect_stance(claim: str, evidence: str) -> dict:
    """Classify evidence stance toward a claim.

    Returns:
        {
            "stance": "SUPPORTS" | "REFUTES" | "NEUTRAL",
            "confidence": float,
            "reason": str
        }
    """
    claim_clean = clean_text(claim)
    evidence_clean = clean_text(evidence)

    if not claim_clean or not evidence_clean:
        return {
            "stance": "NEUTRAL",
            "confidence": 0.0,
            "reason": "Claim or evidence is empty after cleaning.",
        }

    # 1) Semantic baseline (primary signal)
    semantic_score = _semantic_stance_score(claim_clean, evidence_clean)

    # 2) Direct factual match safeguard for explicit claim statements.
    direct_match = has_direct_factual_match(claim_clean, evidence_clean)
    if direct_match:
        return {
            "stance": "SUPPORTS",
            "confidence": 0.9,
            "reason": "Direct factual match detected between claim and evidence.",
            "direct_match": True,
        }

    # 3) Rule-based override (secondary, high-priority cues)
    override = _rule_override(evidence_clean)
    if override:
        stance, confidence, reason = override
        return {"stance": stance, "confidence": confidence, "reason": reason, "direct_match": False}

    # 4) Default semantic decision boundaries
    if semantic_score >= 0.72:
        stance = "SUPPORTS"
        reason = "High semantic alignment between claim and evidence."
    elif semantic_score <= 0.38:
        stance = "REFUTES"
        reason = "Low semantic alignment suggests contradiction or mismatch."
    else:
        stance = "NEUTRAL"
        reason = "Semantic alignment is moderate and inconclusive."

    return {
        "stance": stance,
        "confidence": float(round(semantic_score, 4)),
        "reason": reason,
        "direct_match": False,
    }


if __name__ == "__main__":
    examples = [
        (
            "The company announced a new electric car model.",
            "The CEO confirmed and announced the launch during the event.",
        ),
        (
            "The player won the final match.",
            "Officials denied the result and said the match was not completed.",
        ),
        (
            "A new policy was introduced this week.",
            "The weather remained cloudy across the region.",
        ),
    ]

    for idx, (claim_text, evidence_text) in enumerate(examples, start=1):
        result = detect_stance(claim_text, evidence_text)
        print(f"Example {idx}: {result}")
