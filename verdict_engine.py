"""Final verdict generation from multiple evidence items."""

from typing import List


VALID_CREDIBILITY = {"HIGH", "MEDIUM"}


def _normalize_text(value: str) -> str:
    """Normalize text values for safe comparisons."""
    return (value or "").strip().upper()


def _is_valid_evidence(item: dict) -> bool:
    """Check if evidence should participate in final verdict aggregation."""
    credibility = _normalize_text(item.get("credibility", ""))
    stance = _normalize_text(item.get("stance", ""))
    best_sentence = (item.get("best_sentence") or "").strip().lower()

    if credibility not in VALID_CREDIBILITY:
        return False
    if stance not in {"SUPPORTS", "REFUTES"}:
        return False
    if not best_sentence or best_sentence == "content not available":
        return False
    if float(item.get("score", 0.0)) <= 0.0:
        return False
    return True


def _compute_confidence(scores: list[float], top_k: int = 3) -> float:
    """Compute confidence from strongest stance-aligned evidence scores."""
    if not scores:
        return 0.0

    strongest_scores = sorted(scores, reverse=True)[:top_k]
    return round(sum(strongest_scores) / len(strongest_scores), 4)


def get_verdict(evidences: List[dict]) -> dict:
    """Generate verdict using stance-based aggregation over valid evidence only."""
    if not evidences:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "reason": "No evidence items were provided.",
        }

    valid_evidences = [item for item in evidences if _is_valid_evidence(item)]
    ignored_count = len(evidences) - len(valid_evidences)

    if not valid_evidences:
        print("DEBUG verdict counts -> supports=0, refutes=0, ignored=", ignored_count)
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "reason": "No valid evidence after filtering low-credibility/irrelevant entries.",
        }

    supporting = [item for item in valid_evidences if _normalize_text(item.get("stance", "")) == "SUPPORTS"]
    refuting = [item for item in valid_evidences if _normalize_text(item.get("stance", "")) == "REFUTES"]

    supports_count = len(supporting)
    refutes_count = len(refuting)

    strong_support_count = sum(
        1 for item in supporting if _normalize_text(item.get("evidence_strength", "")) == "STRONG"
    )
    strong_refute_count = sum(
        1 for item in refuting if _normalize_text(item.get("evidence_strength", "")) == "COUNTER"
    )
    direct_support_count = sum(1 for item in supporting if bool(item.get("direct_match", False)))
    direct_refute_count = sum(1 for item in refuting if bool(item.get("direct_match", False)))

    print(
        "DEBUG verdict counts -> "
        f"supports={supports_count}, refutes={refutes_count}, ignored={ignored_count}"
    )

    if supports_count > refutes_count and (strong_support_count >= 1 or direct_support_count >= 1):
        confidence = _compute_confidence([float(item.get("score", 0.0)) for item in supporting])
        verdict = "LIKELY_TRUE"
        reason = (
            f"Supportive evidence outweighs refuting evidence "
            f"({supports_count} SUPPORTS vs {refutes_count} REFUTES), "
            f"with {strong_support_count} strong support item(s) and "
            f"{direct_support_count} direct factual match(es)."
        )
    elif refutes_count > supports_count and (strong_refute_count >= 1 or direct_refute_count >= 1):
        confidence = _compute_confidence([float(item.get("score", 0.0)) for item in refuting])
        verdict = "LIKELY_FALSE"
        reason = (
            f"Refuting evidence outweighs supportive evidence "
            f"({refutes_count} REFUTES vs {supports_count} SUPPORTS), "
            f"with {strong_refute_count} strong refute item(s) and "
            f"{direct_refute_count} direct factual match(es)."
        )
    else:
        confidence = _compute_confidence([float(item.get("score", 0.0)) for item in valid_evidences])
        verdict = "UNVERIFIED"
        reason = (
            "Evidence is inconclusive: support/refute balance or strong-evidence "
            f"requirement not met ({supports_count} SUPPORTS, {refutes_count} REFUTES)."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
    }
