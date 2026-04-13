"""End-to-end claim verification pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from evidence_scorer import score_evidence
from news_fetcher import fetch_news_articles
from semantic_similarity import find_best_evidence_sentence
from source_filter import evaluate_source
from stance_detection import detect_stance, has_direct_factual_match
from verdict_engine import get_verdict


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline behavior."""

    max_articles: int = 5
    debug_print: bool = True
    min_evidences_keep: int = 3


def _debug_print(enabled: bool, message: str) -> None:
    """Print intermediate debug output when enabled."""
    if enabled:
        print(message)


def retrieve_articles(claim: str, config: PipelineConfig) -> list[dict[str, Any]]:
    """Step 1-2: retrieve news articles for a claim."""
    LOGGER.info("Retrieving articles for claim.")
    articles = fetch_news_articles(claim)
    trimmed_articles = articles[: config.max_articles]
    LOGGER.info("Retrieved %s article(s).", len(trimmed_articles))
    _debug_print(config.debug_print, f"[DEBUG] Retrieved articles: {len(trimmed_articles)}")
    return trimmed_articles


def annotate_sources(
    claim: str, articles: list[dict[str, Any]], config: PipelineConfig
) -> list[dict[str, Any]]:
    """Step 3: evaluate source metadata without early hard filtering."""
    LOGGER.info("Evaluating source credibility and relevance.")
    annotated_articles: list[dict[str, Any]] = []

    for article in articles:
        source_value = article.get("url") or article.get("source name", "")
        title = article.get("title", "")
        source_eval = evaluate_source(source=source_value, title=title, claim=claim)
        article_with_source = dict(article)
        article_with_source["source_eval"] = source_eval

        _debug_print(
            config.debug_print,
            (
                "[DEBUG] Source eval -> "
                f"title='{title}', credibility={source_eval['credibility']}, "
                f"relevant={source_eval['is_relevant']}"
            ),
        )

        annotated_articles.append(article_with_source)

    LOGGER.info("Annotated articles after source evaluation: %s", len(annotated_articles))
    return annotated_articles


def extract_and_detect_stance(
    claim: str, articles: list[dict[str, Any]], config: PipelineConfig
) -> list[dict[str, Any]]:
    """Step 4-5: extract evidence sentence and detect stance."""
    LOGGER.info("Extracting evidence sentences and detecting stance.")
    evidences: list[dict[str, Any]] = []

    for article in articles:
        content = article.get("content", "")
        best_sentence, similarity = find_best_evidence_sentence(claim, content)
        if not best_sentence:
            best_sentence = "Content not available"
        direct_match = has_direct_factual_match(claim, best_sentence)

        credibility = article.get("source_eval", {}).get("credibility", "LOW")

        stance_result = detect_stance(claim=claim, evidence=best_sentence)

        evidence = {
            "title": article.get("title", "N/A"),
            "source": article.get("source name", "N/A"),
            "url": article.get("url", "N/A"),
            "credibility": credibility,
            "similarity": round(float(similarity), 4),
            "best_sentence": best_sentence,
            "stance": stance_result.get("stance", "NEUTRAL"),
            "stance_reason": stance_result.get("reason", ""),
            "direct_match": bool(stance_result.get("direct_match", direct_match)),
        }
        evidences.append(evidence)

        _debug_print(
            config.debug_print,
            (
                "[DEBUG]\n"
                f"Source: {evidence['source']}\n"
                f"Similarity: {evidence['similarity']:.4f}\n"
                f"Stance: {evidence['stance']}\n"
                f"Credibility: {evidence['credibility']}\n"
                "Strength: PENDING"
            ),
        )

    LOGGER.info("Extracted and detected stance for %s evidence item(s).", len(evidences))
    return evidences


def filter_evidences(evidences: list[dict[str, Any]], config: PipelineConfig) -> list[dict[str, Any]]:
    """Step 6: safe filtering that prevents evidence collapse."""
    print("RAW:", len(evidences))

    filtered_evidences: list[dict[str, Any]] = []

    for ev in evidences:
        sentence = (ev.get("best_sentence") or "").strip().lower()
        similarity = float(ev.get("similarity", 0.0))

        # Filtering removes only empty/unavailable sentences or zero-similarity entries.
        if not sentence or sentence == "content not available":
            print("Filtered out: evidence sentence is empty")
            continue
        if similarity == 0.0:
            print("Filtered out: similarity == 0.0")
            continue

        # Always keep if reasonable similarity.
        if similarity >= 0.2:
            filtered_evidences.append(ev)
            continue

        # Keep very low-but-nonzero similarity too, to avoid over-filtering.
        filtered_evidences.append(ev)

    # Fallback: if everything removed, keep top 3 original.
    if len(filtered_evidences) == 0:
        print("WARNING: Filter removed all evidence, applying fallback")
        filtered_evidences = evidences[:3]

    print("FILTERED:", len(filtered_evidences))
    return filtered_evidences


def score_evidences(evidences: list[dict[str, Any]], config: PipelineConfig) -> list[dict[str, Any]]:
    """Step 7: score evidence after filtering."""
    scored: list[dict[str, Any]] = []
    for evidence in evidences:
        score_result = score_evidence(
            similarity=float(evidence.get("similarity", 0.0)),
            stance=str(evidence.get("stance", "NEUTRAL")),
            credibility=str(evidence.get("credibility", "LOW")),
            direct_match=bool(evidence.get("direct_match", False)),
        )
        enriched = dict(evidence)
        enriched["evidence_strength"] = score_result.get("evidence_strength", "WEAK")
        enriched["score"] = score_result.get("score", 0.0)
        scored.append(enriched)

        _debug_print(
            config.debug_print,
            (
                "[DEBUG]\n"
                f"Source: {enriched['source']}\n"
                f"Similarity: {enriched['similarity']:.4f}\n"
                f"Stance: {enriched['stance']}\n"
                f"Credibility: {enriched['credibility']}\n"
                f"Strength: {enriched['evidence_strength']}"
            ),
        )

    total_supports = sum(1 for evidence in scored if evidence.get("stance") == "SUPPORTS")
    total_refutes = sum(1 for evidence in scored if evidence.get("stance") == "REFUTES")
    ignored = len(evidences) - len(scored)
    _debug_print(config.debug_print, f"Total Supports: {total_supports}")
    _debug_print(config.debug_print, f"Total Refutes: {total_refutes}")
    _debug_print(config.debug_print, f"Ignored: {ignored}")
    return scored


def build_fallback_evidences(extracted_evidences: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """Create fallback evidence set when filtering removes everything."""
    fallback_candidates = sorted(
        extracted_evidences,
        key=lambda item: float(item.get("similarity", 0.0)),
        reverse=True,
    )[:top_k]

    fallback_evidences: list[dict[str, Any]] = []
    for item in fallback_candidates:
        enriched = dict(item)
        enriched["evidence_strength"] = "LOW CONFIDENCE"
        # Keep a bounded low-confidence score for robustness fallback mode.
        enriched["score"] = 0.35
        fallback_evidences.append(enriched)

    return fallback_evidences


def generate_final_verdict(evidences: list[dict[str, Any]], config: PipelineConfig) -> dict[str, Any]:
    """Step 7: aggregate all evidences into a final verdict."""
    LOGGER.info("Generating final verdict.")
    verdict = get_verdict(evidences)
    _debug_print(config.debug_print, f"[DEBUG] Final verdict payload: {verdict}")
    return verdict


def run_pipeline(claim: str, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Run the full claim verification pipeline in order."""
    active_config = config or PipelineConfig()
    LOGGER.info("Pipeline started.")

    articles = retrieve_articles(claim, active_config)
    annotated_articles = annotate_sources(claim, articles, active_config)
    extracted_evidences = extract_and_detect_stance(claim, annotated_articles, active_config)
    filtered_evidences = filter_evidences(extracted_evidences, active_config)
    if not filtered_evidences:
        print("Fallback triggered: insufficient filtered evidence")
        evidences = build_fallback_evidences(extracted_evidences, top_k=3)
        verdict = {
            "verdict": "UNVERIFIED",
            "confidence": 0.4,
            "reason": "Fallback mode: insufficient filtered evidence, using top original evidences.",
        }
    else:
        evidences = score_evidences(filtered_evidences, active_config)
        verdict = generate_final_verdict(evidences, active_config)

    result = {
        "claim": claim,
        "articles_retrieved": len(articles),
        "articles_used": len(annotated_articles),
        "evidences": evidences,
        "verdict": verdict,
    }
    LOGGER.info("Pipeline completed.")
    _debug_print(active_config.debug_print, f"[DEBUG] Pipeline result keys: {list(result.keys())}")
    return result


def main() -> None:
    """CLI entrypoint to run the full pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    claim = input("Enter claim: ").strip()
    if not claim:
        print("Please enter a valid claim.")
        return

    output = run_pipeline(claim, config=PipelineConfig(debug_print=True))
    print("\n=== FINAL OUTPUT ===")
    print(f"Claim: {output['claim']}")
    print(f"Articles Retrieved: {output['articles_retrieved']}")
    print(f"Articles Used: {output['articles_used']}")
    print(f"Verdict: {output['verdict']['verdict']}")
    print(f"Confidence: {output['verdict']['confidence']:.2f}")
    print(f"Reason: {output['verdict']['reason']}")


if __name__ == "__main__":
    main()
