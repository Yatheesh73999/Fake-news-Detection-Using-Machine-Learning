"""Entry point for the fake news detection backend project."""

from decision_engine import determine_verdict
from news_fetcher import fetch_news_articles
from semantic_similarity import find_best_evidence_sentence


def _get_evidence_label(similarity_score: float) -> str:
    """Classify evidence strength based on similarity thresholds."""
    if similarity_score > 0.55:
        return "Strong Evidence"
    if 0.35 <= similarity_score <= 0.55:
        return "Weak Evidence"
    return "Not Relevant"


def _print_header(claim: str) -> None:
    """Print the main claim verification report header."""
    print("=" * 49)
    print("CLAIM VERIFICATION RESULT")
    print("=" * 49)
    print()
    print(f"Claim: {claim}")
    print()


def _analyze_articles(claim: str, articles: list[dict]) -> tuple[list[dict], list[float]]:
    """Attach evidence details to each article and collect similarity scores."""
    analyzed_articles: list[dict] = []
    similarity_scores: list[float] = []

    for article in articles:
        article_content = article.get("content", "")
        evidence_sentence, similarity_score = (
            find_best_evidence_sentence(claim, article_content)
            if article_content and article_content != "Content not available"
            else ("Content not available", 0.0)
        )
        evidence_label = _get_evidence_label(similarity_score)

        enriched_article = dict(article)
        enriched_article["similarity_score"] = similarity_score
        enriched_article["evidence_label"] = evidence_label
        enriched_article["evidence_sentence"] = evidence_sentence
        analyzed_articles.append(enriched_article)
        similarity_scores.append(similarity_score)

    return analyzed_articles, similarity_scores


def _print_verdict(similarity_scores: list[float]) -> None:
    """Print final verdict and confidence score."""
    decision = determine_verdict(similarity_scores)
    print(f"Final Verdict: {decision['verdict']}")
    print(f"Confidence Score: {decision['confidence']:.2f}\n")


def _print_evidence_section_header() -> None:
    """Print the evidence articles section heading."""
    print("-" * 49)
    print("Evidence Articles")
    print("-" * 49)
    print()


def _print_articles(articles: list[dict]) -> None:
    """Print fetched articles in a professional format."""
    for index, article in enumerate(articles, start=1):
        print(f"{index}. Title: {article.get('title', 'N/A')}")
        print(f"   Source: {article.get('source name', 'N/A')}")
        print(f"   Date: {article.get('published date', 'N/A')}")
        print(f"   Similarity Score: {article.get('similarity_score', 0.0):.2f}")
        print(f"   Evidence Strength: {article.get('evidence_label', 'Not Relevant')}")
        print("   Evidence Sentence:")
        print(f"   {article.get('evidence_sentence', 'Content not available')}")
        print(f"   Link: {article.get('url', 'N/A')}")
        print()
    print("-" * 49)


def main() -> None:
    """Run the application."""
    claim = input("Enter a claim to verify: ").strip()
    if not claim:
        print("Please enter a valid claim.")
        return

    try:
        articles = fetch_news_articles(claim)
    except ValueError:
        _print_header(claim)
        print("Final Verdict: Unverified Claim")
        print("Confidence Score: 0.00\n")
        _print_evidence_section_header()
        print("Insufficient evidence from trusted sources.")
        print("-" * 49)
        return
    except RuntimeError as exc:
        _print_header(claim)
        print("Final Verdict: Unverified Claim")
        print("Confidence Score: 0.00\n")
        _print_evidence_section_header()
        print(f"Error: {exc}")
        print("-" * 49)
        return

    if not articles:
        _print_header(claim)
        print("Final Verdict: Unverified Claim")
        print("Confidence Score: 0.00\n")
        _print_evidence_section_header()
        print("Insufficient evidence from trusted sources.")
        print("-" * 49)
        return

    _print_header(claim)
    analyzed_articles, similarity_scores = _analyze_articles(claim, articles)
    _print_verdict(similarity_scores)
    _print_evidence_section_header()
    _print_articles(analyzed_articles)


if __name__ == "__main__":
    main()
