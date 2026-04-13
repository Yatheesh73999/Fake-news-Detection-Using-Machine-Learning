"""Source credibility and relevance filtering utilities."""

from urllib.parse import urlparse

from semantic_similarity import compute_similarity
from utils.text_cleaning import clean_text

# Easy-to-extend trusted source/domain list.
TRUSTED_DOMAINS = {
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "indianexpress.com",
    "thehindu.com",
    "hindustantimes.com",
    "ndtv.com",
    "apnews.com",
}

# Common non-news or low-signal categories to reject.
LIFESTYLE_CUES = {
    "lifestyle",
    "fashion",
    "beauty",
    "recipe",
    "travel",
    "celebrity",
    "wellness",
    "entertainment",
}

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


def _normalize_domain(source: str) -> str:
    """Normalize source text into a comparable domain-style value."""
    text = clean_text(source).lower()
    if not text:
        return ""

    # Handle plain domains, hostnames, and full URLs.
    if "://" not in text:
        text = f"https://{text}"
    parsed = urlparse(text)
    domain = parsed.netloc or parsed.path
    domain = domain.replace("www.", "").strip().strip("/")
    return domain


def _keyword_set(text: str) -> set[str]:
    """Tokenize text into meaningful keywords for overlap checks."""
    cleaned = clean_text(text).lower()
    tokens = [word.strip(".,!?;:()[]{}\"'") for word in cleaned.split()]
    return {token for token in tokens if token and token not in STOPWORDS and len(token) > 2}


def _keyword_overlap_score(title: str, claim: str) -> float:
    """Compute simple keyword overlap ratio between title and claim."""
    title_keywords = _keyword_set(title)
    claim_keywords = _keyword_set(claim)
    if not title_keywords or not claim_keywords:
        return 0.0

    overlap = title_keywords.intersection(claim_keywords)
    return len(overlap) / max(1, len(claim_keywords))


def _has_lifestyle_cue(text: str) -> bool:
    """Identify lifestyle/blog-like content based on cue terms."""
    lowered = clean_text(text).lower()
    return any(cue in lowered for cue in LIFESTYLE_CUES)


def evaluate_source(source: str, title: str, claim: str) -> dict:
    """Evaluate source credibility and topical relevance."""
    domain = _normalize_domain(source)

    if domain in TRUSTED_DOMAINS:
        credibility = "HIGH"
    elif domain:
        credibility = "MEDIUM"
    else:
        credibility = "LOW"

    # Unknown or malformed source strings are treated as low credibility.
    if not domain:
        credibility = "LOW"

    keyword_overlap = _keyword_overlap_score(title, claim)
    semantic_score = compute_similarity(claim, title)

    # Hard rejection rules first.
    if _has_lifestyle_cue(f"{source} {title}"):
        return {
            "credibility": "LOW" if credibility != "HIGH" else "MEDIUM",
            "is_relevant": False,
            "reason": "Rejected: lifestyle/blog-like content.",
        }

    if semantic_score <= 0.0:
        return {
            "credibility": credibility if credibility != "MEDIUM" else "LOW",
            "is_relevant": False,
            "reason": "Rejected: zero semantic similarity to the claim.",
        }

    # Combined relevance check using both lexical and semantic signals.
    is_relevant = keyword_overlap >= 0.2 or semantic_score >= 0.45
    if not is_relevant:
        return {
            "credibility": credibility,
            "is_relevant": False,
            "reason": "Rejected: unrelated topic based on low overlap and low similarity.",
        }

    return {
        "credibility": credibility,
        "is_relevant": True,
        "reason": (
            f"Relevant source. Keyword overlap={keyword_overlap:.2f}, "
            f"semantic similarity={semantic_score:.2f}."
        ),
    }
