"""Utilities for fetching news articles from external sources."""

from typing import Any

import requests
from newspaper import Article

from config import NEWS_API_KEY
from utils.text_cleaning import clean_text


NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_news_articles(claim: str) -> list[dict[str, Any]]:
    """Fetch top 5 English news articles related to the given claim."""
    params = {
        "q": claim,
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch news articles: {exc}") from exc

    if payload.get("status") != "ok":
        message = payload.get("message", "Unknown API error.")
        raise RuntimeError(f"NewsAPI returned an error: {message}")

    articles = payload.get("articles", [])
    if not articles:
        raise ValueError("No news articles found for the provided claim.")

    formatted_articles: list[dict[str, Any]] = []
    for article in articles:
        article_url = article.get("url") or ""
        full_article_text = extract_full_article(article_url) if article_url else ""
        cleaned_content = clean_text(full_article_text) if full_article_text else ""
        formatted_articles.append(
            {
                "title": article.get("title"),
                "source name": article.get("source", {}).get("name"),
                "published date": article.get("publishedAt"),
                "url": article_url,
                "description": article.get("description"),
                "content": cleaned_content or "Content not available",
            }
        )

    return formatted_articles


def extract_full_article(url: str) -> str:
    """Download and extract cleaned full text from a news article URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        return ""
