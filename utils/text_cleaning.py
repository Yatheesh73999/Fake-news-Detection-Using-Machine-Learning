"""Text normalization helpers."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace without altering punctuation."""
    return " ".join(text.replace("\n", " ").split())


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation rules."""
    # First normalize spacing so sentence parsing is consistent.
    normalized_text = clean_text(text)

    # Split at sentence-ending punctuation: period, question mark, exclamation.
    sentence_parts = re.split(r"[.!?]+", normalized_text)

    # Remove empty entries and trim whitespace around each sentence.
    return [sentence.strip() for sentence in sentence_parts if sentence.strip()]
