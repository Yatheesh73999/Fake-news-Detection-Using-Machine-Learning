"""Semantic similarity utilities for claim vs article comparison."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_cleaning import split_into_sentences

# Load the sentence embedding model once so it can be reused across calls.
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_similarity(claim: str, article_text: str) -> float:
    """Compute cosine similarity score between claim and article text."""
    # Convert input texts into dense vector embeddings.
    claim_embedding = model.encode([claim])
    article_embedding = model.encode([article_text])

    # Compute cosine similarity between the two embeddings.
    similarity = cosine_similarity(claim_embedding, article_embedding)[0][0]

    # Clamp to [0, 1] range for a clean normalized score.
    return float(max(0.0, min(1.0, similarity)))


def find_best_evidence_sentence(claim: str, article_text: str) -> tuple[str, float]:
    """Find the sentence most semantically similar to the claim."""
    # Break article into candidate evidence sentences.
    sentences = split_into_sentences(article_text)
    if not sentences:
        return "", 0.0

    # Embed claim once and all sentences in a single batch for efficiency.
    claim_embedding = model.encode([claim])
    sentence_embeddings = model.encode(sentences)

    # Compute similarity between claim and every sentence.
    scores = cosine_similarity(claim_embedding, sentence_embeddings)[0]

    # Select the sentence with the highest similarity score.
    best_index = int(scores.argmax())
    best_sentence = sentences[best_index]
    best_score = float(max(0.0, min(1.0, scores[best_index])))

    return best_sentence, best_score
