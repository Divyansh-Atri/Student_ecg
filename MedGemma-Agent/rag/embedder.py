"""
Embedding model wrapper — all-MiniLM-L6-v2 via sentence-transformers.

Model is downloaded once to EMBEDDING_CACHE_DIR on first run,
then loaded from cache for all subsequent calls (fully offline).
"""

from __future__ import annotations
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from config.settings import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model (cached — loaded once per process)."""
    return SentenceTransformer(
        settings.EMBEDDING_MODEL,
        cache_folder=settings.EMBEDDING_CACHE_DIR,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns list of float vectors."""
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


@lru_cache(maxsize=1000)
def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
