from __future__ import annotations
from enum import StrEnum

class RecipeRetrievalStrategy(StrEnum):
    """Defines supported recipe-document retrieval strategies."""
    BM25 = "bm25"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"
