from __future__ import annotations
from cuisine_assistant.retrieval import (
    Bm25RecipeRetriever, EmbeddingRecipeRetriever, HybridRecipeRetriever,
    RecipeDocumentRepository, RecipeRetriever, SentenceTransformerTextEmbedder,
    TextEmbedder,
)
from cuisine_assistant.retrieval_strategy import RecipeRetrievalStrategy

class RecipeRetrieverFactory:
    """Creates the configured recipe retriever without changing use cases."""
    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self._embedder = embedder
    def create(self, repository: RecipeDocumentRepository, settings) -> RecipeRetriever:
        """Builds BM25, embedding, or hybrid retrieval from settings."""
        strategy = RecipeRetrievalStrategy(settings.recipe_retrieval_strategy)
        lexical = Bm25RecipeRetriever(
            repository, k1=settings.bm25_k1, b=settings.bm25_b,
            title_weight=settings.recipe_title_weight,
            cuisine_weight=settings.recipe_cuisine_weight,
            ingredient_weight=settings.recipe_ingredient_weight,
        )
        if strategy is RecipeRetrievalStrategy.BM25:
            return lexical
        embedder = self._embedder or SentenceTransformerTextEmbedder(
            settings.recipe_embedding_model_name,
            device=settings.recipe_embedding_device,
        )
        semantic = EmbeddingRecipeRetriever(
            repository, embedder,
            minimum_similarity=settings.recipe_embedding_minimum_similarity,
            cuisine_weight=settings.recipe_cuisine_weight,
            ingredient_weight=settings.recipe_ingredient_weight,
        )
        if strategy is RecipeRetrievalStrategy.EMBEDDING:
            return semantic
        return HybridRecipeRetriever(
            lexical, semantic,
            rrf_constant=settings.recipe_rrf_constant,
            lexical_weight=settings.recipe_lexical_weight,
            embedding_weight=settings.recipe_embedding_weight,
            candidate_multiplier=settings.recipe_candidate_multiplier,
        )
