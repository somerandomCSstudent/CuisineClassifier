from __future__ import annotations
from types import SimpleNamespace
import pytest
from cuisine_assistant.retrieval import (
    Bm25RecipeRetriever, EmbeddingRecipeRetriever, HybridRecipeRetriever, TextEmbedder,
)
from cuisine_assistant.retrieval_factory import RecipeRetrieverFactory

class Repository:
    """Provides an empty document collection for factory tests."""
    def list_all(self): return ()
    def get_by_id(self, recipe_id): return None
class Embedder(TextEmbedder):
    """Provides deterministic empty embeddings."""
    def embed_documents(self, texts): return ()
    def embed_query(self, text): return ()
def settings(strategy):
    """Creates settings accepted by the retriever factory."""
    return SimpleNamespace(
        recipe_retrieval_strategy=strategy, bm25_k1=1.5, bm25_b=.75,
        recipe_title_weight=.2, recipe_cuisine_weight=.15,
        recipe_ingredient_weight=.25, recipe_embedding_model_name="model",
        recipe_embedding_device=None, recipe_embedding_minimum_similarity=0.0,
        recipe_rrf_constant=60, recipe_lexical_weight=1.0,
        recipe_embedding_weight=1.0, recipe_candidate_multiplier=3,
    )
def test_builds_bm25_strategy():
    assert isinstance(RecipeRetrieverFactory(Embedder()).create(Repository(), settings("bm25")), Bm25RecipeRetriever)
def test_builds_embedding_strategy():
    assert isinstance(RecipeRetrieverFactory(Embedder()).create(Repository(), settings("embedding")), EmbeddingRecipeRetriever)
def test_builds_hybrid_strategy():
    assert isinstance(RecipeRetrieverFactory(Embedder()).create(Repository(), settings("hybrid")), HybridRecipeRetriever)
def test_rejects_unknown_strategy():
    with pytest.raises(ValueError): RecipeRetrieverFactory(Embedder()).create(Repository(), settings("unknown"))
