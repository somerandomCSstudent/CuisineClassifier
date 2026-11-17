from __future__ import annotations

import pytest

from cuisine_assistant.retrieval.hybrid_retriever import HybridRecipeRetriever
from cuisine_assistant.retrieval.models import (
    RecipeDocument,
    RecipeSearchEvidence,
    RecipeSearchHit,
    RecipeSearchQuery,
    RecipeSearchResult,
)
from cuisine_assistant.retrieval.ports import RecipeRetriever


def document(recipe_id: str) -> RecipeDocument:
    """Creates one recipe document fixture."""
    return RecipeDocument(
        recipe_id,
        f"dish_{recipe_id}",
        recipe_id.upper(),
        f"Text for {recipe_id}",
        "chicken; paprika",
        "Cook.",
        "hungarian",
        "test",
    )


def evidence(term: str, content_score: float) -> RecipeSearchEvidence:
    """Creates one retriever-specific evidence fixture."""
    return RecipeSearchEvidence(
        matched_terms=(term,) if term else (),
        matched_ingredient_ids=("chicken",),
        title_match_score=0.5,
        content_match_score=content_score,
        cuisine_match_score=1.0,
    )


class StubRetriever(RecipeRetriever):
    """Returns a configured ranking and records the requested candidate limit."""

    def __init__(self, recipe_ids: tuple[str, ...], term: str) -> None:
        self._recipe_ids = recipe_ids
        self._term = term
        self.received_limit: int | None = None

    def search(self, query: RecipeSearchQuery) -> RecipeSearchResult:
        self.received_limit = query.maximum_results
        hits = tuple(
            RecipeSearchHit(
                document(recipe_id),
                1.0 - index * 0.1,
                index + 1,
                evidence(self._term, 1.0 - index * 0.1),
            )
            for index, recipe_id in enumerate(self._recipe_ids)
        )
        return RecipeSearchResult(query=query, hits=hits)


def test_consensus_document_ranks_first() -> None:
    """Verifies a document present in both rankings receives both RRF scores."""
    lexical = StubRetriever(("r1", "r2"), "lexical")
    embedding = StubRetriever(("r2", "r3"), "semantic")
    retriever = HybridRecipeRetriever(lexical, embedding)

    result = retriever.search(
        RecipeSearchQuery("chicken paprika", maximum_results=3)
    )

    assert result.best_hit is not None
    assert result.best_hit.document.recipe_id == "r2"
    assert result.best_hit.score == pytest.approx(1.0)
    assert result.best_hit.evidence.matched_terms == (
        "lexical",
        "semantic",
    )


def test_weighted_fusion_can_prefer_embedding_ranking() -> None:
    """Verifies strategy weights control the fused ranking."""
    retriever = HybridRecipeRetriever(
        StubRetriever(("r1", "r2"), "lexical"),
        StubRetriever(("r3", "r2"), "semantic"),
        lexical_weight=0.0,
        embedding_weight=1.0,
    )

    result = retriever.search(
        RecipeSearchQuery("conceptual query", maximum_results=3)
    )

    assert result.best_hit is not None
    assert result.best_hit.document.recipe_id == "r3"


def test_requests_expanded_candidate_pool_and_returns_top_k() -> None:
    """Verifies fusion retrieves more candidates before truncating the result."""
    lexical = StubRetriever(("r1", "r2", "r3"), "lexical")
    embedding = StubRetriever(("r3", "r4", "r5"), "semantic")
    retriever = HybridRecipeRetriever(
        lexical,
        embedding,
        candidate_multiplier=4,
    )

    result = retriever.search(
        RecipeSearchQuery("query", maximum_results=2)
    )

    assert lexical.received_limit == 8
    assert embedding.received_limit == 8
    assert len(result.hits) == 2
    assert tuple(hit.rank for hit in result.hits) == (1, 2)


def test_returns_empty_result_when_both_retrievers_are_empty() -> None:
    """Verifies fusion preserves an empty retrieval outcome."""
    result = HybridRecipeRetriever(
        StubRetriever((), "lexical"),
        StubRetriever((), "semantic"),
    ).search(RecipeSearchQuery("unknown"))

    assert result.hits == ()


def test_tie_breaking_is_deterministic() -> None:
    """Verifies equal RRF scores are ordered by stable recipe ID."""
    result = HybridRecipeRetriever(
        StubRetriever(("r2",), "lexical"),
        StubRetriever(("r1",), "semantic"),
    ).search(RecipeSearchQuery("query", maximum_results=2))

    assert tuple(hit.document.recipe_id for hit in result.hits) == (
        "r1",
        "r2",
    )


def test_rejects_invalid_fusion_configuration() -> None:
    """Verifies RRF parameters and weights remain valid."""
    lexical = StubRetriever((), "lexical")
    embedding = StubRetriever((), "semantic")
    with pytest.raises(ValueError, match="rrf_constant"):
        HybridRecipeRetriever(lexical, embedding, rrf_constant=0)
    with pytest.raises(ValueError, match="lexical_weight"):
        HybridRecipeRetriever(lexical, embedding, lexical_weight=-1)
    with pytest.raises(ValueError, match="at least one"):
        HybridRecipeRetriever(
            lexical,
            embedding,
            lexical_weight=0,
            embedding_weight=0,
        )
    with pytest.raises(ValueError, match="candidate_multiplier"):
        HybridRecipeRetriever(
            lexical,
            embedding,
            candidate_multiplier=0,
        )
