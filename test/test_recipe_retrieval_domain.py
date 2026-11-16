from __future__ import annotations

import pytest

from cuisine_assistant.retrieval import (
    RecipeDocument,
    RecipeSearchEvidence,
    RecipeSearchHit,
    RecipeSearchQuery,
    RecipeSearchResult,
)


def create_document(identifier: str = "r1") -> RecipeDocument:
    """Creates one searchable recipe fixture."""
    return RecipeDocument(
        recipe_id=identifier,
        dish_id="chicken_paprikash",
        title="Traditional Chicken Paprikash",
        raw_text="Cook chicken with onion and paprika.",
        ingredient_text="chicken; onion; paprika; sour cream",
        instruction_text="Finish with sour cream.",
        cuisine_id="hungarian",
        source="manual",
    )


def create_evidence() -> RecipeSearchEvidence:
    """Creates explainable lexical retrieval evidence."""
    return RecipeSearchEvidence(
        matched_terms=("chicken", "paprika"),
        matched_ingredient_ids=("chicken", "paprika"),
        title_match_score=0.5,
        content_match_score=0.75,
        cuisine_match_score=1.0,
    )


def test_document_builds_searchable_text_from_recipe_fields() -> None:
    """Verifies all relevant textual fields are exposed to retrieval."""
    document = create_document()

    assert "Traditional Chicken Paprikash" in document.searchable_text
    assert "sour cream" in document.searchable_text
    assert "hungarian" in document.searchable_text


def test_query_supports_optional_semantic_filters() -> None:
    """Verifies cuisine and ingredient constraints can guide retrieval."""
    query = RecipeSearchQuery(
        text="Hungarian chicken with paprika",
        maximum_results=3,
        cuisine_id="hungarian",
        required_ingredient_ids=("chicken", "paprika"),
    )

    assert query.maximum_results == 3
    assert query.cuisine_id == "hungarian"


def test_result_returns_best_ranked_hit() -> None:
    """Verifies ranked search results expose the highest-scoring document."""
    query = RecipeSearchQuery("chicken paprika", maximum_results=3)
    result = RecipeSearchResult(
        query=query,
        hits=(
            RecipeSearchHit(create_document(), 0.9, 1, create_evidence()),
        ),
    )

    assert result.best_hit is not None
    assert result.best_hit.document.recipe_id == "r1"


def test_result_rejects_non_consecutive_ranks() -> None:
    """Verifies retrieval ranks are deterministic and start at one."""
    query = RecipeSearchQuery("chicken")

    with pytest.raises(ValueError, match="consecutive"):
        RecipeSearchResult(
            query=query,
            hits=(
                RecipeSearchHit(create_document(), 0.9, 2, create_evidence()),
            ),
        )


def test_result_rejects_unsorted_scores() -> None:
    """Verifies retrieval results are sorted before reaching the application."""
    query = RecipeSearchQuery("chicken", maximum_results=2)

    with pytest.raises(ValueError, match="descending"):
        RecipeSearchResult(
            query=query,
            hits=(
                RecipeSearchHit(create_document("r1"), 0.4, 1, create_evidence()),
                RecipeSearchHit(create_document("r2"), 0.8, 2, create_evidence()),
            ),
        )


def test_rejects_invalid_query_limit_and_scores() -> None:
    """Verifies invalid retrieval configuration fails at the domain boundary."""
    with pytest.raises(ValueError, match="maximum_results"):
        RecipeSearchQuery("chicken", maximum_results=0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        RecipeSearchEvidence((), (), 1.1, 0.0, 0.0)
