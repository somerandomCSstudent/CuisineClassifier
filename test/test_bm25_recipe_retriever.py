from __future__ import annotations

import pytest

from cuisine_assistant.retrieval import (
    Bm25RecipeRetriever,
    RecipeDocument,
    RecipeDocumentRepository,
    RecipeSearchQuery,
)


class InMemoryRecipeDocumentRepository(RecipeDocumentRepository):
    """Provides deterministic recipe documents for BM25 retriever tests."""

    def __init__(self) -> None:
        self._documents = (
            RecipeDocument(
                "r1",
                "chicken_paprikash",
                "Traditional Chicken Paprikash",
                "Cook chicken with onion paprika and sour cream.",
                "chicken; onion; paprika; sour cream",
                "Simmer and finish with sour cream.",
                "hungarian",
                "manual",
            ),
            RecipeDocument(
                "r2",
                "thai_chicken_curry",
                "Thai Chicken Curry",
                "Cook chicken with coconut milk and lemongrass.",
                "chicken; coconut milk; lemongrass",
                "Simmer the curry.",
                "thai",
                "manual",
            ),
            RecipeDocument(
                "r3",
                "caprese_salad",
                "Caprese Salad",
                "Combine tomato mozzarella and basil.",
                "tomato; mozzarella; basil",
                "Serve fresh.",
                "italian",
                "manual",
            ),
        )

    def get_by_id(self, recipe_id: str):
        return next(
            (item for item in self._documents if item.recipe_id == recipe_id),
            None,
        )

    def list_all(self):
        return self._documents


def create_retriever(**kwargs) -> Bm25RecipeRetriever:
    """Creates the BM25 retriever with a deterministic document collection."""
    return Bm25RecipeRetriever(
        InMemoryRecipeDocumentRepository(),
        **kwargs,
    )


def test_ranks_most_relevant_recipe_first() -> None:
    """Verifies rare matching terms and title matches influence ranking."""
    result = create_retriever().search(
        RecipeSearchQuery("chicken paprika sour cream", maximum_results=3)
    )

    assert result.best_hit is not None
    assert result.best_hit.document.recipe_id == "r1"
    assert result.best_hit.evidence.matched_terms == (
        "chicken",
        "paprika",
        "sour",
        "cream",
    )


def test_applies_cuisine_filter() -> None:
    """Verifies explicit cuisine constraints exclude other cuisines."""
    result = create_retriever().search(
        RecipeSearchQuery(
            "chicken",
            maximum_results=3,
            cuisine_id="thai",
        )
    )

    assert tuple(hit.document.recipe_id for hit in result.hits) == ("r2",)
    assert result.hits[0].evidence.cuisine_match_score == 1.0


def test_boosts_required_ingredient_coverage() -> None:
    """Verifies semantic ingredient evidence is included in final ranking."""
    result = create_retriever().search(
        RecipeSearchQuery(
            "chicken recipe",
            maximum_results=3,
            required_ingredient_ids=("chicken", "paprika"),
        )
    )

    assert result.best_hit is not None
    assert result.best_hit.document.recipe_id == "r1"
    assert result.best_hit.evidence.matched_ingredient_ids == (
        "chicken",
        "paprika",
    )


def test_returns_only_requested_number_of_hits() -> None:
    """Verifies the retriever respects the query's top-k limit."""
    result = create_retriever().search(
        RecipeSearchQuery("chicken", maximum_results=1)
    )

    assert len(result.hits) == 1
    assert result.hits[0].rank == 1


def test_returns_empty_result_for_unknown_terms() -> None:
    """Verifies unrelated queries do not produce zero-score documents."""
    result = create_retriever().search(
        RecipeSearchQuery("sushi seaweed wasabi")
    )

    assert result.hits == ()


def test_tie_breaking_is_deterministic_by_recipe_id() -> None:
    """Verifies equal-score documents are ordered by stable recipe ID."""
    first = RecipeDocument(
        "r2", "dish_two", "Same", "shared token", "", "", None, "manual"
    )
    second = RecipeDocument(
        "r1", "dish_one", "Same", "shared token", "", "", None, "manual"
    )

    class EqualScoreRepository(RecipeDocumentRepository):
        """Provides equal-score documents in reverse identifier order."""

        def get_by_id(self, recipe_id: str):
            return next(
                (item for item in (first, second) if item.recipe_id == recipe_id),
                None,
            )

        def list_all(self):
            return (first, second)

    result = Bm25RecipeRetriever(EqualScoreRepository()).search(
        RecipeSearchQuery("shared", maximum_results=2)
    )

    assert tuple(hit.document.recipe_id for hit in result.hits) == ("r1", "r2")


def test_rejects_invalid_bm25_configuration() -> None:
    """Verifies BM25 and boost parameters stay within valid ranges."""
    with pytest.raises(ValueError, match="k1"):
        create_retriever(k1=0)
    with pytest.raises(ValueError, match="b"):
        create_retriever(b=1.5)
    with pytest.raises(ValueError, match="title_weight"):
        create_retriever(title_weight=-0.1)
