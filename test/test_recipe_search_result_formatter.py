from __future__ import annotations

from cuisine_assistant.application import (
    LinkIngredientsResult,
    SearchRecipesResult,
)
from cuisine_assistant.cli.recipe_search_result_formatter import (
    RecipeSearchResultFormatter,
)
from cuisine_assistant.domain.recommendation_models import CuisinePreference
from cuisine_assistant.retrieval import (
    RecipeDocument,
    RecipeSearchEvidence,
    RecipeSearchHit,
    RecipeSearchQuery,
    RecipeSearchResult,
)


def create_result() -> SearchRecipesResult:
    """Creates one ranked recipe search result for formatter tests."""
    query = RecipeSearchQuery(
        "Hungarian chicken with paprika",
        maximum_results=5,
        cuisine_id="hungarian",
        required_ingredient_ids=("chicken", "paprika"),
    )
    document = RecipeDocument(
        recipe_id="r001",
        dish_id="chicken_paprikash",
        title="Traditional Chicken Paprikash",
        raw_text="Cook chicken with paprika.",
        ingredient_text="chicken; paprika; onion; sour cream",
        instruction_text="Finish with sour cream.",
        cuisine_id="hungarian",
        source="manual",
        source_url="https://example.org/r001",
        license_name="CC BY 4.0",
    )
    evidence = RecipeSearchEvidence(
        matched_terms=("hungarian", "chicken", "paprika"),
        matched_ingredient_ids=("chicken", "paprika"),
        title_match_score=0.67,
        content_match_score=1.0,
        cuisine_match_score=1.0,
    )
    search = RecipeSearchResult(
        query=query,
        hits=(RecipeSearchHit(document, 0.93, 1, evidence),),
    )
    return SearchRecipesResult(
        query_text=query.text,
        linking=LinkIngredientsResult(()),
        cuisine_preference=CuisinePreference(
            "hungarian",
            "HungarianCuisine",
            "Hungarian",
            1.0,
        ),
        search=search,
    )


def test_formats_best_match_ranking_and_evidence() -> None:
    """Verifies the CLI exposes ranking, provenance, and matching evidence."""
    output = RecipeSearchResultFormatter().format(create_result())

    assert "Best recipe document match:\nTraditional Chicken Paprikash" in output
    assert "Search score: 93.00%" in output
    assert "1. Traditional Chicken Paprikash: 93.00%" in output
    assert "BM25 content match: 100.00%" in output
    assert "Matched ingredients: chicken, paprika" in output
    assert "Source: manual" in output
    assert "License: CC BY 4.0" in output
    assert "Detected cuisine preference:\nHungarian" in output


def test_formats_missing_result_with_active_filters() -> None:
    """Verifies an empty search explains the applied semantic constraints."""
    query = RecipeSearchQuery(
        "Thai coconut soup",
        cuisine_id="thai",
        required_ingredient_ids=("coconut_milk",),
    )
    result = SearchRecipesResult(
        query_text=query.text,
        linking=LinkIngredientsResult(()),
        cuisine_preference=None,
        search=RecipeSearchResult(query),
    )

    output = RecipeSearchResultFormatter().format(result)

    assert "No matching recipe document is available." in output
    assert "Required ingredients: coconut milk" in output
    assert "Cuisine filter: thai" in output
