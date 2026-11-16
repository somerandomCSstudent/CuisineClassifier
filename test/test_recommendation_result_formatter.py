from __future__ import annotations

from cuisine_assistant.application import (
    ClassifyRecipeResult,
    LinkIngredientsResult,
    RecommendDishesResult,
)
from cuisine_assistant.cli.recommendation_result_formatter import (
    RecommendationResultFormatter,
)
from cuisine_assistant.domain.models import ClassificationResult
from cuisine_assistant.domain.recommendation_models import (
    CuisinePreference,
    Dish,
    DishRecommendation,
    RecommendationEvidence,
    RecommendationRequest,
    RecommendationResult,
)


def create_result() -> RecommendDishesResult:
    """Creates three ranked recommendations for formatter tests."""
    request = RecommendationRequest(
        linked_ingredients=(),
        preferred_cuisine=CuisinePreference(
            cuisine_id="hungarian",
            ontology_id="HungarianCuisine",
            source_text="Hungarian food",
            match_score=1.0,
        ),
    )
    evidence = RecommendationEvidence(
        matched_ingredient_ids=("chicken", "paprika", "onion"),
        missing_ingredient_ids=("sour_cream",),
        ingredient_match_score=0.75,
        cuisine_match_score=1.0,
        used_explicit_cuisine_preference=True,
    )
    dishes = (
        Dish("chicken_paprikash", "Chicken Paprikash", "ChickenPaprikash", "MainCourse", "hungarian"),
        Dish("goulash", "Goulash", "Goulash", "Soup", "hungarian"),
        Dish("thai_chicken_curry", "Thai Chicken Curry", "ThaiChickenCurry", "MainCourse", "thai"),
    )
    recommendations = tuple(
        DishRecommendation(dish, score, rank, evidence)
        for rank, (dish, score) in enumerate(
            zip(dishes, (0.8375, 0.70, 0.49)),
            start=1,
        )
    )
    classification = ClassifyRecipeResult(
        linking=LinkIngredientsResult(()),
        classification=ClassificationResult(),
    )
    return RecommendDishesResult(
        classification=classification,
        cuisine_preference=request.preferred_cuisine,
        recommendation=RecommendationResult(request, recommendations),
    )


def test_formats_best_and_top_three_recommendations() -> None:
    """Verifies the best dish is highlighted before the top-three ranking."""
    output = RecommendationResultFormatter().format(create_result())

    assert "Best recommendation:\nChicken Paprikash" in output
    assert "Recommendation score: 83.75%" in output
    assert "Top 3 dish recommendations:" in output
    assert "1. Chicken Paprikash: 83.75%" in output
    assert "2. Goulash: 70.00%" in output
    assert "3. Thai Chicken Curry: 49.00%" in output
    assert "Ingredient match: 75.00%" in output
    assert "Cuisine match: 100.00%" in output
    assert "Matched ingredients: chicken, paprika, onion" in output
    assert "Detected cuisine preference:" in output


def test_formats_missing_recommendation() -> None:
    """Verifies useful output when no knowledge-graph dish is available."""
    classification = ClassifyRecipeResult(
        linking=LinkIngredientsResult(()),
        classification=ClassificationResult(),
    )
    request = RecommendationRequest(linked_ingredients=())
    result = RecommendDishesResult(
        classification=classification,
        cuisine_preference=None,
        recommendation=RecommendationResult(request),
    )

    assert "No dish recommendation is available." in (
        RecommendationResultFormatter().format(result)
    )
