from __future__ import annotations

import pytest

from cuisine_assistant.domain.recommendation_models import (
    CuisinePreference,
    Dish,
    DishRecommendation,
    RecommendationEvidence,
    RecommendationRequest,
    RecommendationResult,
)


def create_dish() -> Dish:
    """Creates a dish fixture for recommendation domain tests."""
    return Dish(
        dish_id="chicken_paprikash",
        name="Chicken Paprikash",
        ontology_id="ChickenPaprikash",
        dish_category="MainCourse",
        cuisine_id="hungarian",
    )


def create_evidence() -> RecommendationEvidence:
    """Creates recommendation evidence for domain tests."""
    return RecommendationEvidence(
        matched_ingredient_ids=("chicken", "paprika"),
        missing_ingredient_ids=("sour_cream",),
        ingredient_match_score=0.67,
        cuisine_match_score=1.0,
        used_explicit_cuisine_preference=True,
    )


def test_cuisine_preference_is_optional_and_can_be_strict() -> None:
    preference = CuisinePreference(
        cuisine_id="hungarian",
        ontology_id="HungarianCuisine",
        source_text="Hungarian food",
        match_score=1.0,
        strict=True,
    )
    assert preference.strict is True


def test_evidence_rejects_invalid_score() -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        RecommendationEvidence((), (), 1.2, 0.0)


def test_result_returns_best_recommendation() -> None:
    result = RecommendationResult(
        request=RecommendationRequest(linked_ingredients=()),
        recommendations=(
            DishRecommendation(create_dish(), 0.91, 1, create_evidence()),
        ),
    )
    assert result.best_recommendation is not None
    assert result.best_recommendation.dish.dish_id == "chicken_paprikash"


def test_result_rejects_non_consecutive_ranks() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        RecommendationResult(
            request=RecommendationRequest(linked_ingredients=()),
            recommendations=(
                DishRecommendation(create_dish(), 0.91, 2, create_evidence()),
            ),
        )
