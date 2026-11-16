from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.application.classify_recipe import (
    ClassifyRecipeResult,
    ClassifyRecipeUseCase,
)
from cuisine_assistant.domain.recommendation_models import (
    CuisinePreference,
    RecommendationRequest,
    RecommendationResult,
)
from cuisine_assistant.domain.recommendation_ports import (
    CuisinePreferenceExtractor,
    DishRecommender,
)


@dataclass(frozen=True)
class RecommendDishesResult:
    """Combines recipe classification, cuisine preference, and dish recommendations."""

    classification: ClassifyRecipeResult
    cuisine_preference: CuisinePreference | None
    recommendation: RecommendationResult

    @property
    def has_recommendations(self) -> bool:
        """Returns whether at least one dish recommendation is available."""
        return bool(self.recommendation.recommendations)


class RecommendDishesUseCase:
    """Classifies a recipe, extracts cuisine preference, and recommends dishes."""

    def __init__(
        self,
        classify_recipe: ClassifyRecipeUseCase,
        cuisine_preference_extractor: CuisinePreferenceExtractor,
        dish_recommender: DishRecommender,
    ) -> None:
        self._classify_recipe = classify_recipe
        self._cuisine_preference_extractor = cuisine_preference_extractor
        self._dish_recommender = dish_recommender

    def execute(self, user_text: str) -> RecommendDishesResult:
        classification = self._classify_recipe.execute(user_text)
        cuisine_preference = self._cuisine_preference_extractor.extract(
            user_text
        )
        request = RecommendationRequest(
            linked_ingredients=classification.linking.linked_ingredients,
            cuisine_predictions=classification.classification.predictions,
            preferred_cuisine=cuisine_preference,
        )
        recommendation = self._dish_recommender.recommend(request)
        return RecommendDishesResult(
            classification=classification,
            cuisine_preference=cuisine_preference,
            recommendation=recommendation,
        )
