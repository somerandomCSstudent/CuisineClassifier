from __future__ import annotations

from collections.abc import Iterable

import pytest

from cuisine_assistant.domain.models import (
    Cuisine,
    CuisinePrediction,
    ExtractedIngredient,
    Ingredient,
    IngredientCategory,
    LinkedIngredient,
    MatchMethod,
)
from cuisine_assistant.domain.ports import IngredientCatalog, KnowledgeGraphRepository
from cuisine_assistant.domain.recommendation_models import (
    CuisinePreference,
    Dish,
    RecommendationRequest,
)
from cuisine_assistant.domain.recommendation_ports import DishCatalog
from cuisine_assistant.recommendation import KnowledgeBasedDishRecommender


class InMemoryDishCatalog(DishCatalog):
    """Provides deterministic dishes for recommender tests."""

    def __init__(self) -> None:
        self._dishes = (
            Dish("chicken_paprikash", "Chicken Paprikash", "ChickenPaprikash", "MainCourse", "hungarian"),
            Dish("thai_chicken_curry", "Thai Chicken Curry", "ThaiChickenCurry", "MainCourse", "thai"),
        )

    def get_by_id(self, dish_id: str) -> Dish | None:
        return next((item for item in self._dishes if item.dish_id == dish_id), None)

    def get_by_ontology_id(self, ontology_id: str) -> Dish | None:
        return next((item for item in self._dishes if item.ontology_id == ontology_id), None)

    def list_all(self) -> tuple[Dish, ...]:
        return self._dishes


class InMemoryIngredientCatalog(IngredientCatalog):
    """Provides ingredient identifier mappings for recommender tests."""

    def __init__(self) -> None:
        self._items = (
            Ingredient("chicken", "chicken", IngredientCategory.MEAT, "Chicken"),
            Ingredient("paprika", "paprika", IngredientCategory.SPICE, "Paprika"),
            Ingredient("onion", "onion", IngredientCategory.VEGETABLE, "Onion"),
            Ingredient("sour_cream", "sour cream", IngredientCategory.DAIRY_PRODUCT, "SourCream"),
            Ingredient("coconut_milk", "coconut milk", IngredientCategory.DAIRY_ALTERNATIVE, "CoconutMilk"),
            Ingredient("lemongrass", "lemongrass", IngredientCategory.HERB, "Lemongrass"),
        )

    def get_by_id(self, ingredient_id: str) -> Ingredient | None:
        return next((item for item in self._items if item.ingredient_id == ingredient_id), None)

    def get_by_ontology_id(self, ontology_id: str) -> Ingredient | None:
        return next((item for item in self._items if item.ontology_id == ontology_id), None)

    def list_all(self) -> tuple[Ingredient, ...]:
        return self._items


class InMemoryKnowledgeGraph(KnowledgeGraphRepository):
    """Provides dish and ingredient graph relations for recommender tests."""

    _dish_ingredients = {
        "ChickenPaprikash": ("Chicken", "Paprika", "Onion", "SourCream"),
        "ThaiChickenCurry": ("Chicken", "CoconutMilk", "Lemongrass"),
    }

    def get_entity_reference(self, ontology_id: str):
        return None

    def get_entity_types(self, ontology_id: str) -> tuple[str, ...]:
        return ()

    def find_dish_ids_by_ingredient_ids(self, ingredient_ontology_ids: Iterable[str]) -> tuple[str, ...]:
        requested = set(ingredient_ontology_ids)
        return tuple(sorted(dish for dish, ingredients in self._dish_ingredients.items() if requested.intersection(ingredients)))

    def get_dish_ingredient_ids(self, dish_ontology_id: str) -> tuple[str, ...]:
        return self._dish_ingredients.get(dish_ontology_id, ())

    def get_dish_cuisine_id(self, dish_ontology_id: str) -> str | None:
        return {"ChickenPaprikash": "HungarianCuisine", "ThaiChickenCurry": "ThaiCuisine"}.get(dish_ontology_id)


def linked(ingredient_id: str, ontology_id: str, category: IngredientCategory) -> LinkedIngredient:
    """Creates one linked ingredient fixture."""
    text = ingredient_id.replace("_", " ")
    extracted = ExtractedIngredient(text, text, 0, len(text), ingredient_id, MatchMethod.EXACT, 1.0, (ingredient_id,))
    ingredient = Ingredient(ingredient_id, text, category, ontology_id)
    return LinkedIngredient(extracted, ingredient, None)


def create_recommender(**kwargs) -> KnowledgeBasedDishRecommender:
    """Creates the recommender with deterministic in-memory adapters."""
    return KnowledgeBasedDishRecommender(
        InMemoryDishCatalog(),
        InMemoryIngredientCatalog(),
        InMemoryKnowledgeGraph(),
        **kwargs,
    )


def hungarian_prediction(score: float = 0.8) -> CuisinePrediction:
    """Creates a Hungarian cuisine prediction fixture."""
    cuisine = Cuisine("hungarian", "Hungarian cuisine", "HungarianCuisine", "EuropeanCuisine")
    return CuisinePrediction(cuisine, score, 1)


def test_ranks_full_ingredient_match_first() -> None:
    request = RecommendationRequest(
        linked_ingredients=(
            linked("chicken", "Chicken", IngredientCategory.MEAT),
            linked("paprika", "Paprika", IngredientCategory.SPICE),
            linked("onion", "Onion", IngredientCategory.VEGETABLE),
            linked("sour_cream", "SourCream", IngredientCategory.DAIRY_PRODUCT),
        ),
        cuisine_predictions=(hungarian_prediction(),),
    )
    result = create_recommender().recommend(request)
    assert result.best_recommendation is not None
    assert result.best_recommendation.dish.dish_id == "chicken_paprikash"
    assert result.best_recommendation.score == pytest.approx(0.94)
    assert result.best_recommendation.evidence.ingredient_match_score == 1.0


def test_explicit_preference_overrides_inferred_cuisine_score() -> None:
    request = RecommendationRequest(
        linked_ingredients=(linked("chicken", "Chicken", IngredientCategory.MEAT),),
        cuisine_predictions=(hungarian_prediction(),),
        preferred_cuisine=CuisinePreference("thai", "ThaiCuisine", "Thai food", 1.0),
    )
    result = create_recommender().recommend(request)
    assert result.best_recommendation is not None
    assert result.best_recommendation.dish.cuisine_id == "thai"
    assert result.best_recommendation.evidence.used_explicit_cuisine_preference is True


def test_strict_preference_filters_other_cuisines() -> None:
    request = RecommendationRequest(
        linked_ingredients=(linked("chicken", "Chicken", IngredientCategory.MEAT),),
        preferred_cuisine=CuisinePreference("thai", "ThaiCuisine", "only Thai", 1.0, strict=True),
    )
    result = create_recommender().recommend(request)
    assert tuple(item.dish.cuisine_id for item in result.recommendations) == ("thai",)


def test_empty_ingredient_request_returns_no_recommendation() -> None:
    result = create_recommender().recommend(RecommendationRequest(linked_ingredients=()))
    assert result.recommendations == ()


def test_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="add up"):
        create_recommender(ingredient_weight=0.8, inferred_cuisine_weight=0.3)
