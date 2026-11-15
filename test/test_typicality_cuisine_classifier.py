from __future__ import annotations

from collections.abc import Iterable

import pytest

from cuisine_assistant.classification import TypicalityCuisineClassifier
from cuisine_assistant.domain.models import (
    Cuisine,
    ExtractedIngredient,
    Ingredient,
    IngredientCategory,
    LinkedIngredient,
    MatchMethod,
)
from cuisine_assistant.domain.ports import (
    CuisineCatalog,
    CuisineTypicalityRepository,
)


class InMemoryCuisineCatalog(CuisineCatalog):
    """Provides deterministic cuisine fixtures for classifier tests."""

    def __init__(self) -> None:
        self._cuisines = (
            Cuisine(
                cuisine_id="hungarian",
                name="Hungarian cuisine",
                ontology_id="HungarianCuisine",
                cuisine_group="EuropeanCuisine",
            ),
            Cuisine(
                cuisine_id="italian",
                name="Italian cuisine",
                ontology_id="ItalianCuisine",
                cuisine_group="EuropeanCuisine",
            ),
            Cuisine(
                cuisine_id="thai",
                name="Thai cuisine",
                ontology_id="ThaiCuisine",
                cuisine_group="AsianCuisine",
            ),
        )

    def get_by_id(self, cuisine_id: str) -> Cuisine | None:
        return next((item for item in self._cuisines if item.cuisine_id == cuisine_id), None)

    def get_by_ontology_id(self, ontology_id: str) -> Cuisine | None:
        return next((item for item in self._cuisines if item.ontology_id == ontology_id), None)

    def list_all(self) -> tuple[Cuisine, ...]:
        return self._cuisines


class InMemoryTypicalityRepository(CuisineTypicalityRepository):
    """Provides fixed cuisine-ingredient weights for classifier tests."""

    def __init__(self) -> None:
        self._weights = {
            ("hungarian", "paprika"): 0.95,
            ("hungarian", "sour_cream"): 0.80,
            ("hungarian", "chicken"): 0.30,
            ("italian", "tomato"): 0.85,
            ("italian", "basil"): 0.80,
            ("thai", "chicken"): 0.35,
        }

    def get_weight(self, cuisine_id: str, ingredient_id: str) -> float:
        return self._weights.get((cuisine_id, ingredient_id), 0.0)

    def get_typical_ingredient_ids(self, cuisine_id: str) -> tuple[str, ...]:
        return tuple(sorted(ingredient_id for (cid, ingredient_id), weight in self._weights.items() if cid == cuisine_id and weight > 0.0))


def linked(ingredient_id: str, ontology_id: str, category: IngredientCategory) -> LinkedIngredient:
    """Creates a catalog-linked ingredient fixture for classifier tests."""
    extracted = ExtractedIngredient(
        surface_form=ingredient_id.replace("_", " "),
        normalized_text=ingredient_id.replace("_", " "),
        start_char=0,
        end_char=len(ingredient_id.replace("_", " ")),
        matched_ingredient_id=ingredient_id,
        match_method=MatchMethod.EXACT,
        match_score=1.0,
        candidate_ids=(ingredient_id,),
    )
    ingredient = Ingredient(
        ingredient_id=ingredient_id,
        canonical_name=ingredient_id.replace("_", " "),
        category=category,
        ontology_id=ontology_id,
    )
    return LinkedIngredient(extracted, ingredient, None)


def unmatched(surface_form: str) -> LinkedIngredient:
    """Creates an unmatched ingredient fixture for classifier tests."""
    extracted = ExtractedIngredient(
        surface_form=surface_form,
        normalized_text=surface_form,
        start_char=0,
        end_char=len(surface_form),
        matched_ingredient_id=None,
        match_method=MatchMethod.UNMATCHED,
        match_score=0.0,
    )
    return LinkedIngredient(extracted, None, None)


def create_classifier(**kwargs) -> TypicalityCuisineClassifier:
    """Creates the classifier with deterministic in-memory adapters."""
    return TypicalityCuisineClassifier(
        InMemoryCuisineCatalog(),
        InMemoryTypicalityRepository(),
        **kwargs,
    )


def test_ranks_hungarian_cuisine_first() -> None:
    result = create_classifier().classify(
        (
            linked("chicken", "Chicken", IngredientCategory.MEAT),
            linked("paprika", "Paprika", IngredientCategory.SPICE),
            linked("sour_cream", "SourCream", IngredientCategory.DAIRY_PRODUCT),
        )
    )
    assert result.best_prediction is not None
    assert result.best_prediction.cuisine.cuisine_id == "hungarian"
    assert result.best_prediction.score == pytest.approx((0.30 + 0.95 + 0.80) / 3)
    assert tuple(item.ingredient_id for item in result.best_prediction.evidence) == (
        "paprika",
        "sour_cream",
        "chicken",
    )


def test_returns_only_requested_number_of_predictions() -> None:
    result = create_classifier(maximum_predictions=1).classify(
        (linked("chicken", "Chicken", IngredientCategory.MEAT),)
    )
    assert len(result.predictions) == 1


def test_deduplicates_repeated_ingredients() -> None:
    chicken = linked("chicken", "Chicken", IngredientCategory.MEAT)
    result = create_classifier().classify((chicken, chicken))
    assert result.best_prediction is not None
    assert result.best_prediction.score == pytest.approx(0.35)


def test_preserves_unmatched_ingredients() -> None:
    result = create_classifier().classify((unmatched("gochujang"),))
    assert result.predictions == ()
    assert result.unmatched_ingredients == ("gochujang",)


def test_applies_minimum_score_threshold() -> None:
    result = create_classifier(minimum_score=0.40).classify(
        (linked("chicken", "Chicken", IngredientCategory.MEAT),)
    )
    assert result.predictions == ()


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="maximum_predictions"):
        create_classifier(maximum_predictions=0)
    with pytest.raises(ValueError, match="minimum_score"):
        create_classifier(minimum_score=1.1)
