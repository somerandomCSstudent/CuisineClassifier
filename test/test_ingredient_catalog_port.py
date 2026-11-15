from __future__ import annotations

from cuisine_assistant.domain.models import Ingredient, IngredientCategory
from cuisine_assistant.domain.ports import IngredientCatalog


class InMemoryIngredientCatalog(IngredientCatalog):
    def __init__(self, ingredients: tuple[Ingredient, ...]) -> None:
        self._ingredients = ingredients
        self._by_id = {
            ingredient.ingredient_id: ingredient for ingredient in ingredients
        }
        self._by_ontology_id = {
            ingredient.ontology_id: ingredient for ingredient in ingredients
        }

    def get_by_id(self, ingredient_id: str) -> Ingredient | None:
        return self._by_id.get(ingredient_id)

    def get_by_ontology_id(self, ontology_id: str) -> Ingredient | None:
        return self._by_ontology_id.get(ontology_id)

    def list_all(self) -> tuple[Ingredient, ...]:
        return self._ingredients


def create_catalog() -> IngredientCatalog:
    return InMemoryIngredientCatalog(
        (
            Ingredient(
                ingredient_id="tomato",
                canonical_name="tomato",
                category=IngredientCategory.VEGETABLE,
                ontology_id="Tomato",
            ),
            Ingredient(
                ingredient_id="paprika",
                canonical_name="paprika",
                category=IngredientCategory.SPICE,
                ontology_id="Paprika",
            ),
        )
    )


def test_get_by_id_returns_matching_ingredient() -> None:
    ingredient = create_catalog().get_by_id("tomato")

    assert ingredient is not None
    assert ingredient.ontology_id == "Tomato"


def test_get_by_ontology_id_returns_matching_ingredient() -> None:
    ingredient = create_catalog().get_by_ontology_id("Paprika")

    assert ingredient is not None
    assert ingredient.ingredient_id == "paprika"


def test_unknown_identifier_returns_none() -> None:
    catalog = create_catalog()

    assert catalog.get_by_id("unknown") is None
    assert catalog.get_by_ontology_id("Unknown") is None


def test_list_all_returns_immutable_collection() -> None:
    ingredients = create_catalog().list_all()

    assert isinstance(ingredients, tuple)
    assert [item.ingredient_id for item in ingredients] == ["tomato", "paprika"]
