from __future__ import annotations

from collections.abc import Iterable

from cuisine_assistant.application import (
    ClassifyRecipeUseCase,
    LinkIngredientsResult,
)
from cuisine_assistant.domain.models import (
    ClassificationResult,
    Cuisine,
    CuisinePrediction,
    ExtractedIngredient,
    Ingredient,
    IngredientCategory,
    LinkedIngredient,
    MatchMethod,
)
from cuisine_assistant.domain.ports import CuisineClassifier


class StubLinkIngredientsUseCase:
    """Returns deterministic linking results for use-case tests."""

    def __init__(self, result: LinkIngredientsResult) -> None:
        self._result = result
        self.received_text: str | None = None

    def execute(self, recipe_text: str) -> LinkIngredientsResult:
        self.received_text = recipe_text
        return self._result


class StubCuisineClassifier(CuisineClassifier):
    """Returns deterministic cuisine predictions for use-case tests."""

    def __init__(self, result: ClassificationResult) -> None:
        self._result = result
        self.received_ingredients: tuple[LinkedIngredient, ...] = ()

    def classify(
        self,
        linked_ingredients: Iterable[LinkedIngredient],
    ) -> ClassificationResult:
        self.received_ingredients = tuple(linked_ingredients)
        return self._result


def create_linked_tomato() -> LinkedIngredient:
    """Creates a catalog-linked tomato fixture."""
    extracted = ExtractedIngredient(
        surface_form="tomatoes",
        normalized_text="tomatoes",
        start_char=0,
        end_char=8,
        matched_ingredient_id="tomato",
        match_method=MatchMethod.ALIAS,
        match_score=1.0,
        candidate_ids=("tomato",),
    )
    ingredient = Ingredient(
        ingredient_id="tomato",
        canonical_name="tomato",
        category=IngredientCategory.VEGETABLE,
        ontology_id="Tomato",
    )
    return LinkedIngredient(extracted, ingredient, None)


def create_prediction_result() -> ClassificationResult:
    """Creates a deterministic Italian cuisine prediction."""
    cuisine = Cuisine(
        cuisine_id="italian",
        name="Italian cuisine",
        ontology_id="ItalianCuisine",
        cuisine_group="EuropeanCuisine",
    )
    return ClassificationResult(
        predictions=(
            CuisinePrediction(
                cuisine=cuisine,
                score=0.88,
                rank=1,
            ),
        )
    )


def test_executes_linking_before_classification() -> None:
    linked_tomato = create_linked_tomato()
    linker = StubLinkIngredientsUseCase(
        LinkIngredientsResult(linked_ingredients=(linked_tomato,))
    )
    classifier = StubCuisineClassifier(create_prediction_result())
    use_case = ClassifyRecipeUseCase(linker, classifier)

    result = use_case.execute("Tomatoes with basil")

    assert linker.received_text == "Tomatoes with basil"
    assert classifier.received_ingredients == (linked_tomato,)
    assert result.classification.best_prediction is not None
    assert result.classification.best_prediction.cuisine.cuisine_id == "italian"
    assert result.has_predictions is True


def test_preserves_linking_result_for_explanations() -> None:
    linked_tomato = create_linked_tomato()
    linking_result = LinkIngredientsResult(linked_ingredients=(linked_tomato,))
    use_case = ClassifyRecipeUseCase(
        StubLinkIngredientsUseCase(linking_result),
        StubCuisineClassifier(create_prediction_result()),
    )

    result = use_case.execute("Tomatoes")

    assert result.linking is linking_result
    assert result.linking.catalog_only_count == 1


def test_returns_empty_classification_when_classifier_has_no_prediction() -> None:
    use_case = ClassifyRecipeUseCase(
        StubLinkIngredientsUseCase(LinkIngredientsResult(())),
        StubCuisineClassifier(ClassificationResult()),
    )

    result = use_case.execute("")

    assert result.linking.linked_ingredients == ()
    assert result.classification.predictions == ()
    assert result.has_predictions is False
