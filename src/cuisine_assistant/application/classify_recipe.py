from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.application.link_ingredients import (
    LinkIngredientsResult,
    LinkIngredientsUseCase,
)
from cuisine_assistant.domain.models import ClassificationResult
from cuisine_assistant.domain.ports import CuisineClassifier


@dataclass(frozen=True)
class ClassifyRecipeResult:
    """Combines ingredient-linking details with cultural cuisine predictions."""

    linking: LinkIngredientsResult
    classification: ClassificationResult

    @property
    def has_predictions(self) -> bool:
        """Returns whether at least one cuisine prediction is available."""
        return bool(self.classification.predictions)


class ClassifyRecipeUseCase:
    """Links recipe ingredients and classifies the recipe into cuisines."""

    def __init__(
        self,
        link_ingredients: LinkIngredientsUseCase,
        classifier: CuisineClassifier,
    ) -> None:
        self._link_ingredients = link_ingredients
        self._classifier = classifier

    def execute(self, recipe_text: str) -> ClassifyRecipeResult:
        linking_result = self._link_ingredients.execute(recipe_text)
        classification_result = self._classifier.classify(
            linking_result.linked_ingredients
        )
        return ClassifyRecipeResult(
            linking=linking_result,
            classification=classification_result,
        )
