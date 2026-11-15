from __future__ import annotations

from collections.abc import Iterable

from cuisine_assistant.domain.models import (
    ClassificationResult,
    Cuisine,
    CuisinePrediction,
    IngredientEvidence,
    LinkedIngredient,
)
from cuisine_assistant.domain.ports import (
    CuisineCatalog,
    CuisineClassifier,
    CuisineTypicalityRepository,
)


class TypicalityCuisineClassifier(CuisineClassifier):
    """Ranks cuisines by weighted typicality of the linked recipe ingredients."""

    def __init__(
        self,
        cuisine_catalog: CuisineCatalog,
        typicality_repository: CuisineTypicalityRepository,
        *,
        maximum_predictions: int = 3,
        minimum_score: float = 0.0,
    ) -> None:
        if maximum_predictions < 1:
            raise ValueError("maximum_predictions must be at least 1")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0.0 and 1.0")
        self._cuisine_catalog = cuisine_catalog
        self._typicality_repository = typicality_repository
        self._maximum_predictions = maximum_predictions
        self._minimum_score = minimum_score

    def classify(
        self,
        linked_ingredients: Iterable[LinkedIngredient],
    ) -> ClassificationResult:
        items = tuple(linked_ingredients)
        scored_ingredients = self._unique_catalog_ingredients(items)
        unmatched = self._unmatched_surface_forms(items)
        if not scored_ingredients:
            return ClassificationResult(unmatched_ingredients=unmatched)

        candidates: list[tuple[Cuisine, float, tuple[IngredientEvidence, ...]]] = []
        for cuisine in self._cuisine_catalog.list_all():
            score, evidence = self._score_cuisine(cuisine, scored_ingredients)
            if score > self._minimum_score:
                candidates.append((cuisine, score, evidence))

        candidates.sort(key=lambda item: (-item[1], item[0].cuisine_id))
        selected = candidates[: self._maximum_predictions]
        predictions = tuple(
            CuisinePrediction(
                cuisine=cuisine,
                score=score,
                rank=rank,
                evidence=evidence,
            )
            for rank, (cuisine, score, evidence) in enumerate(selected, start=1)
        )
        return ClassificationResult(
            predictions=predictions,
            unmatched_ingredients=unmatched,
        )

    def _score_cuisine(
        self,
        cuisine: Cuisine,
        ingredients,
    ) -> tuple[float, tuple[IngredientEvidence, ...]]:
        evidence: list[IngredientEvidence] = []
        total_weight = 0.0
        for ingredient in ingredients:
            weight = self._typicality_repository.get_weight(
                cuisine.cuisine_id,
                ingredient.ingredient_id,
            )
            if weight <= 0.0:
                continue
            total_weight += weight
            evidence.append(
                IngredientEvidence(
                    ingredient_id=ingredient.ingredient_id,
                    ingredient_ontology_id=ingredient.ontology_id,
                    weight=weight,
                    explanation=(
                        f"{ingredient.canonical_name} contributes {weight:.2f} "
                        f"to {cuisine.name}."
                    ),
                )
            )

        score = total_weight / len(ingredients)
        evidence.sort(key=lambda item: (-item.weight, item.ingredient_id))
        return score, tuple(evidence)

    @staticmethod
    def _unique_catalog_ingredients(
        linked_ingredients: tuple[LinkedIngredient, ...],
    ):
        unique = {}
        for item in linked_ingredients:
            if item.ingredient is not None:
                unique.setdefault(item.ingredient.ingredient_id, item.ingredient)
        return tuple(unique.values())

    @staticmethod
    def _unmatched_surface_forms(
        linked_ingredients: tuple[LinkedIngredient, ...],
    ) -> tuple[str, ...]:
        return tuple(
            item.extracted.surface_form
            for item in linked_ingredients
            if item.ingredient is None
        )
