from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuisine_assistant.domain.models import (
    ClassificationResult,
    Cuisine,
    CuisinePrediction,
)
from cuisine_assistant.evaluation import (
    ClassificationEvaluationFormatter,
    CuisineClassificationEvaluator,
    LabeledRecipe,
)


class StubRecipeClassifier:
    """Returns deterministic ranked predictions for evaluation tests."""

    def __init__(self, predictions_by_text: dict[str, tuple[str, ...]]) -> None:
        self._predictions_by_text = predictions_by_text

    def execute(self, recipe_text: str):
        cuisine_ids = self._predictions_by_text.get(recipe_text, ())
        predictions = tuple(
            CuisinePrediction(
                cuisine=Cuisine(
                    cuisine_id=cuisine_id,
                    name=f"{cuisine_id.title()} cuisine",
                    ontology_id=f"{cuisine_id.title()}Cuisine",
                    cuisine_group="TestCuisine",
                ),
                score=max(0.0, 1.0 - (rank - 1) * 0.2),
                rank=rank,
            )
            for rank, cuisine_id in enumerate(cuisine_ids, start=1)
        )
        return SimpleNamespace(
            classification=ClassificationResult(predictions=predictions)
        )


def test_calculates_accuracy_macro_f1_and_top_k_accuracy() -> None:
    """Verifies aggregate metrics with correct and incorrect predictions."""
    classifier = StubRecipeClassifier(
        {
            "recipe one": ("hungarian", "italian"),
            "recipe two": ("italian", "hungarian"),
            "recipe three": ("italian", "thai"),
        }
    )
    examples = (
        LabeledRecipe("r1", "recipe one", "hungarian"),
        LabeledRecipe("r2", "recipe two", "hungarian"),
        LabeledRecipe("r3", "recipe three", "thai"),
    )

    result = CuisineClassificationEvaluator(classifier, top_k=2).evaluate(examples)

    assert result.accuracy == pytest.approx(1 / 3)
    assert result.top_k_accuracy == pytest.approx(1.0)
    assert result.no_prediction_count == 0
    assert 0.0 <= result.macro_f1 <= 1.0


def test_tracks_missing_predictions() -> None:
    """Verifies no-prediction accounting in the confusion matrix."""
    result = CuisineClassificationEvaluator(
        StubRecipeClassifier({}),
        top_k=3,
    ).evaluate((LabeledRecipe("r1", "unknown", "italian"),))

    assert result.no_prediction_count == 1
    assert result.accuracy == 0.0
    assert result.confusion_matrix["italian"]["__no_prediction__"] == 1


def test_formatter_includes_metrics_and_confusion_matrix() -> None:
    """Verifies evaluation output required for the written report."""
    evaluation = CuisineClassificationEvaluator(
        StubRecipeClassifier({"recipe": ("italian",)}),
        top_k=3,
    ).evaluate((LabeledRecipe("r1", "recipe", "italian"),))

    output = ClassificationEvaluationFormatter().format(evaluation)

    assert "Accuracy: 100.00%" in output
    assert "Macro F1: 100.00%" in output
    assert "Confusion matrix:" in output
