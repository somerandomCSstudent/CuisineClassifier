from __future__ import annotations

from cuisine_assistant.application import (
    ClassifyRecipeResult,
    LinkIngredientsResult,
)
from cuisine_assistant.cli import ClassificationResultFormatter
from cuisine_assistant.domain.models import (
    ClassificationResult,
    Cuisine,
    CuisinePrediction,
    IngredientEvidence,
)


def test_formats_ranked_predictions_and_evidence() -> None:
    """Verifies readable English classification output."""
    cuisine = Cuisine(
        cuisine_id="hungarian",
        name="Hungarian cuisine",
        ontology_id="HungarianCuisine",
        cuisine_group="EuropeanCuisine",
    )
    result = ClassifyRecipeResult(
        linking=LinkIngredientsResult(()),
        classification=ClassificationResult(
            predictions=(
                CuisinePrediction(
                    cuisine=cuisine,
                    score=0.85,
                    rank=1,
                    evidence=(
                        IngredientEvidence(
                            ingredient_id="paprika",
                            ingredient_ontology_id="Paprika",
                            weight=0.95,
                            explanation=(
                                "paprika contributes 0.95 to Hungarian cuisine."
                            ),
                        ),
                    ),
                ),
            ),
            unmatched_ingredients=("gochujang",),
        ),
    )

    output = ClassificationResultFormatter().format(result)

    assert "1. Hungarian cuisine: 85.00%" in output
    assert "paprika contributes 0.95" in output
    assert "Unmatched ingredients:" in output
    assert "- gochujang" in output


def test_formats_missing_prediction() -> None:
    """Verifies output when no cuisine receives a positive score."""
    result = ClassifyRecipeResult(
        linking=LinkIngredientsResult(()),
        classification=ClassificationResult(),
    )

    assert "No cuisine prediction is available." in (
        ClassificationResultFormatter().format(result)
    )
