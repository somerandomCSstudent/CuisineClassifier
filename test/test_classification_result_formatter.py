from __future__ import annotations

import pytest

from cuisine_assistant.application import (
    ClassifyRecipeResult,
    LinkIngredientsResult,
)
from cuisine_assistant.cli.classification_result_formatter import (
    ClassificationResultFormatter,
)
from cuisine_assistant.domain.models import (
    ClassificationResult,
    Cuisine,
    CuisinePrediction,
    IngredientEvidence,
)


def create_cuisine(cuisine_id: str, name: str) -> Cuisine:
    """Creates a cuisine fixture for formatter tests."""
    return Cuisine(
        cuisine_id=cuisine_id,
        name=name,
        ontology_id=f"{cuisine_id.title()}Cuisine",
        cuisine_group="TestCuisine",
    )


def create_result() -> ClassifyRecipeResult:
    """Creates three ranked cuisine predictions for formatter tests."""
    return ClassifyRecipeResult(
        linking=LinkIngredientsResult(()),
        classification=ClassificationResult(
            predictions=(
                CuisinePrediction(
                    cuisine=create_cuisine("hungarian", "Hungarian cuisine"),
                    score=0.6833,
                    rank=1,
                    evidence=(
                        IngredientEvidence(
                            ingredient_id="paprika",
                            ingredient_ontology_id="Paprika",
                            weight=0.95,
                            explanation="Paprika supports Hungarian cuisine.",
                        ),
                        IngredientEvidence(
                            ingredient_id="sour_cream",
                            ingredient_ontology_id="SourCream",
                            weight=0.80,
                            explanation="Sour cream supports Hungarian cuisine.",
                        ),
                    ),
                ),
                CuisinePrediction(
                    cuisine=create_cuisine("thai", "Thai cuisine"),
                    score=0.1167,
                    rank=2,
                ),
                CuisinePrediction(
                    cuisine=create_cuisine("italian", "Italian cuisine"),
                    score=0.05,
                    rank=3,
                ),
            ),
            unmatched_ingredients=("gochujang",),
        ),
    )


def test_formats_best_prediction_and_top_three_matches() -> None:
    """Verifies the best answer is highlighted before the top-three ranking."""
    output = ClassificationResultFormatter().format(create_result())

    assert "Most likely cuisine:\nHungarian cuisine" in output
    assert "Classification score: 68.33%" in output
    assert "Top 3 cuisine matches:" in output
    assert "1. Hungarian cuisine: 68.33%" in output
    assert "2. Thai cuisine: 11.67%" in output
    assert "3. Italian cuisine: 5.00%" in output
    assert "Supporting ingredients:" in output
    assert "- paprika: 0.95" in output
    assert "- sour cream: 0.80" in output
    assert "Unmatched ingredients:" in output
    assert "- gochujang" in output


def test_formats_only_available_positive_matches() -> None:
    """Verifies the formatter does not invent missing ranked predictions."""
    result = create_result()
    shortened = ClassifyRecipeResult(
        linking=result.linking,
        classification=ClassificationResult(
            predictions=result.classification.predictions[:2],
        ),
    )

    output = ClassificationResultFormatter().format(shortened)

    assert "Top 2 cuisine matches:" in output
    assert "3. Italian cuisine" not in output


def test_formats_missing_prediction_and_unmatched_ingredients() -> None:
    """Verifies useful output when classification produces no prediction."""
    result = ClassifyRecipeResult(
        linking=LinkIngredientsResult(()),
        classification=ClassificationResult(
            unmatched_ingredients=("unknown ingredient",),
        ),
    )

    output = ClassificationResultFormatter().format(result)

    assert "No cuisine prediction is available." in output
    assert "- unknown ingredient" in output


def test_rejects_invalid_maximum_matches() -> None:
    """Verifies formatter configuration rejects a non-positive result limit."""
    with pytest.raises(ValueError, match="maximum_matches"):
        ClassificationResultFormatter(maximum_matches=0)
