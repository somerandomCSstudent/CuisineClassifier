from __future__ import annotations

import pytest

from cuisine_assistant.domain.models import (
    ClassificationResult,
    Cuisine,
    CuisinePrediction,
    IngredientEvidence,
)


def create_cuisine(cuisine_id: str, name: str, ontology_id: str) -> Cuisine:
    """Creates a cuisine fixture for domain-model tests."""
    return Cuisine(
        cuisine_id=cuisine_id,
        name=name,
        ontology_id=ontology_id,
        cuisine_group="EuropeanCuisine",
    )


def test_cuisine_stores_canonical_and_ontology_identifiers() -> None:
    cuisine = create_cuisine("hungarian", "Hungarian cuisine", "HungarianCuisine")
    assert cuisine.cuisine_id == "hungarian"
    assert cuisine.ontology_id == "HungarianCuisine"


def test_ingredient_evidence_validates_weight() -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        IngredientEvidence(
            ingredient_id="paprika",
            ingredient_ontology_id="Paprika",
            weight=1.2,
            explanation="Paprika is typical of Hungarian cuisine.",
        )


def test_prediction_requires_positive_rank() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        CuisinePrediction(
            cuisine=create_cuisine("italian", "Italian cuisine", "ItalianCuisine"),
            score=0.8,
            rank=0,
        )


def test_classification_result_returns_best_prediction() -> None:
    hungarian = create_cuisine("hungarian", "Hungarian cuisine", "HungarianCuisine")
    italian = create_cuisine("italian", "Italian cuisine", "ItalianCuisine")
    result = ClassificationResult(
        predictions=(
            CuisinePrediction(
                cuisine=hungarian,
                score=0.85,
                rank=1,
                evidence=(
                    IngredientEvidence(
                        ingredient_id="paprika",
                        ingredient_ontology_id="Paprika",
                        weight=0.95,
                        explanation="Paprika strongly supports Hungarian cuisine.",
                    ),
                ),
            ),
            CuisinePrediction(cuisine=italian, score=0.35, rank=2),
        ),
        unmatched_ingredients=("gochujang",),
    )
    assert result.best_prediction is not None
    assert result.best_prediction.cuisine.cuisine_id == "hungarian"
    assert result.unmatched_ingredients == ("gochujang",)


def test_classification_result_rejects_non_consecutive_ranks() -> None:
    cuisine = create_cuisine("hungarian", "Hungarian cuisine", "HungarianCuisine")
    with pytest.raises(ValueError, match="consecutive"):
        ClassificationResult(
            predictions=(CuisinePrediction(cuisine=cuisine, score=0.8, rank=2),)
        )


def test_classification_result_rejects_unsorted_scores() -> None:
    hungarian = create_cuisine("hungarian", "Hungarian cuisine", "HungarianCuisine")
    italian = create_cuisine("italian", "Italian cuisine", "ItalianCuisine")
    with pytest.raises(ValueError, match="descending score"):
        ClassificationResult(
            predictions=(
                CuisinePrediction(cuisine=hungarian, score=0.4, rank=1),
                CuisinePrediction(cuisine=italian, score=0.8, rank=2),
            )
        )


def test_empty_classification_has_no_best_prediction() -> None:
    assert ClassificationResult().best_prediction is None
