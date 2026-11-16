from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuisine_assistant.domain.recommendation_models import (
    Dish,
    DishRecommendation,
    RecommendationEvidence,
)
from cuisine_assistant.evaluation import (
    DishRecommendationEvaluator,
    LabeledRecommendationExample,
    RecommendationEvaluationFormatter,
)


class StubRecommendationUseCase:
    """Returns deterministic ranked dish recommendations for evaluation tests."""

    def __init__(self, results_by_text: dict[str, tuple[str, ...]]) -> None:
        self._results_by_text = results_by_text

    def execute(self, user_text: str):
        dish_ids = self._results_by_text.get(user_text, ())
        recommendations = tuple(
            DishRecommendation(
                dish=Dish(
                    dish_id=dish_id,
                    name=dish_id.replace("_", " ").title(),
                    ontology_id=dish_id.title().replace("_", ""),
                    dish_category="MainCourse",
                    cuisine_id="test",
                ),
                score=max(0.1, 1.0 - (rank - 1) * 0.2),
                rank=rank,
                evidence=RecommendationEvidence(
                    matched_ingredient_ids=("test",),
                    missing_ingredient_ids=(),
                    ingredient_match_score=0.75,
                    cuisine_match_score=1.0,
                ),
            )
            for rank, dish_id in enumerate(dish_ids, start=1)
        )
        return SimpleNamespace(
            recommendation=SimpleNamespace(recommendations=recommendations)
        )


def test_calculates_top_k_and_reciprocal_rank_metrics() -> None:
    """Verifies top-one, top-three, and reciprocal-rank calculations."""
    evaluator = DishRecommendationEvaluator(
        StubRecommendationUseCase(
            {
                "one": ("dish_a", "dish_b"),
                "two": ("dish_x", "dish_b"),
                "three": (),
            }
        ),
        top_k=3,
    )
    examples = (
        LabeledRecommendationExample("e1", "one", ("dish_a",)),
        LabeledRecommendationExample("e2", "two", ("dish_b",)),
        LabeledRecommendationExample("e3", "three", ("dish_c",)),
    )

    result = evaluator.evaluate(examples)

    assert result.top_1_hit_rate == pytest.approx(1 / 3)
    assert result.top_3_hit_rate == pytest.approx(2 / 3)
    assert result.mean_reciprocal_rank == pytest.approx(0.5)
    assert result.examples_with_recommendations == 2
    assert result.mean_top_ingredient_match == pytest.approx(0.75)


def test_formatter_produces_report_ready_output() -> None:
    """Verifies recommendation metrics can be copied into the report."""
    evaluation = DishRecommendationEvaluator(
        StubRecommendationUseCase({"one": ("dish_a",)}),
    ).evaluate(
        (LabeledRecommendationExample("e1", "one", ("dish_a",)),)
    )

    output = RecommendationEvaluationFormatter().format(evaluation)

    assert "Top-1 hit rate: 100.00%" in output
    assert "Top-3 hit rate: 100.00%" in output
    assert "Mean reciprocal rank: 1.0000" in output
