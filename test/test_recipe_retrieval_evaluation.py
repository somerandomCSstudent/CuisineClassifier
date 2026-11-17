from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuisine_assistant.evaluation.recipe_retrieval import (
    LabeledRecipeSearchExample,
    RecipeRetrievalEvaluator,
)


class StubRecipeSearchOperation:
    """Returns deterministic recipe rankings for evaluation tests."""

    def __init__(self, rankings: dict[str, tuple[str, ...]]) -> None:
        self._rankings = rankings

    def execute(self, query_text: str):
        hits = tuple(
            SimpleNamespace(
                document=SimpleNamespace(recipe_id=recipe_id)
            )
            for recipe_id in self._rankings.get(query_text, ())
        )
        return SimpleNamespace(
            search=SimpleNamespace(hits=hits)
        )


def test_calculates_ranked_retrieval_metrics() -> None:
    """Verifies precision, recall, MRR, nDCG, and coverage calculations."""
    evaluator = RecipeRetrievalEvaluator(
        StubRecipeSearchOperation(
            {
                "one": ("r1", "x"),
                "two": ("x", "r2", "r3"),
                "three": (),
            }
        )
    )
    examples = (
        LabeledRecipeSearchExample("e1", "one", ("r1",)),
        LabeledRecipeSearchExample("e2", "two", ("r2", "r3")),
        LabeledRecipeSearchExample("e3", "three", ("r4",)),
    )

    result = evaluator.evaluate(examples)

    assert result.precision_at_1 == pytest.approx(1 / 3)
    assert result.precision_at_3 == pytest.approx(1 / 3)
    assert result.recall_at_3 == pytest.approx(2 / 3)
    assert result.mean_reciprocal_rank == pytest.approx(0.5)
    assert result.examples_with_results == 2
    assert result.coverage == pytest.approx(2 / 3)
    assert 0.0 < result.ndcg_at_5 < 1.0


def test_rejects_empty_evaluation_set() -> None:
    """Verifies retrieval evaluation requires at least one labeled query."""
    evaluator = RecipeRetrievalEvaluator(
        StubRecipeSearchOperation({})
    )

    with pytest.raises(ValueError, match="at least one"):
        evaluator.evaluate(())
