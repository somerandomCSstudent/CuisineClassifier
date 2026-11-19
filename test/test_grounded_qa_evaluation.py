from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuisine_assistant.evaluation.grounded_qa import (
    GroundedQaEvaluator,
    LabeledCuisineQuestion,
)
from cuisine_assistant.rag import GroundedAnswer, GroundedCitation


class StubQuestionOperation:
    """Returns configured grounded QA outcomes for evaluation tests."""

    def __init__(self, outcomes: dict[str, object]) -> None:
        self._outcomes = outcomes

    def execute(self, question: str):
        return self._outcomes[question]


def context(document_count=1, graph_count=1, source_count=1):
    """Creates context collections with deterministic reference counts."""
    return SimpleNamespace(
        documents=tuple(object() for _ in range(document_count)),
        graph_facts=tuple(object() for _ in range(graph_count)),
        sources=tuple(object() for _ in range(source_count)),
    )


def test_calculates_grounded_answer_and_abstention_metrics() -> None:
    """Verifies citation, source, unsupported-answer, and abstention metrics."""
    answerable = SimpleNamespace(
        answer=GroundedAnswer(
            "Supported answer [D1] [G1]",
            citations=(
                GroundedCitation("D1", "document", "Recipe"),
                GroundedCitation("G1", "graph_fact", "Fact"),
            ),
            source_recipe_ids=("r1",),
        ),
        context=context(),
    )
    unanswerable = SimpleNamespace(
        answer=GroundedAnswer(
            "I could not find enough grounded evidence.",
            insufficient_evidence=True,
        ),
        context=context(0, 0, 0),
    )
    evaluator = GroundedQaEvaluator(
        StubQuestionOperation(
            {"answerable": answerable, "unanswerable": unanswerable}
        )
    )

    result = evaluator.evaluate(
        (
            LabeledCuisineQuestion("e1", "answerable", True, ("r1",)),
            LabeledCuisineQuestion("e2", "unanswerable", False),
        )
    )

    assert result.answerability_accuracy == pytest.approx(1.0)
    assert result.citation_validity == pytest.approx(1.0)
    assert result.citation_coverage == pytest.approx(1.0)
    assert result.source_recipe_recall == pytest.approx(1.0)
    assert result.unsupported_answer_rate == pytest.approx(0.0)


def test_detects_invalid_and_unsupported_answer() -> None:
    """Verifies nonexistent references and uncited answers reduce metrics."""
    invalid = SimpleNamespace(
        answer=GroundedAnswer(
            "Unsupported answer [G9]",
            source_recipe_ids=("wrong",),
        ),
        context=context(),
    )
    evaluator = GroundedQaEvaluator(
        StubQuestionOperation({"question": invalid})
    )

    result = evaluator.evaluate(
        (LabeledCuisineQuestion("e1", "question", True, ("r1",)),)
    )

    assert result.citation_validity == 0.0
    assert result.citation_coverage == 1.0
    assert result.source_recipe_recall == 0.0


def test_rejects_empty_evaluation_set() -> None:
    """Verifies grounded QA evaluation requires at least one example."""
    with pytest.raises(ValueError, match="at least one"):
        GroundedQaEvaluator(StubQuestionOperation({})).evaluate(())
