from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class CuisineQuestionAnsweringOperation(Protocol):
    """Defines the grounded question-answer operation required by evaluation."""

    def execute(self, question: str):
        """Returns a grounded question-answer result."""
        ...


@dataclass(frozen=True)
class LabeledCuisineQuestion:
    """Represents one question with expected answerability and recipe evidence."""

    example_id: str
    question: str
    expected_answerable: bool
    relevant_recipe_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.expected_answerable and not self.relevant_recipe_ids:
            raise ValueError(
                "answerable examples require relevant_recipe_ids"
            )


@dataclass(frozen=True)
class GroundedQaEvaluation:
    """Stores aggregate groundedness and citation metrics for cuisine QA."""

    total_examples: int
    answered_examples: int
    correctly_answered_or_abstained: int
    answerability_accuracy: float
    citation_validity: float
    citation_coverage: float
    source_recipe_recall: float
    unsupported_answer_rate: float


class GroundedQaEvaluator:
    """Evaluates answerability, citations, source recall, and unsupported answers."""

    _REFERENCE_PATTERN = re.compile(r"\[([DGS]\d+)\]")

    def __init__(
        self,
        operation: CuisineQuestionAnsweringOperation,
    ) -> None:
        self._operation = operation

    def evaluate(
        self,
        examples: tuple[LabeledCuisineQuestion, ...],
    ) -> GroundedQaEvaluation:
        """Calculates deterministic grounded QA metrics over labeled questions."""
        if not examples:
            raise ValueError("at least one grounded QA example is required")

        answered_examples = 0
        correct_answerability = 0
        valid_citations = 0
        total_citations = 0
        answered_with_citations = 0
        source_recall_sum = 0.0
        unsupported_answers = 0

        for example in examples:
            result = self._operation.execute(example.question)
            answer = result.answer
            answered = not answer.insufficient_evidence
            if answered:
                answered_examples += 1

            if answered == example.expected_answerable:
                correct_answerability += 1

            available_ids = self._available_reference_ids(result.context)
            cited_ids = tuple(
                citation.reference_id for citation in answer.citations
            )
            text_ids = tuple(self._REFERENCE_PATTERN.findall(answer.text))
            all_claimed_ids = tuple(dict.fromkeys((*cited_ids, *text_ids)))
            total_citations += len(all_claimed_ids)
            valid_citations += sum(
                reference_id in available_ids
                for reference_id in all_claimed_ids
            )

            if answered and all_claimed_ids:
                answered_with_citations += 1
            if answered and not all_claimed_ids:
                unsupported_answers += 1
            if answer.insufficient_evidence and answer.citations:
                unsupported_answers += 1

            if example.relevant_recipe_ids:
                expected = set(example.relevant_recipe_ids)
                actual = set(answer.source_recipe_ids)
                source_recall_sum += len(expected.intersection(actual)) / len(
                    expected
                )
            else:
                source_recall_sum += 1.0 if not answer.source_recipe_ids else 0.0

        total = len(examples)
        return GroundedQaEvaluation(
            total_examples=total,
            answered_examples=answered_examples,
            correctly_answered_or_abstained=correct_answerability,
            answerability_accuracy=correct_answerability / total,
            citation_validity=(
                valid_citations / total_citations
                if total_citations
                else 1.0
            ),
            citation_coverage=(
                answered_with_citations / answered_examples
                if answered_examples
                else 1.0
            ),
            source_recipe_recall=source_recall_sum / total,
            unsupported_answer_rate=unsupported_answers / total,
        )

    @staticmethod
    def _available_reference_ids(context) -> set[str]:
        document_ids = {
            f"D{index}"
            for index, _ in enumerate(context.documents, start=1)
        }
        graph_ids = {
            f"G{index}"
            for index, _ in enumerate(context.graph_facts, start=1)
        }
        source_ids = {
            f"S{index}"
            for index, _ in enumerate(context.sources, start=1)
        }
        return document_ids | graph_ids | source_ids
