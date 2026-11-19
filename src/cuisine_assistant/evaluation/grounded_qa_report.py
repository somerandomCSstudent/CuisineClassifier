from __future__ import annotations

from cuisine_assistant.evaluation.grounded_qa import GroundedQaEvaluation


class GroundedQaEvaluationFormatter:
    """Formats grounded cuisine question-answer metrics for reporting."""

    def format(self, evaluation: GroundedQaEvaluation) -> str:
        """Returns one stable text report containing all QA metrics."""
        return "\n".join(
            [
                "Grounded cuisine question-answer evaluation",
                f"Examples: {evaluation.total_examples}",
                f"Answered examples: {evaluation.answered_examples}",
                (
                    "Correct answers or abstentions: "
                    f"{evaluation.correctly_answered_or_abstained}"
                ),
                (
                    "Answerability accuracy: "
                    f"{evaluation.answerability_accuracy:.2%}"
                ),
                f"Citation validity: {evaluation.citation_validity:.2%}",
                f"Citation coverage: {evaluation.citation_coverage:.2%}",
                (
                    "Source recipe recall: "
                    f"{evaluation.source_recipe_recall:.2%}"
                ),
                (
                    "Unsupported answer rate: "
                    f"{evaluation.unsupported_answer_rate:.2%}"
                ),
            ]
        )
