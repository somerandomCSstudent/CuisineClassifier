from __future__ import annotations

from cuisine_assistant.evaluation.recipe_retrieval import (
    RecipeRetrievalEvaluation,
)


class RecipeRetrievalEvaluationFormatter:
    """Formats recipe-search metrics as a report-ready text block."""

    def format(self, evaluation: RecipeRetrievalEvaluation) -> str:
        """Returns localized-independent metric labels for project reporting."""
        return "\n".join(
            [
                "Recipe document retrieval evaluation",
                f"Examples: {evaluation.total_examples}",
                f"Examples with results: {evaluation.examples_with_results}",
                f"Precision@1: {evaluation.precision_at_1:.2%}",
                f"Precision@3: {evaluation.precision_at_3:.2%}",
                f"Recall@3: {evaluation.recall_at_3:.2%}",
                (
                    "Mean reciprocal rank: "
                    f"{evaluation.mean_reciprocal_rank:.4f}"
                ),
                f"nDCG@5: {evaluation.ndcg_at_5:.4f}",
                f"Coverage: {evaluation.coverage:.2%}",
            ]
        )
