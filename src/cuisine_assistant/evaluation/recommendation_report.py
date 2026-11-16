from __future__ import annotations

from cuisine_assistant.evaluation.dish_recommendation import (
    RecommendationEvaluation,
)


class RecommendationEvaluationFormatter:
    """Formats dish-recommendation metrics as a report-ready text block."""

    def format(self, evaluation: RecommendationEvaluation) -> str:
        return "\n".join(
            [
                "Dish recommendation evaluation",
                f"Examples: {evaluation.total_examples}",
                (
                    "Examples with recommendations: "
                    f"{evaluation.examples_with_recommendations}"
                ),
                f"Top-1 hit rate: {evaluation.top_1_hit_rate:.2%}",
                f"Top-3 hit rate: {evaluation.top_3_hit_rate:.2%}",
                (
                    "Mean reciprocal rank: "
                    f"{evaluation.mean_reciprocal_rank:.4f}"
                ),
                (
                    "Mean top ingredient match: "
                    f"{evaluation.mean_top_ingredient_match:.2%}"
                ),
            ]
        )
