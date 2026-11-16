from __future__ import annotations

from cuisine_assistant.application import RecommendDishesResult


class RecommendationResultFormatter:
    """Formats the best dish recommendation and the top ranked dish matches."""

    def __init__(self, maximum_matches: int = 3) -> None:
        if maximum_matches < 1:
            raise ValueError("maximum_matches must be at least 1")
        self._maximum_matches = maximum_matches

    def format(self, result: RecommendDishesResult) -> str:
        recommendations = result.recommendation.recommendations
        if not recommendations:
            return self._format_missing_recommendation(result)

        best = recommendations[0]
        lines = [
            "Best recommendation:",
            best.dish.name,
            "",
            f"Recommendation score: {best.score:.2%}",
            "",
            f"Top {min(self._maximum_matches, len(recommendations))} dish recommendations:",
        ]

        for recommendation in recommendations[: self._maximum_matches]:
            lines.append(
                f"{recommendation.rank}. {recommendation.dish.name}: "
                f"{recommendation.score:.2%}"
            )

        lines.extend(["", "Recommendation evidence:"])
        lines.append(
            f"- Ingredient match: "
            f"{best.evidence.ingredient_match_score:.2%}"
        )
        lines.append(
            f"- Cuisine match: {best.evidence.cuisine_match_score:.2%}"
        )
        if best.evidence.matched_ingredient_ids:
            lines.append(
                "- Matched ingredients: "
                + ", ".join(
                    ingredient_id.replace("_", " ")
                    for ingredient_id in best.evidence.matched_ingredient_ids
                )
            )
        if best.evidence.missing_ingredient_ids:
            lines.append(
                "- Additional dish ingredients: "
                + ", ".join(
                    ingredient_id.replace("_", " ")
                    for ingredient_id in best.evidence.missing_ingredient_ids
                )
            )

        preference = result.cuisine_preference
        if preference is not None:
            lines.extend(
                [
                    "",
                    "Detected cuisine preference:",
                    f"{preference.source_text} "
                    f"({'strict' if preference.strict else 'preferred'})",
                ]
            )
        else:
            best_prediction = result.classification.classification.best_prediction
            if best_prediction is not None:
                lines.extend(
                    [
                        "",
                        "Inferred cuisine used for recommendation:",
                        f"{best_prediction.cuisine.name}: "
                        f"{best_prediction.score:.2%}",
                    ]
                )

        return "\n".join(lines)

    @staticmethod
    def _format_missing_recommendation(
        result: RecommendDishesResult,
    ) -> str:
        lines = [
            "Best recommendation:",
            "No dish recommendation is available.",
        ]
        if result.cuisine_preference is not None:
            lines.extend(
                [
                    "",
                    "Detected cuisine preference:",
                    result.cuisine_preference.source_text,
                ]
            )
        return "\n".join(lines)
