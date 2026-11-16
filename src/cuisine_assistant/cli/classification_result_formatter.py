from __future__ import annotations

from cuisine_assistant.application import ClassifyRecipeResult


class ClassificationResultFormatter:
    """Formats the best cuisine prediction and the top ranked cuisine matches."""

    def __init__(self, maximum_matches: int = 3) -> None:
        if maximum_matches < 1:
            raise ValueError("maximum_matches must be at least 1")
        self._maximum_matches = maximum_matches

    def format(self, result: ClassifyRecipeResult) -> str:
        predictions = result.classification.predictions
        if not predictions:
            return self._format_missing_prediction(result)

        best = predictions[0]
        lines = [
            "Most likely cuisine:",
            best.cuisine.name,
            "",
            f"Classification score: {best.score:.2%}",
            "",
            f"Top {min(self._maximum_matches, len(predictions))} cuisine matches:",
        ]

        for prediction in predictions[: self._maximum_matches]:
            lines.append(
                f"{prediction.rank}. {prediction.cuisine.name}: "
                f"{prediction.score:.2%}"
            )

        if best.evidence:
            lines.extend(["", "Supporting ingredients:"])
            for evidence in best.evidence:
                lines.append(
                    f"- {evidence.ingredient_id.replace('_', ' ')}: "
                    f"{evidence.weight:.2f}"
                )

        self._append_unmatched_ingredients(
            lines,
            result.classification.unmatched_ingredients,
        )
        return "\n".join(lines)

    @staticmethod
    def _format_missing_prediction(result: ClassifyRecipeResult) -> str:
        lines = [
            "Most likely cuisine:",
            "No cuisine prediction is available.",
        ]
        ClassificationResultFormatter._append_unmatched_ingredients(
            lines,
            result.classification.unmatched_ingredients,
        )
        return "\n".join(lines)

    @staticmethod
    def _append_unmatched_ingredients(
        lines: list[str],
        unmatched_ingredients: tuple[str, ...],
    ) -> None:
        if not unmatched_ingredients:
            return
        lines.extend(["", "Unmatched ingredients:"])
        lines.extend(f"- {ingredient}" for ingredient in unmatched_ingredients)
