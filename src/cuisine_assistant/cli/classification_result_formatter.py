from __future__ import annotations

from cuisine_assistant.application import ClassifyRecipeResult


class ClassificationResultFormatter:
    """Formats cuisine classification results for the English command-line interface."""

    def format(self, result: ClassifyRecipeResult) -> str:
        lines = ["Cuisine classification:"]
        if not result.classification.predictions:
            lines.append("No cuisine prediction is available.")
        else:
            for prediction in result.classification.predictions:
                lines.append(
                    f"{prediction.rank}. {prediction.cuisine.name}: "
                    f"{prediction.score:.2%}"
                )
                for evidence in prediction.evidence:
                    lines.append(f"   - {evidence.explanation}")

        if result.classification.unmatched_ingredients:
            lines.append("")
            lines.append("Unmatched ingredients:")
            lines.extend(
                f"- {ingredient}"
                for ingredient in result.classification.unmatched_ingredients
            )

        return "\n".join(lines)
