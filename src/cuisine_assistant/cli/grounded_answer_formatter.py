from __future__ import annotations

from cuisine_assistant.application import AskCuisineQuestionResult


class GroundedAnswerFormatter:
    """Formats a grounded cuisine answer, evidence, and sources for the CLI."""

    def format(self, result: AskCuisineQuestionResult) -> str:
        """Returns a readable grounded answer report with stable references."""
        answer = result.answer
        lines = [
            "Grounded cuisine answer:",
            answer.text,
        ]

        if answer.insufficient_evidence:
            lines.extend(
                [
                    "",
                    "Evidence status: insufficient grounded evidence",
                ]
            )
            return "\n".join(lines)

        if answer.citations:
            lines.extend(["", "Evidence:"])
            for citation in answer.citations:
                lines.append(
                    f"[{citation.reference_id}] "
                    f"{citation.reference_type}: "
                    f"{citation.description}"
                )

        if result.context.sources:
            lines.extend(["", "Sources:"])
            for index, source in enumerate(
                result.context.sources,
                start=1,
            ):
                location = source.source_url or source.source
                lines.append(
                    f"[S{index}] {source.title} | "
                    f"recipe_id={source.recipe_id} | {location}"
                )
                if source.license_name:
                    lines.append(
                        f"     License: {source.license_name}"
                    )

        if result.context.linked_ingredient_ids:
            lines.extend(
                [
                    "",
                    "Linked ingredients:",
                    ", ".join(
                        ingredient_id.replace("_", " ")
                        for ingredient_id in (
                            result.context.linked_ingredient_ids
                        )
                    ),
                ]
            )

        if result.context.cuisine_id:
            lines.extend(
                [
                    "",
                    "Detected cuisine preference:",
                    result.context.cuisine_id,
                ]
            )

        return "\n".join(lines)
