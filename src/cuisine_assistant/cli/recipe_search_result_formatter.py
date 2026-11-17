from __future__ import annotations

from cuisine_assistant.application import SearchRecipesResult


class RecipeSearchResultFormatter:
    """Formats ranked recipe documents and their retrieval evidence for the CLI."""

    def __init__(self, maximum_matches: int = 5) -> None:
        if maximum_matches < 1:
            raise ValueError("maximum_matches must be at least 1")
        self._maximum_matches = maximum_matches

    def format(self, result: SearchRecipesResult) -> str:
        """Returns a readable top-k recipe search report."""
        hits = result.search.hits[: self._maximum_matches]
        if not hits:
            return self._format_missing_result(result)

        best = hits[0]
        lines = [
            "Best recipe document match:",
            best.document.title,
            "",
            f"Search score: {best.score:.2%}",
            "",
            f"Top {len(hits)} recipe document matches:",
        ]
        for hit in hits:
            lines.append(
                f"{hit.rank}. {hit.document.title}: {hit.score:.2%}"
            )

        lines.extend(
            [
                "",
                "Search evidence for the best match:",
                f"- BM25 content match: {best.evidence.content_match_score:.2%}",
                f"- Title match: {best.evidence.title_match_score:.2%}",
                f"- Cuisine match: {best.evidence.cuisine_match_score:.2%}",
            ]
        )
        if best.evidence.matched_terms:
            lines.append(
                "- Matched terms: "
                + ", ".join(best.evidence.matched_terms)
            )
        if best.evidence.matched_ingredient_ids:
            lines.append(
                "- Matched ingredients: "
                + ", ".join(
                    identifier.replace("_", " ")
                    for identifier in best.evidence.matched_ingredient_ids
                )
            )

        lines.extend(
            [
                "",
                "Best document details:",
                f"- Recipe ID: {best.document.recipe_id}",
                f"- Dish ID: {best.document.dish_id}",
                f"- Cuisine: {best.document.cuisine_id or 'not specified'}",
                f"- Source: {best.document.source}",
            ]
        )
        if best.document.license_name:
            lines.append(f"- License: {best.document.license_name}")
        if best.document.source_url:
            lines.append(f"- Source URL: {best.document.source_url}")
        if best.document.ingredient_text:
            lines.append(
                f"- Ingredients: {best.document.ingredient_text}"
            )
        if best.document.instruction_text:
            lines.append(
                f"- Instructions: {best.document.instruction_text}"
            )
        elif best.document.raw_text:
            lines.append(f"- Text: {best.document.raw_text}")

        if result.cuisine_preference is not None:
            lines.extend(
                [
                    "",
                    "Detected cuisine preference:",
                    result.cuisine_preference.source_text,
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _format_missing_result(result: SearchRecipesResult) -> str:
        lines = [
            "Recipe document search:",
            "No matching recipe document is available.",
        ]
        if result.search.query.required_ingredient_ids:
            lines.append(
                "Required ingredients: "
                + ", ".join(
                    identifier.replace("_", " ")
                    for identifier in result.search.query.required_ingredient_ids
                )
            )
        if result.search.query.cuisine_id:
            lines.append(
                f"Cuisine filter: {result.search.query.cuisine_id}"
            )
        return "\n".join(lines)
