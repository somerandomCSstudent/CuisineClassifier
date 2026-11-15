from __future__ import annotations

from cuisine_assistant.domain.models import ExtractedIngredient
from cuisine_assistant.domain.ports import EntitySearch, IngredientExtractor, IngredientMentionParser


class StructuredIngredientExtractor(IngredientExtractor):
    """Extracts ingredients from one-item-per-line or semicolon-separated lists."""

    def __init__(self, parser: IngredientMentionParser, search: EntitySearch) -> None:
        self._parser = parser
        self._search = search

    def extract(self, recipe_text: str) -> list[ExtractedIngredient]:
        results: list[ExtractedIngredient] = []
        for mention in self._parser.parse(recipe_text):
            candidates = self._search.find(mention.normalized_text)
            best = candidates[0] if candidates else None
            results.append(
                ExtractedIngredient(
                    surface_form=mention.surface_form,
                    normalized_text=mention.normalized_text,
                    start_char=mention.start_char,
                    end_char=mention.end_char,
                    matched_ingredient_id=best.entity_id if best else None,
                    match_method=best.method if best else "unmatched",
                    match_score=best.score if best else 0.0,
                    candidate_ids=[candidate.entity_id for candidate in candidates],
                )
            )
        return results
