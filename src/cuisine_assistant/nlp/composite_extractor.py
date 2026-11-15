from __future__ import annotations

from cuisine_assistant.domain.models import ExtractedIngredient
from cuisine_assistant.domain.ports import IngredientExtractor


class CompositeIngredientExtractor(IngredientExtractor):
    """Combines extractors without coupling callers to concrete NLP methods."""

    def __init__(self, extractors: list[IngredientExtractor]) -> None:
        self._extractors = list(extractors)

    def extract(self, recipe_text: str) -> list[ExtractedIngredient]:
        best_by_span: dict[tuple[int, int, str], ExtractedIngredient] = {}
        for extractor in self._extractors:
            for item in extractor.extract(recipe_text):
                key = (item.start_char, item.end_char, item.normalized_text)
                previous = best_by_span.get(key)
                if previous is None or item.match_score > previous.match_score:
                    best_by_span[key] = item
        return sorted(best_by_span.values(), key=lambda item: (item.start_char, item.end_char))
