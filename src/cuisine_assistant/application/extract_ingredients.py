from __future__ import annotations

from cuisine_assistant.domain.models import ExtractedIngredient
from cuisine_assistant.domain.ports import IngredientExtractor


class ExtractIngredientsUseCase:
    def __init__(self, extractor: IngredientExtractor) -> None:
        self._extractor = extractor

    def execute(self, recipe_text: str) -> list[ExtractedIngredient]:
        if not recipe_text.strip():
            return []
        return self._extractor.extract(recipe_text)
