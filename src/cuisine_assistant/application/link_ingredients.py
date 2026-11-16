from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.domain.models import LinkedIngredient
from cuisine_assistant.domain.ports import IngredientExtractor
from cuisine_assistant.linking import IngredientEntityLinker


@dataclass(frozen=True)
class LinkIngredientsResult:
    linked_ingredients: tuple[LinkedIngredient, ...]

    @property
    def linked_count(self) -> int:
        return sum(
            item.is_linked_to_knowledge_graph
            for item in self.linked_ingredients
        )

    @property
    def catalog_only_count(self) -> int:
        return sum(
            item.ingredient is not None
            and not item.is_linked_to_knowledge_graph
            for item in self.linked_ingredients
        )

    @property
    def unmatched_count(self) -> int:
        return sum(
            item.ingredient is None
            for item in self.linked_ingredients
        )


class LinkIngredientsUseCase:
    """Extracts ingredients and links them to canonical and OWL entities."""

    def __init__(
        self,
        extractor: IngredientExtractor,
        linker: IngredientEntityLinker,
    ) -> None:
        self._extractor = extractor
        self._linker = linker

    def execute(self, recipe_text: str) -> LinkIngredientsResult:
        if not recipe_text or not recipe_text.strip():
            return LinkIngredientsResult(linked_ingredients=())

        extracted = self._extractor.extract(recipe_text)
        linked = self._linker.link_all(extracted)
        return LinkIngredientsResult(linked_ingredients=linked)
