from __future__ import annotations

from collections.abc import Iterable

from cuisine_assistant.domain.models import (
    ExtractedIngredient,
    LinkedIngredient,
    MatchMethod,
)
from cuisine_assistant.domain.ports import (
    IngredientCatalog,
    KnowledgeGraphRepository,
)


class IngredientEntityLinker:
    """Links extracted ingredient IDs to catalog records and OWL entities."""

    def __init__(
        self,
        ingredient_catalog: IngredientCatalog,
        knowledge_graph: KnowledgeGraphRepository,
    ) -> None:
        self._ingredient_catalog = ingredient_catalog
        self._knowledge_graph = knowledge_graph

    def link(self, extracted: ExtractedIngredient) -> LinkedIngredient:
        if extracted.match_method is MatchMethod.UNMATCHED:
            return LinkedIngredient(
                extracted=extracted,
                ingredient=None,
                ontology_entity=None,
            )

        ingredient_id = extracted.matched_ingredient_id
        if ingredient_id is None:
            raise ValueError(
                "Matched extraction must contain matched_ingredient_id"
            )

        ingredient = self._ingredient_catalog.get_by_id(ingredient_id)
        if ingredient is None:
            raise LookupError(
                f"Ingredient is missing from the catalog: {ingredient_id}"
            )

        ontology_entity = self._knowledge_graph.get_entity_reference(
            ingredient.ontology_id
        )

        return LinkedIngredient(
            extracted=extracted,
            ingredient=ingredient,
            ontology_entity=ontology_entity,
        )

    def link_all(
        self,
        extracted_ingredients: Iterable[ExtractedIngredient],
    ) -> tuple[LinkedIngredient, ...]:
        return tuple(self.link(item) for item in extracted_ingredients)
