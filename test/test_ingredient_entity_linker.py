from __future__ import annotations

from collections.abc import Iterable

import pytest

from cuisine_assistant.domain.models import (
    ExtractedIngredient,
    Ingredient,
    IngredientCategory,
    MatchMethod,
    OntologyEntityReference,
)
from cuisine_assistant.domain.ports import (
    IngredientCatalog,
    KnowledgeGraphRepository,
)
from cuisine_assistant.linking import IngredientEntityLinker


class InMemoryIngredientCatalog(IngredientCatalog):
    def __init__(self) -> None:
        self._ingredients = (
            Ingredient(
                ingredient_id="tomato",
                canonical_name="tomato",
                category=IngredientCategory.VEGETABLE,
                ontology_id="Tomato",
            ),
            Ingredient(
                ingredient_id="paprika",
                canonical_name="paprika",
                category=IngredientCategory.SPICE,
                ontology_id="Paprika",
            ),
        )
        self._by_id = {
            ingredient.ingredient_id: ingredient
            for ingredient in self._ingredients
        }
        self._by_ontology_id = {
            ingredient.ontology_id: ingredient
            for ingredient in self._ingredients
        }

    def get_by_id(self, ingredient_id: str) -> Ingredient | None:
        return self._by_id.get(ingredient_id)

    def get_by_ontology_id(self, ontology_id: str) -> Ingredient | None:
        return self._by_ontology_id.get(ontology_id)

    def list_all(self) -> tuple[Ingredient, ...]:
        return self._ingredients


class InMemoryKnowledgeGraphRepository(KnowledgeGraphRepository):
    def __init__(self, existing_entities: set[str]) -> None:
        self._existing_entities = existing_entities

    def get_entity_reference(
        self,
        ontology_id: str,
    ) -> OntologyEntityReference | None:
        if ontology_id not in self._existing_entities:
            return None
        return OntologyEntityReference(
            ontology_id=ontology_id,
            iri=f"http://example.org/cuisine#{ontology_id}",
            rdf_types=("Ingredient",),
            exists_in_knowledge_graph=True,
        )

    def get_entity_types(self, ontology_id: str) -> tuple[str, ...]:
        return ("Ingredient",) if ontology_id in self._existing_entities else ()

    def find_dish_ids_by_ingredient_ids(
        self,
        ingredient_ontology_ids: Iterable[str],
    ) -> tuple[str, ...]:
        return ()

    def get_dish_ingredient_ids(self, dish_ontology_id: str) -> tuple[str, ...]:
        return ()

    def get_dish_cuisine_id(self, dish_ontology_id: str) -> str | None:
        return None


def matched_tomato() -> ExtractedIngredient:
    return ExtractedIngredient(
        surface_form="fresh tomatoes",
        normalized_text="fresh tomatoes",
        start_char=0,
        end_char=14,
        matched_ingredient_id="tomato",
        match_method=MatchMethod.ALIAS,
        match_score=1.0,
        candidate_ids=("tomato",),
    )


def unmatched_gochujang() -> ExtractedIngredient:
    return ExtractedIngredient(
        surface_form="gochujang",
        normalized_text="gochujang",
        start_char=0,
        end_char=9,
        matched_ingredient_id=None,
        match_method=MatchMethod.UNMATCHED,
        match_score=0.0,
    )


def create_linker(existing_entities: set[str]) -> IngredientEntityLinker:
    return IngredientEntityLinker(
        InMemoryIngredientCatalog(),
        InMemoryKnowledgeGraphRepository(existing_entities),
    )


def test_links_extraction_to_catalog_and_knowledge_graph() -> None:
    linked = create_linker({"Tomato"}).link(matched_tomato())

    assert linked.ingredient is not None
    assert linked.ingredient.ingredient_id == "tomato"
    assert linked.ontology_entity is not None
    assert linked.ontology_entity.ontology_id == "Tomato"
    assert linked.is_linked_to_knowledge_graph is True


def test_preserves_catalog_match_when_ontology_entity_is_missing() -> None:
    linked = create_linker(set()).link(matched_tomato())

    assert linked.ingredient is not None
    assert linked.ingredient.ontology_id == "Tomato"
    assert linked.ontology_entity is None
    assert linked.is_linked_to_knowledge_graph is False


def test_preserves_unmatched_extraction_without_links() -> None:
    linked = create_linker({"Tomato"}).link(unmatched_gochujang())

    assert linked.ingredient is None
    assert linked.ontology_entity is None
    assert linked.is_linked_to_knowledge_graph is False


def test_missing_catalog_record_raises_lookup_error() -> None:
    extracted = ExtractedIngredient(
        surface_form="unknown spice",
        normalized_text="unknown spice",
        start_char=0,
        end_char=13,
        matched_ingredient_id="unknown_spice",
        match_method=MatchMethod.ALIAS,
        match_score=0.9,
        candidate_ids=("unknown_spice",),
    )

    with pytest.raises(LookupError, match="unknown_spice"):
        create_linker(set()).link(extracted)


def test_link_all_preserves_input_order() -> None:
    paprika = ExtractedIngredient(
        surface_form="paprika",
        normalized_text="paprika",
        start_char=16,
        end_char=23,
        matched_ingredient_id="paprika",
        match_method=MatchMethod.EXACT,
        match_score=1.0,
        candidate_ids=("paprika",),
    )

    linked = create_linker({"Tomato", "Paprika"}).link_all(
        (matched_tomato(), paprika)
    )

    assert tuple(
        item.ingredient.ingredient_id
        for item in linked
        if item.ingredient is not None
    ) == ("tomato", "paprika")
