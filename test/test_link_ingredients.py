from __future__ import annotations

from collections.abc import Iterable

from cuisine_assistant.application import LinkIngredientsUseCase
from cuisine_assistant.domain.models import (
    ExtractedIngredient,
    Ingredient,
    IngredientCategory,
    MatchMethod,
    OntologyEntityReference,
)
from cuisine_assistant.domain.ports import (
    IngredientCatalog,
    IngredientExtractor,
    KnowledgeGraphRepository,
)
from cuisine_assistant.linking import IngredientEntityLinker


class StubIngredientExtractor(IngredientExtractor):
    def extract(self, recipe_text: str) -> list[ExtractedIngredient]:
        return [
            ExtractedIngredient(
                surface_form="tomatoes",
                normalized_text="tomatoes",
                start_char=0,
                end_char=8,
                matched_ingredient_id="tomato",
                match_method=MatchMethod.ALIAS,
                match_score=1.0,
                candidate_ids=("tomato",),
            ),
            ExtractedIngredient(
                surface_form="paprika",
                normalized_text="paprika",
                start_char=10,
                end_char=17,
                matched_ingredient_id="paprika",
                match_method=MatchMethod.EXACT,
                match_score=1.0,
                candidate_ids=("paprika",),
            ),
            ExtractedIngredient(
                surface_form="gochujang",
                normalized_text="gochujang",
                start_char=19,
                end_char=28,
                matched_ingredient_id=None,
                match_method=MatchMethod.UNMATCHED,
                match_score=0.0,
            ),
        ]


class FailingExtractor(IngredientExtractor):
    def extract(self, recipe_text: str) -> list[ExtractedIngredient]:
        raise AssertionError("Extractor must not run for blank input")


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

    def get_by_id(self, ingredient_id: str) -> Ingredient | None:
        return self._by_id.get(ingredient_id)

    def get_by_ontology_id(self, ontology_id: str) -> Ingredient | None:
        return next(
            (
                item
                for item in self._ingredients
                if item.ontology_id == ontology_id
            ),
            None,
        )

    def list_all(self) -> tuple[Ingredient, ...]:
        return self._ingredients


class InMemoryKnowledgeGraphRepository(KnowledgeGraphRepository):
    def get_entity_reference(
        self,
        ontology_id: str,
    ) -> OntologyEntityReference | None:
        if ontology_id != "Tomato":
            return None
        return OntologyEntityReference(
            ontology_id="Tomato",
            iri="http://example.org/cuisine#Tomato",
            rdf_types=("Vegetable",),
            exists_in_knowledge_graph=True,
        )

    def get_entity_types(self, ontology_id: str) -> tuple[str, ...]:
        return ("Vegetable",) if ontology_id == "Tomato" else ()

    def find_dish_ids_by_ingredient_ids(
        self,
        ingredient_ontology_ids: Iterable[str],
    ) -> tuple[str, ...]:
        return ()

    def get_dish_ingredient_ids(self, dish_ontology_id: str) -> tuple[str, ...]:
        return ()

    def get_dish_cuisine_id(self, dish_ontology_id: str) -> str | None:
        return None


def create_linker() -> IngredientEntityLinker:
    return IngredientEntityLinker(
        InMemoryIngredientCatalog(),
        InMemoryKnowledgeGraphRepository(),
    )


def test_executes_extraction_and_linking_pipeline() -> None:
    use_case = LinkIngredientsUseCase(
        StubIngredientExtractor(),
        create_linker(),
    )

    result = use_case.execute("tomatoes, paprika and gochujang")

    assert len(result.linked_ingredients) == 3
    assert result.linked_count == 1
    assert result.catalog_only_count == 1
    assert result.unmatched_count == 1


def test_preserves_extraction_order() -> None:
    use_case = LinkIngredientsUseCase(
        StubIngredientExtractor(),
        create_linker(),
    )

    result = use_case.execute("example")

    assert tuple(
        item.extracted.surface_form
        for item in result.linked_ingredients
    ) == ("tomatoes", "paprika", "gochujang")


def test_blank_input_returns_empty_result_without_running_extractor() -> None:
    use_case = LinkIngredientsUseCase(
        FailingExtractor(),
        create_linker(),
    )

    result = use_case.execute("   ")

    assert result.linked_ingredients == ()
    assert result.linked_count == 0
    assert result.catalog_only_count == 0
    assert result.unmatched_count == 0
