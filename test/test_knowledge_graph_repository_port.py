from __future__ import annotations

from collections.abc import Iterable

from cuisine_assistant.domain.models import OntologyEntityReference
from cuisine_assistant.domain.ports import KnowledgeGraphRepository


class InMemoryKnowledgeGraphRepository(KnowledgeGraphRepository):
    def __init__(self) -> None:
        self._base_iri = "http://example.org/cuisine#"
        self._types = {
            "Tomato": ("Vegetable",),
            "Mozzarella": ("DairyProduct",),
            "MargheritaPizza": ("MainCourse", "Dish"),
            "ItalianCuisine": ("EuropeanCuisine",),
        }
        self._dish_ingredients = {
            "MargheritaPizza": ("Tomato", "Mozzarella", "Basil", "OliveOil"),
        }
        self._dish_cuisines = {
            "MargheritaPizza": "ItalianCuisine",
        }

    def get_entity_reference(
        self,
        ontology_id: str,
    ) -> OntologyEntityReference | None:
        types = self._types.get(ontology_id)
        if types is None:
            return None
        return OntologyEntityReference(
            ontology_id=ontology_id,
            iri=f"{self._base_iri}{ontology_id}",
            rdf_types=types,
            exists_in_knowledge_graph=True,
        )

    def get_entity_types(self, ontology_id: str) -> tuple[str, ...]:
        return self._types.get(ontology_id, ())

    def find_dish_ids_by_ingredient_ids(
        self,
        ingredient_ontology_ids: Iterable[str],
    ) -> tuple[str, ...]:
        requested = set(ingredient_ontology_ids)
        if not requested:
            return ()
        matches = [
            dish_id
            for dish_id, ingredient_ids in self._dish_ingredients.items()
            if requested.intersection(ingredient_ids)
        ]
        return tuple(sorted(matches))

    def get_dish_ingredient_ids(self, dish_ontology_id: str) -> tuple[str, ...]:
        return self._dish_ingredients.get(dish_ontology_id, ())

    def get_dish_cuisine_id(self, dish_ontology_id: str) -> str | None:
        return self._dish_cuisines.get(dish_ontology_id)


def create_repository() -> KnowledgeGraphRepository:
    return InMemoryKnowledgeGraphRepository()


def test_get_entity_reference_returns_technology_independent_reference() -> None:
    reference = create_repository().get_entity_reference("Tomato")

    assert reference is not None
    assert reference.iri == "http://example.org/cuisine#Tomato"
    assert reference.rdf_types == ("Vegetable",)
    assert reference.exists_in_knowledge_graph is True


def test_unknown_entity_returns_none_and_empty_types() -> None:
    repository = create_repository()

    assert repository.get_entity_reference("Unknown") is None
    assert repository.get_entity_types("Unknown") == ()


def test_find_dishes_by_ingredient_ids_returns_matching_dishes() -> None:
    dishes = create_repository().find_dish_ids_by_ingredient_ids(
        ("Tomato", "Mozzarella")
    )

    assert dishes == ("MargheritaPizza",)


def test_empty_ingredient_query_returns_no_dishes() -> None:
    assert create_repository().find_dish_ids_by_ingredient_ids(()) == ()


def test_get_dish_relations_returns_graph_identifiers() -> None:
    repository = create_repository()

    assert repository.get_dish_ingredient_ids("MargheritaPizza") == (
        "Tomato",
        "Mozzarella",
        "Basil",
        "OliveOil",
    )
    assert repository.get_dish_cuisine_id("MargheritaPizza") == "ItalianCuisine"


def test_unknown_dish_returns_empty_relations() -> None:
    repository = create_repository()

    assert repository.get_dish_ingredient_ids("UnknownDish") == ()
    assert repository.get_dish_cuisine_id("UnknownDish") is None
