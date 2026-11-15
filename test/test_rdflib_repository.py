from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph, Namespace, OWL, RDF

from cuisine_assistant.knowledge_graph.rdflib_repository import (
    RdfLibKnowledgeGraphRepository,
)


BASE_IRI = "http://example.org/cuisine#"
CUISINE = Namespace(BASE_IRI)


def create_ontology(tmp_path: Path) -> Path:
    graph = Graph()
    graph.add((CUISINE.Vegetable, RDF.type, OWL.Class))
    graph.add((CUISINE.DairyProduct, RDF.type, OWL.Class))
    graph.add((CUISINE.MainCourse, RDF.type, OWL.Class))
    graph.add((CUISINE.EuropeanCuisine, RDF.type, OWL.Class))
    graph.add((CUISINE.hasIngredient, RDF.type, OWL.ObjectProperty))
    graph.add((CUISINE.belongsToCuisine, RDF.type, OWL.ObjectProperty))
    graph.add((CUISINE.Tomato, RDF.type, CUISINE.Vegetable))
    graph.add((CUISINE.Mozzarella, RDF.type, CUISINE.DairyProduct))
    graph.add((CUISINE.Basil, RDF.type, CUISINE.Herb))
    graph.add((CUISINE.OliveOil, RDF.type, CUISINE.Oil))
    graph.add((CUISINE.ItalianCuisine, RDF.type, CUISINE.EuropeanCuisine))
    graph.add((CUISINE.MargheritaPizza, RDF.type, CUISINE.MainCourse))

    for ingredient in (
        CUISINE.Tomato,
        CUISINE.Mozzarella,
        CUISINE.Basil,
        CUISINE.OliveOil,
    ):
        graph.add((CUISINE.MargheritaPizza, CUISINE.hasIngredient, ingredient))
    graph.add(
        (
            CUISINE.MargheritaPizza,
            CUISINE.belongsToCuisine,
            CUISINE.ItalianCuisine,
        )
    )

    ontology_path = tmp_path / "cuisine.owl"
    graph.serialize(ontology_path, format="xml")
    return ontology_path


def create_repository(tmp_path: Path) -> RdfLibKnowledgeGraphRepository:
    return RdfLibKnowledgeGraphRepository(create_ontology(tmp_path), BASE_IRI)


def test_returns_entity_reference_and_rdf_types(tmp_path: Path) -> None:
    reference = create_repository(tmp_path).get_entity_reference("Tomato")
    assert reference is not None
    assert reference.iri == f"{BASE_IRI}Tomato"
    assert reference.rdf_types == ("Vegetable",)


def test_unknown_entity_returns_none(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    assert repository.get_entity_reference("Unknown") is None
    assert repository.get_entity_types("Unknown") == ()


def test_finds_dishes_by_any_requested_ingredient(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    assert repository.find_dish_ids_by_ingredient_ids(
        ("Tomato", "Mozzarella")
    ) == ("MargheritaPizza",)


def test_returns_dish_relations(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    assert repository.get_dish_ingredient_ids("MargheritaPizza") == (
        "Basil",
        "Mozzarella",
        "OliveOil",
        "Tomato",
    )
    assert repository.get_dish_cuisine_id("MargheritaPizza") == "ItalianCuisine"


def test_missing_ontology_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RdfLibKnowledgeGraphRepository(tmp_path / "missing.owl", BASE_IRI)


def test_multiple_cuisine_relations_are_rejected(tmp_path: Path) -> None:
    ontology_path = create_ontology(tmp_path)
    graph = Graph().parse(ontology_path, format="xml")
    graph.add(
        (
            CUISINE.MargheritaPizza,
            CUISINE.belongsToCuisine,
            CUISINE.OtherCuisine,
        )
    )
    graph.serialize(ontology_path, format="xml")
    repository = RdfLibKnowledgeGraphRepository(ontology_path, BASE_IRI)

    with pytest.raises(ValueError, match="multiple cuisine"):
        repository.get_dish_cuisine_id("MargheritaPizza")
