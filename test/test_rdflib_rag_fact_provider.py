from __future__ import annotations

import csv
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS

from cuisine_assistant.rag import RdfLibRagFactProvider


def test_returns_deterministic_dish_graph_facts(tmp_path: Path) -> None:
    """Verifies CSV dish IDs are resolved to labeled RDF facts."""
    base = "http://example.org/cuisine#"
    ns = Namespace(base)
    graph = Graph()
    graph.add((ns.ChickenPaprikash, RDF.type, ns.Dish))
    graph.add((ns.ChickenPaprikash, ns.belongsToCuisine, ns.HungarianCuisine))
    graph.add((ns.ChickenPaprikash, ns.hasIngredient, ns.Paprika))
    graph.add((ns.HungarianCuisine, RDFS.label, Literal("Hungarian cuisine")))
    graph.add((ns.Paprika, RDFS.label, Literal("paprika")))
    ontology = tmp_path / "cuisine.rdf"
    graph.serialize(ontology, format="xml")
    dishes = tmp_path / "dishes.csv"
    with dishes.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("dish_id", "ontology_id"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "dish_id": "chicken_paprikash",
                "ontology_id": "ChickenPaprikash",
            }
        )

    provider = RdfLibRagFactProvider(ontology, dishes, base)
    facts = provider.facts_for_dishes(("chicken_paprikash",), 10)

    assert any(
        fact.predicate == "belongsToCuisine"
        and fact.object_value == "Hungarian cuisine"
        for fact in facts
    )
    assert any(
        fact.predicate == "hasIngredient"
        and fact.object_value == "paprika"
        for fact in facts
    )
    assert any(
        fact.predicate == "type"
        and fact.object_value == "Dish"
        for fact in facts
    )
    
def test_skips_empty_rdf_literal_values(
    tmp_path: Path,
) -> None:
    """Verifies empty RDF labels do not create invalid RAG facts."""
    base_iri = "http://example.org/cuisine#"
    namespace = Namespace(base_iri)

    graph = Graph()
    graph.add(
        (
            namespace.TestDish,
            RDF.type,
            namespace.Dish,
        )
    )
    graph.add(
        (
            namespace.TestDish,
            RDFS.label,
            Literal(""),
        )
    )

    ontology_path = tmp_path / "cuisine.rdf"
    graph.serialize(
        ontology_path,
        format="xml",
    )

    dishes_path = tmp_path / "dishes.csv"

    with dishes_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "dish_id",
                "ontology_id",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "dish_id": "test_dish",
                "ontology_id": "TestDish",
            }
        )

    provider = RdfLibRagFactProvider(
        ontology_path=ontology_path,
        dishes_csv=dishes_path,
        ontology_base_iri=base_iri,
    )

    facts = provider.facts_for_dishes(
        ("test_dish",),
        maximum_facts=10,
    )

    assert all(
        fact.object_value.strip()
        for fact in facts
    )
    assert not any(
        fact.predicate == "label"
        for fact in facts
    )
