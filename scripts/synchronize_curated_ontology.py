from __future__ import annotations

import csv
import json
from pathlib import Path

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "ontology" / "cuisine.rdf"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "knowledge_bases"
    / "curated"
    / "knowledge_base.json"
)


def read_rows(file_name: str) -> list[dict[str, str]]:
    """Loads one processed CSV catalog as dictionaries."""
    path = DATA_ROOT / file_name
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_base_iri() -> str:
    """Loads the curated ontology base IRI from the current manifest."""
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    base_iri = document["ontology_base_iri"]

    if "<a " in base_iri or "</a>" in base_iri:
        raise ValueError(
            "The curated ontology_base_iri contains HTML markup."
        )

    return base_iri


def add_individual(
    graph: Graph,
    namespace: Namespace,
    ontology_id: str,
    class_id: str,
    label: str,
) -> None:
    """Adds one named individual and its readable English label."""
    subject = namespace[ontology_id]

    graph.add((subject, RDF.type, OWL.NamedIndividual))
    graph.add((subject, RDF.type, namespace[class_id]))
    graph.add((subject, RDFS.label, Literal(label, lang="en")))


def main() -> None:
    """Synchronizes curated CSV entities and relations into the ontology."""
    base_iri = load_base_iri()
    namespace = Namespace(base_iri)

    graph = Graph()
    graph.parse(ONTOLOGY_PATH, format="xml")
    graph.bind("cuisine", namespace)

    graph.add((namespace.Ingredient, RDF.type, OWL.Class))
    graph.add((namespace.Cuisine, RDF.type, OWL.Class))
    graph.add((namespace.Dish, RDF.type, OWL.Class))
    graph.add((namespace.hasIngredient, RDF.type, OWL.ObjectProperty))
    graph.add((namespace.belongsToCuisine, RDF.type, OWL.ObjectProperty))

    ingredient_rows = read_rows("ingredients.csv")
    cuisine_rows = read_rows("cuisines.csv")
    dish_rows = read_rows("dishes.csv")
    relation_rows = read_rows("dish_ingredients.csv")

    ingredients = {
        row["ingredient_id"]: row
        for row in ingredient_rows
    }
    cuisines = {
        row["cuisine_id"]: row
        for row in cuisine_rows
    }
    dishes = {
        row["dish_id"]: row
        for row in dish_rows
    }

    for row in ingredient_rows:
        class_id = row.get("ingredient_category") or "Ingredient"

        graph.add((namespace[class_id], RDF.type, OWL.Class))
        graph.add(
            (
                namespace[class_id],
                RDFS.subClassOf,
                namespace.Ingredient,
            )
        )

        add_individual(
            graph=graph,
            namespace=namespace,
            ontology_id=row["ontology_id"],
            class_id=class_id,
            label=row["canonical_name"],
        )

    for row in cuisine_rows:
        class_id = row.get("cuisine_group") or "Cuisine"

        graph.add((namespace[class_id], RDF.type, OWL.Class))
        graph.add(
            (
                namespace[class_id],
                RDFS.subClassOf,
                namespace.Cuisine,
            )
        )

        add_individual(
            graph=graph,
            namespace=namespace,
            ontology_id=row["ontology_id"],
            class_id=class_id,
            label=row["name"],
        )

    for row in dish_rows:
        class_id = row.get("dish_category") or "Dish"

        graph.add((namespace[class_id], RDF.type, OWL.Class))
        graph.add(
            (
                namespace[class_id],
                RDFS.subClassOf,
                namespace.Dish,
            )
        )

        add_individual(
            graph=graph,
            namespace=namespace,
            ontology_id=row["ontology_id"],
            class_id=class_id,
            label=row["name"],
        )

        cuisine = cuisines.get(row["cuisine_id"])
        if cuisine is not None:
            graph.add(
                (
                    namespace[row["ontology_id"]],
                    namespace.belongsToCuisine,
                    namespace[cuisine["ontology_id"]],
                )
            )

    for row in relation_rows:
        dish = dishes.get(row["dish_id"])
        ingredient = ingredients.get(row["ingredient_id"])

        if dish is None or ingredient is None:
            continue

        graph.add(
            (
                namespace[dish["ontology_id"]],
                namespace.hasIngredient,
                namespace[ingredient["ontology_id"]],
            )
        )

    graph.serialize(ONTOLOGY_PATH, format="xml")

    print(f"Curated ontology synchronized: {ONTOLOGY_PATH}")
    print(f"RDF triples: {len(graph)}")


if __name__ == "__main__":
    main()