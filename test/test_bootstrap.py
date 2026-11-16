from __future__ import annotations

import csv
from pathlib import Path

from rdflib import Graph, Namespace, OWL, RDF

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)


BASE_IRI = "http://example.org/cuisine#"
CUISINE = Namespace(BASE_IRI)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_project_files(project_root: Path) -> None:
    write_csv(
        project_root / "data/processed/ingredients.csv",
        [
            "ingredient_id",
            "canonical_name",
            "ingredient_category",
            "ontology_id",
            "description",
        ],
        [
            {
                "ingredient_id": "tomato",
                "canonical_name": "tomato",
                "ingredient_category": "Vegetable",
                "ontology_id": "Tomato",
                "description": "",
            },
            {
                "ingredient_id": "paprika",
                "canonical_name": "paprika",
                "ingredient_category": "Spice",
                "ontology_id": "Paprika",
                "description": "",
            },
        ],
    )
    write_csv(
        project_root / "data/processed/ingredient_aliases.csv",
        [
            "alias_id",
            "ingredient_id",
            "alias",
            "alias_type",
            "language",
            "match_priority",
            "verified",
        ],
        [
            {
                "alias_id": "tomato_1",
                "ingredient_id": "tomato",
                "alias": "tomatoes",
                "alias_type": "plural",
                "language": "en",
                "match_priority": 95,
                "verified": "true",
            },
            {
                "alias_id": "paprika_1",
                "ingredient_id": "paprika",
                "alias": "paprika",
                "alias_type": "canonical",
                "language": "en",
                "match_priority": 100,
                "verified": "true",
            },
        ],
    )
    write_csv(
    project_root / "data/processed/cuisines.csv",
    [
        "cuisine_id",
        "name",
        "cuisine_group",
        "origin_place_id",
        "description",
        "ontology_id",
    ],
    [
            {
                "cuisine_id": "italian",
                "name": "Italian cuisine",
                "cuisine_group": "EuropeanCuisine",
                "origin_place_id": "italy",
                "description": "",
                "ontology_id": "ItalianCuisine",
            },
        ],
    )
    write_csv(
    project_root / "data/processed/cuisine_ingredients.csv",
    [
        "cuisine_id",
        "ingredient_id",
        "typicality_score",
    ],
    [
            {
                "cuisine_id": "italian",
                "ingredient_id": "tomato",
                "typicality_score": "0.90",
            },
        ],
    )
    write_csv(
    project_root / "data/processed/dishes.csv",
    [
        "dish_id",
        "name",
        "dish_category",
        "cuisine_id",
        "description",
        "ontology_id",
    ],
    [
            {
                "dish_id": "tomato_dish",
                "name": "Tomato Dish",
                "dish_category": "MainCourse",
                "cuisine_id": "italian",
                "description": "",
                "ontology_id": "TomatoDish",
            },
        ],
    )
    write_csv(
    project_root / "data/processed/cuisine_aliases.csv",
    [
        "alias_id",
        "cuisine_id",
        "alias",
        "alias_type",
        "language",
        "match_priority",
        "verified",
    ],
    [
        {
            "alias_id": "italian_1",
            "cuisine_id": "italian",
            "alias": "italian",
            "alias_type": "canonical",
            "language": "en",
            "match_priority": "100",
            "verified": "true",
        },
            {
                "alias_id": "italian_2",
                "cuisine_id": "italian",
                "alias": "italian food",
                "alias_type": "phrase",
                "language": "en",
                "match_priority": "95",
                "verified": "true",
            },
        ],
    )

    graph = Graph()
    graph.add((CUISINE.Vegetable, RDF.type, OWL.Class))
    graph.add((CUISINE.Spice, RDF.type, OWL.Class))
    graph.add((CUISINE.Tomato, RDF.type, CUISINE.Vegetable))
    graph.add((CUISINE.TomatoDish, RDF.type, CUISINE.MainCourse))
    graph.add((CUISINE.TomatoDish, CUISINE.hasIngredient, CUISINE.Tomato))
    graph.add((CUISINE.TomatoDish, CUISINE.belongsToCuisine, CUISINE.ItalianCuisine))
    ontology_path = project_root / "ontology/cuisine.rdf"
    ontology_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(ontology_path, format="xml")


def test_bootstrap_builds_complete_linking_pipeline(tmp_path: Path) -> None:
    create_project_files(tmp_path)
    container = build_application_container(
        ApplicationSettings(
            project_root=tmp_path,
            ontology_base_iri=BASE_IRI,
        )
    )

    result = container.link_ingredients.execute("Tomatoes and paprika")

    assert result.linked_count == 1
    assert result.catalog_only_count == 1
    assert result.unmatched_count == 0
    assert tuple(
        item.ingredient.ingredient_id
        for item in result.linked_ingredients
        if item.ingredient is not None
    ) == ("tomato", "paprika")


def test_settings_resolve_paths_from_project_root(
    tmp_path: Path,
) -> None:
    """Verifies all knowledge-base paths are resolved from the project root."""
    settings = ApplicationSettings(
        project_root=tmp_path,
        ontology_base_iri=BASE_IRI,
    )

    assert settings.ontology_path == (
        tmp_path / "ontology/cuisine.rdf"
    )
    assert settings.ingredients_path == (
        tmp_path / "data/processed/ingredients.csv"
    )
    assert settings.aliases_path == (
        tmp_path / "data/processed/ingredient_aliases.csv"
    )
    assert settings.cuisines_path == (
        tmp_path / "data/processed/cuisines.csv"
    )
    assert settings.cuisine_ingredients_path == (
        tmp_path / "data/processed/cuisine_ingredients.csv"
    )
    assert settings.dishes_path == (
        tmp_path / "data/processed/dishes.csv"
    )
    assert settings.cuisine_aliases_path == (
        tmp_path / "data/processed/cuisine_aliases.csv"
    )
