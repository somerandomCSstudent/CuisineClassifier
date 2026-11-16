from __future__ import annotations

import csv
import json
from pathlib import Path

from rdflib import Graph, Namespace, OWL, RDF

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.knowledge_base import JsonKnowledgeBaseManifestLoader


BASE_IRI = "http://example.org/cuisine#"
CUISINE = Namespace(BASE_IRI)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    """Writes one bundle CSV fixture with a deterministic schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_complete_bundle(root: Path):
    """Creates a complete bundle used by bootstrap integration tests."""
    root.mkdir(parents=True)
    manifest = {
        "knowledge_base_id": "test_bundle",
        "name": "Test Bundle",
        "version": "1.0",
        "source": "test",
        "language": "en",
        "ontology_base_iri": BASE_IRI,
    }
    (root / "knowledge_base.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    write_csv(
        root / "ingredients.csv",
        [
            "ingredient_id",
            "canonical_name",
            "ingredient_category",
            "description",
            "ontology_id",
        ],
        [
            {
                "ingredient_id": "tomato",
                "canonical_name": "tomato",
                "ingredient_category": "Vegetable",
                "description": "",
                "ontology_id": "Tomato",
            }
        ],
    )
    write_csv(
        root / "ingredient_aliases.csv",
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
            }
        ],
    )
    write_csv(
        root / "cuisines.csv",
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
            }
        ],
    )
    write_csv(
        root / "cuisine_aliases.csv",
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
                "alias": "italian food",
                "alias_type": "phrase",
                "language": "en",
                "match_priority": 95,
                "verified": "true",
            }
        ],
    )
    write_csv(
        root / "cuisine_ingredients.csv",
        ["cuisine_id", "ingredient_id", "typicality_score"],
        [
            {
                "cuisine_id": "italian",
                "ingredient_id": "tomato",
                "typicality_score": 0.9,
            }
        ],
    )
    write_csv(
        root / "dishes.csv",
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
            }
        ],
    )
    write_csv(
        root / "dish_ingredients.csv",
        ["dish_id", "ingredient_id"],
        [{"dish_id": "tomato_dish", "ingredient_id": "tomato"}],
    )
    write_csv(
        root / "recipes.csv",
        ["recipe_id", "dish_id", "raw_text", "cuisine_id"],
        [
            {
                "recipe_id": "r1",
                "dish_id": "tomato_dish",
                "raw_text": "Tomatoes",
                "cuisine_id": "italian",
            }
        ],
    )
    write_csv(
        root / "search_index.csv",
        ["normalized_term", "entity_type", "entity_id"],
        [
            {
                "normalized_term": "tomatoes",
                "entity_type": "ingredient",
                "entity_id": "tomato",
            }
        ],
    )

    graph = Graph()
    graph.add((CUISINE.Vegetable, RDF.type, OWL.Class))
    graph.add((CUISINE.Tomato, RDF.type, CUISINE.Vegetable))
    graph.add((CUISINE.TomatoDish, RDF.type, CUISINE.MainCourse))
    graph.add((CUISINE.TomatoDish, CUISINE.hasIngredient, CUISINE.Tomato))
    graph.add(
        (
            CUISINE.TomatoDish,
            CUISINE.belongsToCuisine,
            CUISINE.ItalianCuisine,
        )
    )
    graph.serialize(root / "cuisine.rdf", format="xml")

    return JsonKnowledgeBaseManifestLoader().load(
        root / "knowledge_base.json"
    )


def test_builds_all_use_cases_from_selected_bundle(tmp_path: Path) -> None:
    """Verifies linking, classification, and recommendation use one bundle."""
    bundle = create_complete_bundle(tmp_path / "bundle")

    container = build_application_container(bundle)
    result = container.recommend_dishes.execute(
        "I have tomatoes and I would like Italian food."
    )

    assert container.knowledge_base is bundle
    assert result.recommendation.best_recommendation is not None
    assert (
        result.recommendation.best_recommendation.dish.dish_id
        == "tomato_dish"
    )


def test_settings_validate_scoring_configuration() -> None:
    """Verifies bundle-independent scoring settings reject invalid values."""
    try:
        ApplicationSettings(maximum_recommendations=0)
    except ValueError as error:
        assert "maximum_recommendations" in str(error)
    else:
        raise AssertionError("Expected invalid settings to raise ValueError")

def test_settings_keep_top_three_results_by_default() -> None:
    """Verifies the default result limits remain three."""
    settings = ApplicationSettings()

    assert settings.maximum_predictions == 3
    assert settings.maximum_recommendations == 3
