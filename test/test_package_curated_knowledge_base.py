from __future__ import annotations

import csv
import json
from pathlib import Path

from rdflib import Graph, Namespace, RDF

from scripts.package_curated_knowledge_base import (
    BundleSource,
    CuratedKnowledgeBasePackager,
)


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    """Writes one source CSV fixture for packager tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(headers)
        writer.writerows(rows)


def create_source_project(root: Path) -> None:
    """Creates all required curated source files without any image content."""
    ontology = root / "ontology" / "cuisine.rdf"
    ontology.parent.mkdir(parents=True)
    graph = Graph()
    namespace = Namespace("http://example.org/cuisine#")
    graph.add((namespace.Tomato, RDF.type, namespace.Vegetable))
    graph.serialize(ontology, format="xml")

    processed = root / "data" / "processed"
    files = {
        "ingredients.csv": (["ingredient_id"], [["tomato"]]),
        "ingredient_aliases.csv": (["alias"], [["tomatoes"]]),
        "cuisines.csv": (["cuisine_id"], [["italian"]]),
        "cuisine_aliases.csv": (["alias"], [["italian food"]]),
        "cuisine_ingredients.csv": (["cuisine_id"], [["italian"]]),
        "dishes.csv": (["dish_id"], [["tomato_dish"]]),
        "dish_ingredients.csv": (["dish_id"], [["tomato_dish"]]),
        "recipes.csv": (["recipe_id"], [["r1"]]),
        "search_index.csv": (["normalized_term"], [["tomatoes"]]),
    }
    for name, (headers, rows) in files.items():
        write_csv(processed / name, headers, rows)


def test_packages_complete_bundle_and_generates_counts(tmp_path: Path) -> None:
    """Verifies all curated files switch together with generated metadata."""
    create_source_project(tmp_path)
    destination = tmp_path / "knowledge_bases" / "curated"

    CuratedKnowledgeBasePackager(
        BundleSource(tmp_path),
        destination,
        "http://example.org/cuisine#",
    ).package()

    expected = {
        "knowledge_base.json",
        "cuisine.rdf",
        "ingredients.csv",
        "ingredient_aliases.csv",
        "cuisines.csv",
        "cuisine_aliases.csv",
        "cuisine_ingredients.csv",
        "dishes.csv",
        "dish_ingredients.csv",
        "recipes.csv",
        "search_index.csv",
    }
    assert {path.name for path in destination.iterdir()} == expected
    manifest = json.loads(
        (destination / "knowledge_base.json").read_text(encoding="utf-8")
    )
    assert manifest["counts"]["dish_count"] == 1
    assert manifest["counts"]["ingredient_count"] == 1
    assert manifest["counts"]["cuisine_count"] == 1
    assert manifest["counts"]["triple_count"] == 1


def test_fails_without_required_source_file(tmp_path: Path) -> None:
    """Verifies an incomplete source set never replaces the destination bundle."""
    create_source_project(tmp_path)
    (tmp_path / "data" / "processed" / "recipes.csv").unlink()
    destination = tmp_path / "knowledge_bases" / "curated"

    try:
        CuratedKnowledgeBasePackager(
            BundleSource(tmp_path),
            destination,
            "http://example.org/cuisine#",
        ).package()
    except FileNotFoundError as error:
        assert "recipes.csv" in str(error)
    else:
        raise AssertionError("Expected missing source file to stop packaging")
    assert not destination.exists()
