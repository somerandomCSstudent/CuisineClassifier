from __future__ import annotations

import csv
import json
from pathlib import Path

from rdflib import Graph, Namespace, RDF

from cuisine_assistant.knowledge_base import (
    JsonKnowledgeBaseManifestLoader,
    KnowledgeBaseConsistencyValidator,
)


BASE_IRI = "http://example.org/test#"
NS = Namespace(BASE_IRI)


def write_csv(path: Path, headers: tuple[str, ...], rows: tuple[tuple, ...]) -> None:
    """Writes one deterministic bundle CSV fixture."""
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(headers)
        writer.writerows(rows)


def create_valid_bundle(root: Path):
    """Creates a cross-file consistent bundle for validator tests."""
    root.mkdir(parents=True)
    write_csv(
        root / "ingredients.csv",
        ("ingredient_id", "canonical_name", "ontology_id"),
        (("tomato", "tomato", "Tomato"),),
    )
    write_csv(
        root / "ingredient_aliases.csv",
        ("alias_id", "ingredient_id", "alias"),
        (("a1", "tomato", "tomatoes"),),
    )
    write_csv(
        root / "cuisines.csv",
        ("cuisine_id", "name", "ontology_id"),
        (("italian", "Italian cuisine", "ItalianCuisine"),),
    )
    write_csv(
        root / "cuisine_aliases.csv",
        ("alias_id", "cuisine_id", "alias"),
        (("c1", "italian", "Italian food"),),
    )
    write_csv(
        root / "cuisine_ingredients.csv",
        ("cuisine_id", "ingredient_id", "typicality_score"),
        (("italian", "tomato", "1.0"),),
    )
    write_csv(
        root / "dishes.csv",
        ("dish_id", "name", "cuisine_id", "ontology_id"),
        (("tomato_dish", "Tomato Dish", "italian", "TomatoDish"),),
    )
    write_csv(
        root / "dish_ingredients.csv",
        ("dish_id", "ingredient_id"),
        (("tomato_dish", "tomato"),),
    )
    write_csv(
        root / "recipes.csv",
        ("recipe_id", "dish_id", "cuisine_id"),
        (("r1", "tomato_dish", "italian"),),
    )
    write_csv(
        root / "search_index.csv",
        ("normalized_term", "entity_type", "entity_id"),
        (("tomato", "ingredient", "tomato"),),
    )

    graph = Graph()
    graph.add((NS.Tomato, RDF.type, NS.Ingredient))
    graph.add((NS.ItalianCuisine, RDF.type, NS.Cuisine))
    graph.add((NS.TomatoDish, RDF.type, NS.Dish))
    graph.serialize(root / "cuisine.rdf", format="xml")

    manifest = {
        "knowledge_base_id": "test",
        "name": "Test",
        "version": "1.0",
        "source": "test",
        "language": "en",
        "ontology_base_iri": BASE_IRI,
        "counts": {
            "dish_count": 1,
            "ingredient_count": 1,
            "cuisine_count": 1,
            "triple_count": len(graph),
        },
    }
    (root / "knowledge_base.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return JsonKnowledgeBaseManifestLoader().load(root / "knowledge_base.json")


def test_accepts_cross_file_consistent_bundle(tmp_path: Path) -> None:
    """Verifies valid foreign keys, RDF entities, and counts produce no issue."""
    bundle = create_valid_bundle(tmp_path / "bundle")

    report = KnowledgeBaseConsistencyValidator().validate(bundle)

    assert report.is_valid is True
    assert report.issues == ()


def test_reports_unknown_foreign_key(tmp_path: Path) -> None:
    """Verifies an orphan dish-ingredient relation is detected."""
    bundle = create_valid_bundle(tmp_path / "bundle")
    write_csv(
        bundle.paths.dish_ingredients,
        ("dish_id", "ingredient_id"),
        (("tomato_dish", "missing"),),
    )

    report = KnowledgeBaseConsistencyValidator().validate(bundle)

    assert report.is_valid is False
    assert any(issue.code == "UNKNOWN_FOREIGN_KEY" for issue in report.issues)


def test_reports_missing_rdf_entity(tmp_path: Path) -> None:
    """Verifies CSV ontology IDs must exist as RDF subjects."""
    bundle = create_valid_bundle(tmp_path / "bundle")
    write_csv(
        bundle.paths.ingredients,
        ("ingredient_id", "canonical_name", "ontology_id"),
        (("garlic", "garlic", "Garlic"),),
    )

    report = KnowledgeBaseConsistencyValidator().validate(bundle)

    assert any(issue.code == "MISSING_RDF_ENTITY" for issue in report.issues)


def test_reports_manifest_count_mismatch(tmp_path: Path) -> None:
    """Verifies generated manifest counts must equal actual bundle contents."""
    bundle = create_valid_bundle(tmp_path / "bundle")
    manifest_path = bundle.paths.manifest
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["counts"]["triple_count"] = 999
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    bundle = JsonKnowledgeBaseManifestLoader().load(manifest_path)

    report = KnowledgeBaseConsistencyValidator().validate(bundle)

    assert any(
        issue.code == "MANIFEST_COUNT_MISMATCH"
        for issue in report.issues
    )
