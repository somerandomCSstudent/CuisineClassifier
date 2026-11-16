from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from cuisine_assistant.knowledge_base.hybrid_builder import HybridKnowledgeBaseBuilder


def write_csv(path: Path, headers, rows) -> None:
    """Writes one bundle CSV fixture."""
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(headers)
        writer.writerows(rows)


def create_bundle(root: Path, identifier: str, curated_name: str):
    """Creates the minimal complete bundle consumed by the hybrid builder."""
    root.mkdir(parents=True)
    write_csv(root / "ingredients.csv", ("ingredient_id", "canonical_name", "ontology_id"), (("tomato", curated_name, "Tomato"),))
    write_csv(root / "ingredient_aliases.csv", ("alias_id", "ingredient_id", "alias"), ((f"{identifier}_a", "tomato", "tomatoes"),))
    write_csv(root / "cuisines.csv", ("cuisine_id", "name", "ontology_id"), (("italian", "Italian cuisine", "ItalianCuisine"),))
    write_csv(root / "cuisine_aliases.csv", ("alias_id", "cuisine_id", "alias"), ((f"{identifier}_c", "italian", "Italian food"),))
    write_csv(root / "cuisine_ingredients.csv", ("cuisine_id", "ingredient_id", "typicality_score"), (("italian", "tomato", "1.0"),))
    write_csv(root / "dishes.csv", ("dish_id", "name", "cuisine_id", "ontology_id"), ((f"{identifier}_dish", f"{identifier} Dish", "italian", f"{identifier.title()}Dish"),))
    write_csv(root / "dish_ingredients.csv", ("dish_id", "ingredient_id"), ((f"{identifier}_dish", "tomato"),))
    write_csv(root / "recipes.csv", ("recipe_id", "dish_id", "cuisine_id"), ((f"{identifier}_r", f"{identifier}_dish", "italian"),))
    write_csv(root / "search_index.csv", ("normalized_term", "entity_type", "entity_id"), (("tomato", "ingredient", "tomato"),))
    (root / "cuisine.rdf").write_text("<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'/>", encoding="utf-8")
    (root / "knowledge_base.json").write_text("{}", encoding="utf-8")
    paths = SimpleNamespace(root=root)
    return SimpleNamespace(metadata=SimpleNamespace(knowledge_base_id=identifier), paths=paths)


def test_curated_records_override_imported_records(tmp_path: Path) -> None:
    """Verifies stable IDs use curated values while imported new records remain."""
    curated = create_bundle(tmp_path / "curated", "curated", "curated tomato")
    imported = create_bundle(tmp_path / "world", "world", "imported tomato")
    destination = tmp_path / "hybrid"

    HybridKnowledgeBaseBuilder().build(
        curated,
        imported,
        destination,
        ontology_base_iri="urn:cuisine:hybrid#",
    )

    with (destination / "ingredients.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["canonical_name"] == "curated tomato"
    with (destination / "dishes.csv").open(encoding="utf-8", newline="") as source:
        dishes = list(csv.DictReader(source))
    assert {row["dish_id"] for row in dishes} == {"curated_dish", "world_dish"}


def test_creates_complete_image_free_hybrid_bundle(tmp_path: Path) -> None:
    """Verifies the full bundle and merge metadata are generated."""
    curated = create_bundle(tmp_path / "curated", "curated", "tomato")
    imported = create_bundle(tmp_path / "world", "world_cuisines", "tomato")
    destination = tmp_path / "hybrid"
    HybridKnowledgeBaseBuilder().build(
        curated,
        imported,
        destination,
        ontology_base_iri="urn:cuisine:hybrid#",
    )
    assert len(tuple(destination.iterdir())) == 11
    manifest = json.loads((destination / "knowledge_base.json").read_text(encoding="utf-8"))
    assert manifest["knowledge_base_id"] == "hybrid"
    assert manifest["merge"]["conflict_policy"] == "curated_wins"
    assert manifest["merge"]["images_included"] is False
    assert manifest["counts"]["dish_count"] == 2
