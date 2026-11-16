from __future__ import annotations

import csv
import json
from pathlib import Path

from cuisine_assistant.importing import (
    BundleBuildConfiguration,
    ImportedCuisine,
    ImportedDishRecord,
    ImportedIngredientMention,
    ImportedKnowledgeBaseBundleBuilder,
    ImportSourceType,
    TextualImportBatch,
    TextualImportOptions,
    TextualImportSource,
    WorldCuisinesBundleBuilder,
)


def create_batch() -> TextualImportBatch:
    """Creates normalized World Cuisines records for bundle-builder tests."""
    cuisine = ImportedCuisine(
        cuisine_id="indonesian",
        name="Indonesian cuisine",
        ontology_id="IndonesianCuisine",
        aliases=("Indonesian", "Indonesian food"),
        origin_place_id="indonesia",
    )
    record = ImportedDishRecord(
        external_id="rawon__indonesian",
        name="Rawon",
        description="Beef soup with garlic and ginger.",
        aliases=("Nasi Rawon",),
        categories=("soup",),
        cuisines=(cuisine,),
        places=(),
        ingredients=(
            ImportedIngredientMention("beef", "beef", 1.0),
            ImportedIngredientMention("garlic", "garlic", 0.95),
            ImportedIngredientMention("ginger", "ginger", 0.95),
        ),
        source_url="https://huggingface.co/datasets/worldcuisines/food-kb",
        source_license="CC BY-SA 4.0",
    )
    return TextualImportBatch(
        source=TextualImportSource(
            ImportSourceType.HUGGING_FACE,
            "worldcuisines/food-kb",
            "default",
            split="main",
            license_name="CC BY-SA 4.0",
        ),
        options=TextualImportOptions(maximum_records=500),
        records=(record,),
    )


def test_builds_complete_image_free_bundle(tmp_path: Path) -> None:
    """Verifies all required CSV, RDF, index, and manifest files are generated."""
    destination = tmp_path / "knowledge_bases" / "world_cuisines"
    WorldCuisinesBundleBuilder(
        ImportedKnowledgeBaseBundleBuilder()
    ).build(
        create_batch(),
        destination,
        "http://example.org/world-cuisines#",
    )

    expected = {
        "knowledge_base.json", "cuisine.rdf", "ingredients.csv",
        "ingredient_aliases.csv", "cuisines.csv", "cuisine_aliases.csv",
        "cuisine_ingredients.csv", "dishes.csv", "dish_ingredients.csv",
        "recipes.csv", "search_index.csv",
    }
    assert {path.name for path in destination.iterdir()} == expected
    assert not any(
        path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        for path in destination.rglob("*")
    )

    manifest = json.loads(
        (destination / "knowledge_base.json").read_text(encoding="utf-8")
    )
    assert manifest["knowledge_base_id"] == "world_cuisines"
    assert manifest["counts"]["dish_count"] == 1
    assert manifest["counts"]["ingredient_count"] == 3
    assert manifest["counts"]["cuisine_count"] == 1
    assert manifest["counts"]["triple_count"] > 0
    assert manifest["import"]["images_included"] is False


def test_writes_consistent_csv_relations_and_rdf_links(tmp_path: Path) -> None:
    """Verifies generated identifiers agree across catalogs and RDF/XML."""
    destination = tmp_path / "world"
    ImportedKnowledgeBaseBundleBuilder().build(
        create_batch(),
        BundleBuildConfiguration(
            knowledge_base_id="world",
            name="World",
            version="1.0",
            ontology_base_iri="http://example.org/world#",
            destination=destination,
            source_name="worldcuisines/food-kb",
        ),
    )

    with (destination / "dish_ingredients.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        relations = list(csv.DictReader(source))
    assert {item["ingredient_id"] for item in relations} == {
        "beef", "garlic", "ginger"
    }

    rdf_text = (destination / "cuisine.rdf").read_text(encoding="utf-8")
    assert "RawonDish" in rdf_text
    assert "IndonesianCuisine" in rdf_text
    assert "hasIngredient" in rdf_text
    assert "belongsToCuisine" in rdf_text


def test_rebuild_replaces_previous_bundle_atomically(tmp_path: Path) -> None:
    """Verifies a successful rebuild replaces old bundle files as one unit."""
    destination = tmp_path / "world"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("old", encoding="utf-8")

    WorldCuisinesBundleBuilder(
        ImportedKnowledgeBaseBundleBuilder()
    ).build(
        create_batch(),
        destination,
        "http://example.org/world#",
    )

    assert not (destination / "old.txt").exists()
    assert (destination / "knowledge_base.json").is_file()
