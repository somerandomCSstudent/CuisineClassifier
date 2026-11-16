from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuisine_assistant.knowledge_base import JsonKnowledgeBaseManifestLoader


def write_manifest(tmp_path: Path, document: dict) -> Path:
    """Writes a JSON knowledge base manifest fixture."""
    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def valid_document() -> dict:
    """Returns a valid curated knowledge base manifest."""
    return {
        "knowledge_base_id": "curated",
        "name": "Curated Demo",
        "version": "1.0",
        "source": "manual",
        "language": "en",
        "ontology_base_iri": "http://example.org/cuisine#",
        "counts": {
            "dish_count": 4,
            "ingredient_count": 13,
            "cuisine_count": 5,
            "triple_count": 120,
        },
    }


def test_loads_metadata_and_default_bundle_paths(tmp_path: Path) -> None:
    loader = JsonKnowledgeBaseManifestLoader()

    bundle = loader.load(write_manifest(tmp_path, valid_document()))

    assert bundle.metadata.knowledge_base_id == "curated"
    assert bundle.metadata.dish_count == 4
    assert bundle.paths.root == tmp_path
    assert bundle.paths.ontology == tmp_path / "cuisine.rdf"
    assert bundle.paths.manifest == tmp_path / "knowledge_base.json"


def test_supports_safe_relative_file_overrides(tmp_path: Path) -> None:
    document = valid_document()
    document["files"] = {
        "ontology": "ontology/generated.rdf",
        "recipes": "documents/recipes.csv",
        "manifest": "knowledge_base.json",
    }

    bundle = JsonKnowledgeBaseManifestLoader().load(
        write_manifest(tmp_path, document)
    )

    assert bundle.paths.ontology == tmp_path / "ontology/generated.rdf"
    assert bundle.paths.recipes == tmp_path / "documents/recipes.csv"


def test_rejects_missing_required_field(tmp_path: Path) -> None:
    document = valid_document()
    del document["ontology_base_iri"]

    with pytest.raises(ValueError, match="ontology_base_iri"):
        JsonKnowledgeBaseManifestLoader().load(
            write_manifest(tmp_path, document)
        )


def test_rejects_negative_or_non_integer_counts(tmp_path: Path) -> None:
    negative = valid_document()
    negative["counts"]["dish_count"] = -1
    with pytest.raises(ValueError, match="dish_count"):
        JsonKnowledgeBaseManifestLoader().load(
            write_manifest(tmp_path, negative)
        )

    invalid = valid_document()
    invalid["counts"]["triple_count"] = "120"
    with pytest.raises(ValueError, match="triple_count"):
        JsonKnowledgeBaseManifestLoader().load(
            write_manifest(tmp_path, invalid)
        )


def test_rejects_path_traversal_and_absolute_paths(tmp_path: Path) -> None:
    traversal = valid_document()
    traversal["files"] = {
        "ontology": "../outside.rdf",
        "manifest": "knowledge_base.json",
    }
    with pytest.raises(ValueError, match="escapes"):
        JsonKnowledgeBaseManifestLoader().load(
            write_manifest(tmp_path, traversal)
        )

    absolute = valid_document()
    absolute["files"] = {
        "ontology": str((tmp_path / "outside.rdf").resolve()),
        "manifest": "knowledge_base.json",
    }
    with pytest.raises(ValueError, match="relative"):
        JsonKnowledgeBaseManifestLoader().load(
            write_manifest(tmp_path, absolute)
        )


def test_rejects_image_related_fields_anywhere(tmp_path: Path) -> None:
    document = valid_document()
    document["import_options"] = {
        "image_urls": ["https://example.invalid/food.jpg"]
    }

    with pytest.raises(ValueError, match="Image-related"):
        JsonKnowledgeBaseManifestLoader().load(
            write_manifest(tmp_path, document)
        )


def test_rejects_manifest_mapping_to_another_file(tmp_path: Path) -> None:
    document = valid_document()
    document["files"] = {"manifest": "different.json"}

    with pytest.raises(ValueError, match="loaded manifest itself"):
        JsonKnowledgeBaseManifestLoader().load(
            write_manifest(tmp_path, document)
        )


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "knowledge_base.json"
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid knowledge base JSON"):
        JsonKnowledgeBaseManifestLoader().load(path)
