from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuisine_assistant.knowledge_base import (
    FileKnowledgeBaseRegistry,
    FileKnowledgeBaseValidator,
    JsonKnowledgeBaseManifestLoader,
    KnowledgeBasePaths,
)


def create_bundle(
    registry_root: Path,
    directory_name: str,
    knowledge_base_id: str,
    display_name: str,
    *,
    complete: bool = True,
) -> Path:
    """Creates a local bundle fixture with a valid JSON manifest."""
    bundle_root = registry_root / directory_name
    bundle_root.mkdir(parents=True)
    manifest = {
        "knowledge_base_id": knowledge_base_id,
        "name": display_name,
        "version": "1.0",
        "source": "manual",
        "language": "en",
        "ontology_base_iri": f"http://example.org/{knowledge_base_id}#",
    }
    manifest_path = bundle_root / "knowledge_base.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    if complete:
        paths = KnowledgeBasePaths.from_root(bundle_root)
        for path in paths.required_files():
            if path == manifest_path:
                continue
            path.write_text("header\n", encoding="utf-8")
    return manifest_path


def create_registry(root: Path, *, strict: bool = False) -> FileKnowledgeBaseRegistry:
    """Creates the file registry with production loader and validator components."""
    return FileKnowledgeBaseRegistry(
        root,
        JsonKnowledgeBaseManifestLoader(),
        FileKnowledgeBaseValidator(),
        strict=strict,
    )


def test_discovers_valid_bundles_and_sorts_by_name(tmp_path: Path) -> None:
    create_bundle(tmp_path, "world", "world", "World Cuisines")
    create_bundle(tmp_path, "curated", "curated", "Curated Demo")

    bundles = create_registry(tmp_path).list_available()

    assert tuple(item.metadata.knowledge_base_id for item in bundles) == (
        "curated",
        "world",
    )


def test_get_by_id_returns_only_valid_bundle(tmp_path: Path) -> None:
    create_bundle(tmp_path, "curated", "curated", "Curated Demo")
    registry = create_registry(tmp_path)

    assert registry.get_by_id("curated") is not None
    assert registry.get_by_id("missing") is None


def test_non_strict_registry_skips_incomplete_bundle(tmp_path: Path) -> None:
    create_bundle(
        tmp_path,
        "incomplete",
        "incomplete",
        "Incomplete Bundle",
        complete=False,
    )

    assert create_registry(tmp_path).list_available() == ()


def test_strict_registry_rejects_incomplete_bundle(tmp_path: Path) -> None:
    create_bundle(
        tmp_path,
        "incomplete",
        "incomplete",
        "Incomplete Bundle",
        complete=False,
    )

    with pytest.raises(ValueError, match="Incomplete knowledge base bundle"):
        create_registry(tmp_path, strict=True)


def test_rejects_duplicate_identifiers_in_strict_mode(tmp_path: Path) -> None:
    create_bundle(tmp_path, "first", "duplicate", "First Bundle")
    create_bundle(tmp_path, "second", "duplicate", "Second Bundle")

    with pytest.raises(ValueError, match="Duplicate knowledge_base_id"):
        create_registry(tmp_path, strict=True)


def test_refresh_discovers_new_bundle(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)
    assert registry.list_available() == ()

    create_bundle(tmp_path, "curated", "curated", "Curated Demo")

    refreshed = registry.refresh()
    assert len(refreshed) == 1
    assert refreshed[0].metadata.knowledge_base_id == "curated"


def test_missing_registry_root_returns_empty_collection(tmp_path: Path) -> None:
    registry = create_registry(tmp_path / "missing")

    assert registry.list_available() == ()


def test_rejects_registry_root_that_is_a_file(tmp_path: Path) -> None:
    root_file = tmp_path / "knowledge_bases"
    root_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        create_registry(root_file)
