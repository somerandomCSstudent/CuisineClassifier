from __future__ import annotations

from pathlib import Path

from cuisine_assistant.knowledge_base import (
    FileKnowledgeBaseValidator,
    KnowledgeBaseBundle,
    KnowledgeBaseMetadata,
    KnowledgeBasePaths,
)


def create_bundle(root: Path) -> KnowledgeBaseBundle:
    """Creates a complete bundle descriptor for foundation tests."""
    return KnowledgeBaseBundle(
        metadata=KnowledgeBaseMetadata(
            knowledge_base_id="curated",
            name="Curated Demo",
            version="1.0",
            source="manual",
            language="en",
            ontology_base_iri="http://example.org/cuisine#",
        ),
        paths=KnowledgeBasePaths.from_root(root),
    )


def test_standard_paths_include_the_complete_bundle(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)

    assert bundle.paths.ontology == tmp_path / "cuisine.rdf"
    assert bundle.paths.ingredients == tmp_path / "ingredients.csv"
    assert bundle.paths.cuisine_aliases == tmp_path / "cuisine_aliases.csv"
    assert bundle.paths.search_index == tmp_path / "search_index.csv"
    assert bundle.paths.manifest == tmp_path / "knowledge_base.json"
    assert len(bundle.paths.required_files()) == 11


def test_validator_reports_every_missing_file(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)

    result = FileKnowledgeBaseValidator().validate(bundle)

    assert result.is_valid is False
    assert set(result.missing_files) == set(bundle.paths.required_files())


def test_validator_accepts_only_a_complete_bundle(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    for path in bundle.paths.required_files():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "header\n", encoding="utf-8")

    result = FileKnowledgeBaseValidator().validate(bundle)

    assert result.is_valid is True
    assert result.missing_files == ()
