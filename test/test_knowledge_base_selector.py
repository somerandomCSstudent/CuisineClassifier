from __future__ import annotations

from pathlib import Path

from cuisine_assistant.application import (
    KnowledgeBaseSummary,
    SelectKnowledgeBaseResult,
)
from cuisine_assistant.cli.knowledge_base_selector import KnowledgeBaseSelector
from cuisine_assistant.knowledge_base.models import (
    KnowledgeBaseBundle,
    KnowledgeBaseMetadata,
    KnowledgeBasePaths,
)


class StubSelectionUseCase:
    """Provides deterministic summaries and selections for CLI selector tests."""

    def __init__(self, summaries, result) -> None:
        self._summaries = summaries
        self._result = result
        self.received_id: str | None = None

    def list_available(self):
        return self._summaries

    def execute(self, knowledge_base_id: str):
        self.received_id = knowledge_base_id
        return self._result


def create_result(tmp_path: Path) -> SelectKnowledgeBaseResult:
    """Creates one selected bundle result for selector tests."""
    metadata = KnowledgeBaseMetadata(
        knowledge_base_id="curated",
        name="Curated Demo",
        version="1.0",
        source="manual",
        language="en",
        ontology_base_iri="http://example.org/cuisine#",
    )
    return SelectKnowledgeBaseResult(
        KnowledgeBaseBundle(
            metadata=metadata,
            paths=KnowledgeBasePaths.from_root(tmp_path / "curated"),
        )
    )


def test_selects_bundle_by_numeric_choice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verifies a displayed bundle can be selected by its menu number."""
    summary = KnowledgeBaseSummary(
        knowledge_base_id="curated",
        name="Curated Demo",
        version="1.0",
        source="manual",
        language="en",
        dish_count=4,
        ingredient_count=13,
        cuisine_count=5,
        triple_count=120,
    )
    use_case = StubSelectionUseCase((summary,), create_result(tmp_path))
    monkeypatch.setattr("builtins.input", lambda _: "1")

    result = KnowledgeBaseSelector(use_case).select()

    assert result.knowledge_base_id == "curated"
    assert use_case.received_id == "curated"


def test_rejects_empty_registry(tmp_path: Path) -> None:
    """Verifies the CLI fails clearly when no complete bundle is available."""
    use_case = StubSelectionUseCase((), create_result(tmp_path))

    try:
        KnowledgeBaseSelector(use_case).select()
    except RuntimeError as error:
        assert "No valid knowledge base bundle" in str(error)
    else:
        raise AssertionError("Expected empty registry to raise RuntimeError")
