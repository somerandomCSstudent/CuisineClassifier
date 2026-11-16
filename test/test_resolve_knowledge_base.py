from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.application.resolve_knowledge_base import (
    ResolveKnowledgeBaseUseCase,
)
from cuisine_assistant.application.select_knowledge_base import (
    KnowledgeBaseSummary,
    SelectKnowledgeBaseResult,
)
from cuisine_assistant.knowledge_base.models import (
    KnowledgeBaseBundle,
    KnowledgeBaseMetadata,
    KnowledgeBasePaths,
)


class StubSelectionUseCase:
    """Provides selectable bundles for non-interactive resolution tests."""

    def __init__(self, bundles: tuple[KnowledgeBaseBundle, ...]) -> None:
        self._bundles = bundles

    def list_available(self):
        return tuple(
            KnowledgeBaseSummary(
                knowledge_base_id=item.metadata.knowledge_base_id,
                name=item.metadata.name,
                version=item.metadata.version,
                source=item.metadata.source,
                language=item.metadata.language,
                dish_count=item.metadata.dish_count,
                ingredient_count=item.metadata.ingredient_count,
                cuisine_count=item.metadata.cuisine_count,
                triple_count=item.metadata.triple_count,
            )
            for item in self._bundles
        )

    def execute(self, knowledge_base_id: str):
        bundle = next(
            item
            for item in self._bundles
            if item.metadata.knowledge_base_id == knowledge_base_id
        )
        return SelectKnowledgeBaseResult(bundle)


def create_bundle(root: Path, identifier: str) -> KnowledgeBaseBundle:
    """Creates a bundle fixture for resolver tests."""
    return KnowledgeBaseBundle(
        KnowledgeBaseMetadata(
            knowledge_base_id=identifier,
            name=identifier.title(),
            version="1.0",
            source="test",
            language="en",
            ontology_base_iri=f"http://example.org/{identifier}#",
        ),
        KnowledgeBasePaths.from_root(root / identifier),
    )


def test_resolves_explicit_bundle_identifier(tmp_path: Path) -> None:
    curated = create_bundle(tmp_path, "curated")
    world = create_bundle(tmp_path, "world")
    resolver = ResolveKnowledgeBaseUseCase(
        StubSelectionUseCase((curated, world))
    )

    result = resolver.execute("world")

    assert result.bundle is world


def test_uses_only_available_bundle_as_default(tmp_path: Path) -> None:
    curated = create_bundle(tmp_path, "curated")
    resolver = ResolveKnowledgeBaseUseCase(
        StubSelectionUseCase((curated,))
    )

    assert resolver.execute(None).bundle is curated


def test_requires_explicit_id_when_multiple_bundles_exist(tmp_path: Path) -> None:
    resolver = ResolveKnowledgeBaseUseCase(
        StubSelectionUseCase(
            (
                create_bundle(tmp_path, "curated"),
                create_bundle(tmp_path, "world"),
            )
        )
    )

    with pytest.raises(ValueError, match="Specify one of"):
        resolver.execute(None)
