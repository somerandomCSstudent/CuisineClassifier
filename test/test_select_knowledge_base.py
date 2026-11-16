from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.application.select_knowledge_base import (
    SelectKnowledgeBaseUseCase,
)
from cuisine_assistant.knowledge_base.models import (
    KnowledgeBaseBundle,
    KnowledgeBaseMetadata,
    KnowledgeBasePaths,
    KnowledgeBaseValidationResult,
)
from cuisine_assistant.knowledge_base.ports import (
    KnowledgeBaseRegistry,
    KnowledgeBaseValidator,
)


class InMemoryKnowledgeBaseRegistry(KnowledgeBaseRegistry):
    """Provides deterministic knowledge base bundles for use-case tests."""

    def __init__(self, bundles: tuple[KnowledgeBaseBundle, ...]) -> None:
        self._bundles = bundles

    def list_available(self) -> tuple[KnowledgeBaseBundle, ...]:
        return self._bundles

    def get_by_id(self, knowledge_base_id: str) -> KnowledgeBaseBundle | None:
        return next(
            (
                bundle
                for bundle in self._bundles
                if bundle.metadata.knowledge_base_id == knowledge_base_id
            ),
            None,
        )


class ConfigurableKnowledgeBaseValidator(KnowledgeBaseValidator):
    """Returns a configured validation result for selection tests."""

    def __init__(self, result: KnowledgeBaseValidationResult) -> None:
        self._result = result

    def validate(
        self,
        bundle: KnowledgeBaseBundle,
    ) -> KnowledgeBaseValidationResult:
        return self._result


def create_bundle(
    root: Path,
    knowledge_base_id: str,
    name: str,
) -> KnowledgeBaseBundle:
    """Creates a knowledge base fixture with display metadata."""
    return KnowledgeBaseBundle(
        metadata=KnowledgeBaseMetadata(
            knowledge_base_id=knowledge_base_id,
            name=name,
            version="1.0",
            source="manual",
            language="en",
            ontology_base_iri=f"http://example.org/{knowledge_base_id}#",
            dish_count=10,
            ingredient_count=20,
            cuisine_count=5,
            triple_count=100,
        ),
        paths=KnowledgeBasePaths.from_root(root / knowledge_base_id),
    )


def valid_validator() -> ConfigurableKnowledgeBaseValidator:
    """Creates a validator that accepts the selected bundle."""
    return ConfigurableKnowledgeBaseValidator(
        KnowledgeBaseValidationResult(is_valid=True)
    )


def test_lists_display_ready_bundle_summaries(tmp_path: Path) -> None:
    bundles = (
        create_bundle(tmp_path, "curated", "Curated Demo"),
        create_bundle(tmp_path, "world", "World Cuisines"),
    )
    use_case = SelectKnowledgeBaseUseCase(
        InMemoryKnowledgeBaseRegistry(bundles),
        valid_validator(),
    )

    summaries = use_case.list_available()

    assert tuple(item.knowledge_base_id for item in summaries) == (
        "curated",
        "world",
    )
    assert summaries[0].dish_count == 10
    assert summaries[1].triple_count == 100


def test_selects_and_revalidates_complete_bundle(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path, "curated", "Curated Demo")
    use_case = SelectKnowledgeBaseUseCase(
        InMemoryKnowledgeBaseRegistry((bundle,)),
        valid_validator(),
    )

    result = use_case.execute("  curated  ")

    assert result.bundle is bundle
    assert result.knowledge_base_id == "curated"


def test_rejects_empty_identifier(tmp_path: Path) -> None:
    use_case = SelectKnowledgeBaseUseCase(
        InMemoryKnowledgeBaseRegistry(()),
        valid_validator(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        use_case.execute("   ")


def test_unknown_identifier_lists_available_choices(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path, "curated", "Curated Demo")
    use_case = SelectKnowledgeBaseUseCase(
        InMemoryKnowledgeBaseRegistry((bundle,)),
        valid_validator(),
    )

    with pytest.raises(LookupError, match="Available knowledge bases: curated"):
        use_case.execute("missing")


def test_rejects_bundle_that_became_incomplete(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path, "curated", "Curated Demo")
    missing_path = bundle.paths.recipes
    validator = ConfigurableKnowledgeBaseValidator(
        KnowledgeBaseValidationResult(
            is_valid=False,
            missing_files=(missing_path,),
        )
    )
    use_case = SelectKnowledgeBaseUseCase(
        InMemoryKnowledgeBaseRegistry((bundle,)),
        validator,
    )

    with pytest.raises(ValueError, match="recipes.csv"):
        use_case.execute("curated")
