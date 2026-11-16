from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.knowledge_base.models import KnowledgeBaseBundle
from cuisine_assistant.knowledge_base.ports import (
    KnowledgeBaseRegistry,
    KnowledgeBaseValidator,
)


@dataclass(frozen=True)
class KnowledgeBaseSummary:
    """Provides display-ready information about one selectable knowledge base."""

    knowledge_base_id: str
    name: str
    version: str
    source: str
    language: str
    dish_count: int
    ingredient_count: int
    cuisine_count: int
    triple_count: int


@dataclass(frozen=True)
class SelectKnowledgeBaseResult:
    """Contains the validated knowledge base bundle selected by the user."""

    bundle: KnowledgeBaseBundle

    @property
    def knowledge_base_id(self) -> str:
        """Returns the stable identifier of the selected knowledge base."""
        return self.bundle.metadata.knowledge_base_id


class SelectKnowledgeBaseUseCase:
    """Lists valid knowledge bases and selects one complete bundle by identifier."""

    def __init__(
        self,
        registry: KnowledgeBaseRegistry,
        validator: KnowledgeBaseValidator,
    ) -> None:
        self._registry = registry
        self._validator = validator

    def list_available(self) -> tuple[KnowledgeBaseSummary, ...]:
        """Returns display-ready summaries for every registered valid bundle."""
        return tuple(
            self._to_summary(bundle)
            for bundle in self._registry.list_available()
        )

    def execute(self, knowledge_base_id: str) -> SelectKnowledgeBaseResult:
        """Returns a validated bundle or raises a clear selection error."""
        normalized_id = knowledge_base_id.strip()
        if not normalized_id:
            raise ValueError("knowledge_base_id must not be empty")

        bundle = self._registry.get_by_id(normalized_id)
        if bundle is None:
            available_ids = tuple(
                item.knowledge_base_id for item in self.list_available()
            )
            available_text = ", ".join(available_ids) or "none"
            raise LookupError(
                f"Unknown knowledge base '{normalized_id}'. "
                f"Available knowledge bases: {available_text}"
            )

        validation = self._validator.validate(bundle)
        if not validation.is_valid:
            missing = ", ".join(
                str(path.relative_to(bundle.paths.root))
                for path in validation.missing_files
            )
            raise ValueError(
                f"Knowledge base '{normalized_id}' is incomplete: {missing}"
            )

        return SelectKnowledgeBaseResult(bundle=bundle)

    @staticmethod
    def _to_summary(bundle: KnowledgeBaseBundle) -> KnowledgeBaseSummary:
        metadata = bundle.metadata
        return KnowledgeBaseSummary(
            knowledge_base_id=metadata.knowledge_base_id,
            name=metadata.name,
            version=metadata.version,
            source=metadata.source,
            language=metadata.language,
            dish_count=metadata.dish_count,
            ingredient_count=metadata.ingredient_count,
            cuisine_count=metadata.cuisine_count,
            triple_count=metadata.triple_count,
        )
