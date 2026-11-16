from __future__ import annotations

from abc import ABC, abstractmethod

from cuisine_assistant.knowledge_base.models import (
    KnowledgeBaseBundle,
    KnowledgeBaseValidationResult,
)


class KnowledgeBaseRegistry(ABC):
    """Lists and resolves locally available knowledge base bundles."""

    @abstractmethod
    def list_available(self) -> tuple[KnowledgeBaseBundle, ...]:
        """Returns all registered local knowledge base bundles."""
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, knowledge_base_id: str) -> KnowledgeBaseBundle | None:
        """Returns one registered bundle by its stable identifier."""
        raise NotImplementedError


class KnowledgeBaseValidator(ABC):
    """Validates bundle completeness before it becomes active."""

    @abstractmethod
    def validate(
        self,
        bundle: KnowledgeBaseBundle,
    ) -> KnowledgeBaseValidationResult:
        """Checks that all mandatory files exist and belong to the bundle."""
        raise NotImplementedError


class KnowledgeBaseBuilder(ABC):
    """Builds a local knowledge base bundle from a supported textual data source."""

    @abstractmethod
    def build(self, source: str, target_id: str) -> KnowledgeBaseBundle:
        """Creates and validates a new local bundle without downloading images."""
        raise NotImplementedError
