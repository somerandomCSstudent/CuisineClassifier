from __future__ import annotations

from abc import ABC, abstractmethod

from cuisine_assistant.rag.models import RagGraphFact


class RagKnowledgeGraphFactProvider(ABC):
    """Provides bounded human-readable RDF facts for RAG grounding."""

    @abstractmethod
    def facts_for_dishes(
        self,
        dish_ids: tuple[str, ...],
        maximum_facts: int,
    ) -> tuple[RagGraphFact, ...]:
        """Returns facts associated with the requested dish identifiers."""
        raise NotImplementedError
