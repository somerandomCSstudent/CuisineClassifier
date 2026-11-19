from __future__ import annotations

from abc import ABC, abstractmethod

from cuisine_assistant.rag.answer_models import GroundedAnswer
from cuisine_assistant.rag.models import RagContext


class GroundedAnswerGenerator(ABC):
    """Generates an answer using only evidence supplied in a RAG context."""

    @abstractmethod
    def generate(self, context: RagContext) -> GroundedAnswer:
        """Returns a grounded answer with explicit evidence references."""
        raise NotImplementedError
