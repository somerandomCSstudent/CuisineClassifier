from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.application.select_knowledge_base import (
    SelectKnowledgeBaseUseCase,
)
from cuisine_assistant.knowledge_base.models import KnowledgeBaseBundle


@dataclass(frozen=True)
class ResolveKnowledgeBaseResult:
    """Contains the bundle resolved for a non-interactive application command."""

    bundle: KnowledgeBaseBundle


class ResolveKnowledgeBaseUseCase:
    """Resolves an explicit or default knowledge base for scripts and automation."""

    def __init__(self, selection_use_case: SelectKnowledgeBaseUseCase) -> None:
        self._selection_use_case = selection_use_case

    def execute(
        self,
        requested_id: str | None,
    ) -> ResolveKnowledgeBaseResult:
        """Resolves the requested bundle or the only available bundle."""
        if requested_id and requested_id.strip():
            selected = self._selection_use_case.execute(requested_id.strip())
            return ResolveKnowledgeBaseResult(bundle=selected.bundle)

        available = self._selection_use_case.list_available()
        if not available:
            raise RuntimeError("No valid knowledge base bundle is available")
        if len(available) > 1:
            identifiers = ", ".join(
                summary.knowledge_base_id for summary in available
            )
            raise ValueError(
                "Multiple knowledge bases are available. "
                f"Specify one of: {identifiers}"
            )

        selected = self._selection_use_case.execute(
            available[0].knowledge_base_id
        )
        return ResolveKnowledgeBaseResult(bundle=selected.bundle)
