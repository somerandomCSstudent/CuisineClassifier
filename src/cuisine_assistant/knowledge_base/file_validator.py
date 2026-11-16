from __future__ import annotations

from cuisine_assistant.knowledge_base.models import (
    KnowledgeBaseBundle,
    KnowledgeBaseValidationResult,
)
from cuisine_assistant.knowledge_base.ports import KnowledgeBaseValidator


class FileKnowledgeBaseValidator(KnowledgeBaseValidator):
    """Validates that every mandatory knowledge base bundle file exists."""

    def validate(
        self,
        bundle: KnowledgeBaseBundle,
    ) -> KnowledgeBaseValidationResult:
        missing = tuple(
            path
            for path in bundle.paths.required_files()
            if not path.is_file()
        )
        return KnowledgeBaseValidationResult(
            is_valid=not missing,
            missing_files=missing,
        )
