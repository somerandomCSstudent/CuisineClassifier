from __future__ import annotations

import os
from pathlib import Path

from cuisine_assistant.application import (
    ResolveKnowledgeBaseUseCase,
    SelectKnowledgeBaseUseCase,
)
from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.knowledge_base import (
    FileKnowledgeBaseRegistry,
    FileKnowledgeBaseValidator,
    JsonKnowledgeBaseManifestLoader,
)


class EvaluationRuntime:
    """Builds evaluation commands from one validated knowledge base bundle."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def build_container(self):
        """Resolves a bundle from the environment and builds the application."""
        validator = FileKnowledgeBaseValidator()
        registry = FileKnowledgeBaseRegistry(
            self._project_root / "knowledge_bases",
            JsonKnowledgeBaseManifestLoader(),
            validator,
            strict=False,
        )
        selector = SelectKnowledgeBaseUseCase(registry, validator)
        resolver = ResolveKnowledgeBaseUseCase(selector)
        requested_id = os.environ.get("CUISINE_KNOWLEDGE_BASE_ID")
        bundle = resolver.execute(requested_id).bundle
        return build_application_container(
            bundle=bundle,
            settings=ApplicationSettings(),
        )
