from __future__ import annotations

from pathlib import Path

from cuisine_assistant.application import SelectKnowledgeBaseUseCase
from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.cli import (
    ClassificationResultFormatter,
    ConsoleApplication,
    KnowledgeBaseSelector,
    LinkIngredientsResultFormatter,
    RecommendationResultFormatter,
)
from cuisine_assistant.knowledge_base import (
    FileKnowledgeBaseRegistry,
    FileKnowledgeBaseValidator,
    JsonKnowledgeBaseManifestLoader,
)


def main() -> None:
    """Selects a complete knowledge base bundle and starts the CLI."""
    project_root = Path(__file__).resolve().parents[2]
    validator = FileKnowledgeBaseValidator()
    registry = FileKnowledgeBaseRegistry(
        knowledge_bases_root=project_root / "knowledge_bases",
        manifest_loader=JsonKnowledgeBaseManifestLoader(),
        validator=validator,
        strict=False,
    )
    selection_use_case = SelectKnowledgeBaseUseCase(
        registry=registry,
        validator=validator,
    )
    selected = KnowledgeBaseSelector(selection_use_case).select()
    container = build_application_container(
        bundle=selected.bundle,
        settings=ApplicationSettings(),
    )

    ConsoleApplication(
        knowledge_base=container.knowledge_base.metadata,
        link_ingredients=container.link_ingredients,
        classify_recipe=container.classify_recipe,
        recommend_dishes=container.recommend_dishes,
        link_formatter=LinkIngredientsResultFormatter(),
        classification_formatter=ClassificationResultFormatter(),
        recommendation_formatter=RecommendationResultFormatter(),
    ).run()


if __name__ == "__main__":
    main()
