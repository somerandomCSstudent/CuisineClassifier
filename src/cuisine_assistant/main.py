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
    RecipeSearchResultFormatter,
    RecommendationResultFormatter,
    GroundedAnswerFormatter
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
        search_recipes=container.search_recipes,
        ask_cuisine_question=container.ask_cuisine_question,
        link_formatter=LinkIngredientsResultFormatter(),
        classification_formatter=ClassificationResultFormatter(),
        recommendation_formatter=RecommendationResultFormatter(),
        recipe_search_formatter=RecipeSearchResultFormatter(),
        grounded_answer_formatter=GroundedAnswerFormatter(),
    ).run()



if __name__ == "__main__":
    main()
