from __future__ import annotations

from pathlib import Path

from cuisine_assistant.evaluation import (
    JsonRecipeRetrievalDatasetLoader,
    RecipeRetrievalEvaluationFormatter,
    RecipeRetrievalEvaluator,
)
from evaluation_runtime import EvaluationRuntime


def main() -> None:
    """Evaluates BM25 recipe retrieval with the selected knowledge base bundle."""
    project_root = Path(__file__).resolve().parents[1]
    container = EvaluationRuntime(project_root).build_container()
    examples = JsonRecipeRetrievalDatasetLoader().load(
        project_root / "data/evaluation/recipe_retrieval_examples.json"
    )
    evaluation = RecipeRetrievalEvaluator(
        container.search_recipes
    ).evaluate(examples)
    print(
        f"Knowledge base: {container.knowledge_base.metadata.name} "
        f"({container.knowledge_base.metadata.version})"
    )
    print(RecipeRetrievalEvaluationFormatter().format(evaluation))


if __name__ == "__main__":
    main()
