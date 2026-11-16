from __future__ import annotations

from pathlib import Path

from cuisine_assistant.evaluation import (
    ClassificationEvaluationFormatter,
    CuisineClassificationEvaluator,
    JsonClassificationDatasetLoader,
)
from evaluation_runtime import EvaluationRuntime


def main() -> None:
    """Evaluates cuisine classification with the selected knowledge base bundle."""
    project_root = Path(__file__).resolve().parents[1]
    container = EvaluationRuntime(project_root).build_container()
    examples = JsonClassificationDatasetLoader().load(
        project_root / "data/evaluation/cuisine_classification_examples.json"
    )
    evaluation = CuisineClassificationEvaluator(
        container.classify_recipe,
        top_k=3,
    ).evaluate(examples)
    print(
        f"Knowledge base: {container.knowledge_base.metadata.name} "
        f"({container.knowledge_base.metadata.version})"
    )
    print(ClassificationEvaluationFormatter().format(evaluation))


if __name__ == "__main__":
    main()
