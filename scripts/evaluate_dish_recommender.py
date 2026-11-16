from __future__ import annotations

from pathlib import Path

from cuisine_assistant.evaluation import (
    DishRecommendationEvaluator,
    JsonRecommendationDatasetLoader,
    RecommendationEvaluationFormatter,
)
from evaluation_runtime import EvaluationRuntime


def main() -> None:
    """Evaluates dish recommendation with the selected knowledge base bundle."""
    project_root = Path(__file__).resolve().parents[1]
    container = EvaluationRuntime(project_root).build_container()
    examples = JsonRecommendationDatasetLoader().load(
        project_root / "data/evaluation/dish_recommendation_examples.json"
    )
    evaluation = DishRecommendationEvaluator(
        container.recommend_dishes,
        top_k=3,
    ).evaluate(examples)
    print(
        f"Knowledge base: {container.knowledge_base.metadata.name} "
        f"({container.knowledge_base.metadata.version})"
    )
    print(RecommendationEvaluationFormatter().format(evaluation))


if __name__ == "__main__":
    main()
