from __future__ import annotations

import os
from pathlib import Path

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.evaluation import (
    DishRecommendationEvaluator,
    JsonRecommendationDatasetLoader,
    RecommendationEvaluationFormatter,
)


def main() -> None:
    """Runs recommendation evaluation on the labeled dish dataset."""
    project_root = Path(__file__).resolve().parents[1]
    base_iri = os.environ.get("CUISINE_ONTOLOGY_BASE_IRI")
    if not base_iri:
        raise RuntimeError("CUISINE_ONTOLOGY_BASE_IRI must be set")

    container = build_application_container(
        ApplicationSettings(
            project_root=project_root,
            ontology_base_iri=base_iri,
        )
    )
    examples = JsonRecommendationDatasetLoader().load(
        project_root / "data/evaluation/dish_recommendation_examples.json"
    )
    evaluation = DishRecommendationEvaluator(
        container.recommend_dishes,
        top_k=3,
    ).evaluate(examples)
    print(RecommendationEvaluationFormatter().format(evaluation))


if __name__ == "__main__":
    main()
