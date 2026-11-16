from __future__ import annotations

import os
from pathlib import Path

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.evaluation import (
    ClassificationEvaluationFormatter,
    CuisineClassificationEvaluator,
    JsonClassificationDatasetLoader,
)


def main() -> None:
    """Runs cuisine-classification evaluation on the labeled recipe dataset."""
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
    examples = JsonClassificationDatasetLoader().load(
        project_root / "data/evaluation/cuisine_classification_examples.json"
    )
    evaluation = CuisineClassificationEvaluator(
        container.classify_recipe,
        top_k=3,
    ).evaluate(examples)
    print(ClassificationEvaluationFormatter().format(evaluation))


if __name__ == "__main__":
    main()
