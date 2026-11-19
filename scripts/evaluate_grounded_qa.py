from __future__ import annotations

from pathlib import Path

from cuisine_assistant.evaluation import (
    GroundedQaEvaluationFormatter,
    GroundedQaEvaluator,
    JsonGroundedQaDatasetLoader,
)
from evaluation_runtime import EvaluationRuntime


def main() -> None:
    """Evaluates deterministic grounded QA with the selected knowledge base."""
    project_root = Path(__file__).resolve().parents[1]
    container = EvaluationRuntime(project_root).build_container()
    examples = JsonGroundedQaDatasetLoader().load(
        project_root / "data/evaluation/grounded_qa_examples.json"
    )
    evaluation = GroundedQaEvaluator(
        container.ask_cuisine_question
    ).evaluate(examples)
    print(
        f"Knowledge base: {container.knowledge_base.metadata.name} "
        f"({container.knowledge_base.metadata.version})"
    )
    print(GroundedQaEvaluationFormatter().format(evaluation))


if __name__ == "__main__":
    main()
