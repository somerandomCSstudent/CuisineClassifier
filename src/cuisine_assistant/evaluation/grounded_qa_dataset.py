from __future__ import annotations

import json
from pathlib import Path

from cuisine_assistant.evaluation.grounded_qa import LabeledCuisineQuestion


class JsonGroundedQaDatasetLoader:
    """Loads labeled grounded cuisine questions from a UTF-8 JSON file."""

    def load(
        self,
        path: str | Path,
    ) -> tuple[LabeledCuisineQuestion, ...]:
        """Returns immutable grounded QA examples."""
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Grounded QA dataset does not exist: {dataset_path}"
            )
        try:
            document = json.loads(
                dataset_path.read_text(encoding="utf-8-sig")
            )
        except json.JSONDecodeError as error:
            raise ValueError("Invalid grounded QA JSON dataset") from error
        if not isinstance(document, list):
            raise ValueError("Grounded QA dataset root must be a JSON array")

        examples = []
        for index, record in enumerate(document, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Dataset record {index} must be an object")
            try:
                recipe_ids = record.get("relevant_recipe_ids", [])
                if not isinstance(recipe_ids, list):
                    raise ValueError(
                        f"Dataset record {index} relevant_recipe_ids "
                        "must be a list"
                    )
                examples.append(
                    LabeledCuisineQuestion(
                        example_id=str(record["example_id"]),
                        question=str(record["question"]),
                        expected_answerable=bool(
                            record["expected_answerable"]
                        ),
                        relevant_recipe_ids=tuple(
                            str(item) for item in recipe_ids
                        ),
                    )
                )
            except KeyError as error:
                raise ValueError(
                    f"Dataset record {index} is missing field: "
                    f"{error.args[0]}"
                ) from error
        return tuple(examples)
