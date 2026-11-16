from __future__ import annotations

import json
from pathlib import Path

from cuisine_assistant.evaluation.cuisine_classification import LabeledRecipe


class JsonClassificationDatasetLoader:
    """Loads manually labeled English recipe examples from a JSON file."""

    def load(self, path: str | Path) -> tuple[LabeledRecipe, ...]:
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Classification dataset does not exist: {dataset_path}"
            )
        raw_records = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw_records, list):
            raise ValueError("Classification dataset root must be a JSON array")
        return tuple(
            self._map_record(record, index)
            for index, record in enumerate(raw_records, start=1)
        )

    @staticmethod
    def _map_record(record, index: int) -> LabeledRecipe:
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {index} must be an object")
        try:
            return LabeledRecipe(
                recipe_id=str(record["recipe_id"]),
                text=str(record["text"]),
                expected_cuisine_id=str(record["expected_cuisine_id"]),
            )
        except KeyError as error:
            raise ValueError(
                f"Dataset record {index} is missing field: {error.args[0]}"
            ) from error
