from __future__ import annotations

import json
from pathlib import Path

from cuisine_assistant.evaluation.dish_recommendation import (
    LabeledRecommendationExample,
)


class JsonRecommendationDatasetLoader:
    """Loads manually labeled dish-recommendation examples from JSON."""

    def load(
        self,
        path: str | Path,
    ) -> tuple[LabeledRecommendationExample, ...]:
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Recommendation dataset does not exist: {dataset_path}"
            )
        records = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Recommendation dataset root must be a JSON array")
        return tuple(
            self._map_record(record, index)
            for index, record in enumerate(records, start=1)
        )

    @staticmethod
    def _map_record(record, index: int) -> LabeledRecommendationExample:
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {index} must be an object")
        try:
            relevant = tuple(str(item) for item in record["relevant_dish_ids"])
            return LabeledRecommendationExample(
                example_id=str(record["example_id"]),
                text=str(record["text"]),
                relevant_dish_ids=relevant,
            )
        except KeyError as error:
            raise ValueError(
                f"Dataset record {index} is missing field: {error.args[0]}"
            ) from error
