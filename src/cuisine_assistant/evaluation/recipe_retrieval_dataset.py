from __future__ import annotations

import json
from pathlib import Path

from cuisine_assistant.evaluation.recipe_retrieval import (
    LabeledRecipeSearchExample,
)


class JsonRecipeRetrievalDatasetLoader:
    """Loads manually labeled recipe-search examples from JSON."""

    def load(
        self,
        path: str | Path,
    ) -> tuple[LabeledRecipeSearchExample, ...]:
        """Returns immutable labeled queries from a UTF-8 JSON array."""
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Recipe retrieval dataset does not exist: {dataset_path}"
            )
        try:
            records = json.loads(
                dataset_path.read_text(encoding="utf-8-sig")
            )
        except json.JSONDecodeError as error:
            raise ValueError("Invalid recipe retrieval JSON dataset") from error
        if not isinstance(records, list):
            raise ValueError("Recipe retrieval dataset root must be a JSON array")
        return tuple(
            self._map_record(record, index)
            for index, record in enumerate(records, start=1)
        )

    @staticmethod
    def _map_record(record, index: int) -> LabeledRecipeSearchExample:
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {index} must be an object")
        try:
            relevant = record["relevant_recipe_ids"]
            if not isinstance(relevant, list):
                raise ValueError(
                    f"Dataset record {index} relevant_recipe_ids must be a list"
                )
            return LabeledRecipeSearchExample(
                example_id=str(record["example_id"]),
                query_text=str(record["query_text"]),
                relevant_recipe_ids=tuple(str(item) for item in relevant),
            )
        except KeyError as error:
            raise ValueError(
                f"Dataset record {index} is missing field: {error.args[0]}"
            ) from error
