from __future__ import annotations

import json
from pathlib import Path

from cuisine_assistant.domain.ports import IngredientExtractor
from cuisine_assistant.evaluation.metrics import ExtractionMetrics, calculate_metrics


def evaluate_dataset(extractor: IngredientExtractor, dataset_path: Path) -> ExtractionMetrics:
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected_all: set[str] = set()
    predicted_all: set[str] = set()
    for index, record in enumerate(records):
        expected = {f"{index}:{item}" for item in record["expected_ingredient_ids"]}
        predicted = {
            f"{index}:{item.matched_ingredient_id}"
            for item in extractor.extract(record["text"])
            if item.matched_ingredient_id is not None
        }
        expected_all.update(expected)
        predicted_all.update(predicted)
    return calculate_metrics(expected_all, predicted_all)
