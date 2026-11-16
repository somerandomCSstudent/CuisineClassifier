from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.evaluation import JsonRecommendationDatasetLoader


def test_loads_recommendation_examples(tmp_path: Path) -> None:
    """Verifies JSON records are converted into labeled recommendation examples."""
    path = tmp_path / "recommendations.json"
    path.write_text(
        '[{"example_id":"e1","text":"Chicken and paprika",'
        '"relevant_dish_ids":["chicken_paprikash"]}]',
        encoding="utf-8",
    )

    examples = JsonRecommendationDatasetLoader().load(path)

    assert len(examples) == 1
    assert examples[0].relevant_dish_ids == ("chicken_paprikash",)


def test_rejects_empty_relevant_dish_ids(tmp_path: Path) -> None:
    """Verifies every evaluation example has a relevant target dish."""
    path = tmp_path / "recommendations.json"
    path.write_text(
        '[{"example_id":"e1","text":"Chicken","relevant_dish_ids":[]}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant_dish_ids"):
        JsonRecommendationDatasetLoader().load(path)
