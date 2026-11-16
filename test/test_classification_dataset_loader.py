from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.evaluation import JsonClassificationDatasetLoader


def test_loads_labeled_recipe_dataset(tmp_path: Path) -> None:
    """Verifies JSON records are converted into immutable labeled recipes."""
    path = tmp_path / "examples.json"
    path.write_text(
        '[{"recipe_id":"r1","text":"Tomato and basil",'
        '"expected_cuisine_id":"italian"}]',
        encoding="utf-8",
    )

    examples = JsonClassificationDatasetLoader().load(path)

    assert len(examples) == 1
    assert examples[0].expected_cuisine_id == "italian"


def test_rejects_missing_required_field(tmp_path: Path) -> None:
    """Verifies malformed evaluation examples fail with a clear message."""
    path = tmp_path / "examples.json"
    path.write_text('[{"recipe_id":"r1","text":"Tomato"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="expected_cuisine_id"):
        JsonClassificationDatasetLoader().load(path)
