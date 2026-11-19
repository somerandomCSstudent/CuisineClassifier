from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.evaluation.grounded_qa_dataset import (
    JsonGroundedQaDatasetLoader,
)


def test_loads_answerable_and_unanswerable_questions(tmp_path: Path) -> None:
    """Verifies labeled grounded QA questions load from JSON."""
    path = tmp_path / "qa.json"
    path.write_text(
        '[{"example_id":"q1","question":"Question?",'
        '"expected_answerable":true,"relevant_recipe_ids":["r1"]},'
        '{"example_id":"q2","question":"Unknown?",'
        '"expected_answerable":false,"relevant_recipe_ids":[]}]',
        encoding="utf-8",
    )

    examples = JsonGroundedQaDatasetLoader().load(path)

    assert len(examples) == 2
    assert examples[0].relevant_recipe_ids == ("r1",)
    assert examples[1].expected_answerable is False


def test_rejects_answerable_question_without_relevant_recipe(tmp_path: Path) -> None:
    """Verifies answerable labels include at least one supporting recipe."""
    path = tmp_path / "qa.json"
    path.write_text(
        '[{"example_id":"q1","question":"Question?",'
        '"expected_answerable":true,"relevant_recipe_ids":[]}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant_recipe_ids"):
        JsonGroundedQaDatasetLoader().load(path)
