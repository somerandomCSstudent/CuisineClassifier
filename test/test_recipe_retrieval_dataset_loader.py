from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.evaluation.recipe_retrieval_dataset import (
    JsonRecipeRetrievalDatasetLoader,
)


def test_loads_labeled_recipe_search_examples(tmp_path: Path) -> None:
    """Verifies JSON records become immutable labeled search examples."""
    path = tmp_path / "search.json"
    path.write_text(
        '[{"example_id":"e1","query_text":"chicken paprika",'
        '"relevant_recipe_ids":["r1"]}]',
        encoding="utf-8",
    )

    examples = JsonRecipeRetrievalDatasetLoader().load(path)

    assert examples[0].query_text == "chicken paprika"
    assert examples[0].relevant_recipe_ids == ("r1",)


def test_rejects_empty_relevant_recipe_ids(tmp_path: Path) -> None:
    """Verifies every search query has at least one relevant document."""
    path = tmp_path / "search.json"
    path.write_text(
        '[{"example_id":"e1","query_text":"chicken",'
        '"relevant_recipe_ids":[]}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant_recipe_ids"):
        JsonRecipeRetrievalDatasetLoader().load(path)
