from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.catalog import CsvCuisineTypicalityRepository


def write_csv(tmp_path: Path, content: str) -> Path:
    """Writes a typicality CSV fixture for adapter tests."""
    path = tmp_path / "cuisine_ingredients.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_weights_and_typical_ingredient_ids(tmp_path: Path) -> None:
    repository = CsvCuisineTypicalityRepository(
        write_csv(
            tmp_path,
            "cuisine_id,ingredient_id,typicality_score,evidence_count,source\n"
            "hungarian,paprika,0.95,18,manual\n"
            "hungarian,sour_cream,0.78,12,manual\n"
            "italian,basil,0.76,14,manual\n",
        )
    )
    assert repository.get_weight("hungarian", "paprika") == pytest.approx(0.95)
    assert repository.get_weight("hungarian", "unknown") == 0.0
    assert repository.get_typical_ingredient_ids("hungarian") == ("paprika", "sour_cream")


def test_rejects_out_of_range_score(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "cuisine_id,ingredient_id,typicality_score\n"
        "hungarian,paprika,1.20\n",
    )
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        CsvCuisineTypicalityRepository(path)


def test_rejects_duplicate_relation(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "cuisine_id,ingredient_id,typicality_score\n"
        "hungarian,paprika,0.95\n"
        "hungarian,paprika,0.90\n",
    )
    with pytest.raises(ValueError, match="Duplicate cuisine-ingredient"):
        CsvCuisineTypicalityRepository(path)


def test_rejects_invalid_numeric_score(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "cuisine_id,ingredient_id,typicality_score\n"
        "hungarian,paprika,high\n",
    )
    with pytest.raises(ValueError, match="Invalid typicality_score"):
        CsvCuisineTypicalityRepository(path)
