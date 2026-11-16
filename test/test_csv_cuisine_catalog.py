from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.catalog import CsvCuisineCatalog


def write_csv(tmp_path: Path, content: str) -> Path:
    """Writes a cuisine CSV fixture for adapter tests."""
    path = tmp_path / "cuisines.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_cuisines_and_optional_fields(tmp_path: Path) -> None:
    catalog = CsvCuisineCatalog(
        write_csv(
            tmp_path,
            "cuisine_id,name,cuisine_group,origin_place_id,description,ontology_id\n"
            "hungarian,Hungarian cuisine,EuropeanCuisine,hungary,Central European cuisine,HungarianCuisine\n"
            "thai,Thai cuisine,AsianCuisine,thailand,,ThaiCuisine\n",
        )
    )
    hungarian = catalog.get_by_id("hungarian")
    assert hungarian is not None
    assert hungarian.ontology_id == "HungarianCuisine"
    assert hungarian.origin_place_id == "hungary"
    assert catalog.get_by_ontology_id("ThaiCuisine").cuisine_id == "thai"
    assert len(catalog.list_all()) == 2


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "cuisine_id,name,cuisine_group\nhungarian,Hungarian cuisine,EuropeanCuisine\n")
    with pytest.raises(ValueError, match="ontology_id"):
        CsvCuisineCatalog(path)


def test_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "cuisine_id,name,cuisine_group,ontology_id\n"
        "hungarian,Hungarian cuisine,EuropeanCuisine,HungarianCuisine\n"
        "hungarian,Other name,EuropeanCuisine,OtherCuisine\n",
    )
    with pytest.raises(ValueError, match="Duplicate cuisine_id"):
        CsvCuisineCatalog(path)
