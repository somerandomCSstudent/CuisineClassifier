from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.domain.models import Cuisine
from cuisine_assistant.domain.ports import CuisineCatalog
from cuisine_assistant.catalog.csv_cuisine_preference_catalog import (
    CsvCuisinePreferenceCatalog,
)


class InMemoryCuisineCatalog(CuisineCatalog):
    """Provides deterministic cuisines for preference-catalog tests."""

    def __init__(self) -> None:
        self._cuisines = (
            Cuisine(
                cuisine_id="hungarian",
                name="Hungarian cuisine",
                ontology_id="HungarianCuisine",
                cuisine_group="EuropeanCuisine",
            ),
            Cuisine(
                cuisine_id="italian",
                name="Italian cuisine",
                ontology_id="ItalianCuisine",
                cuisine_group="EuropeanCuisine",
            ),
        )

    def get_by_id(self, cuisine_id: str) -> Cuisine | None:
        return next((item for item in self._cuisines if item.cuisine_id == cuisine_id), None)

    def get_by_ontology_id(self, ontology_id: str) -> Cuisine | None:
        return next((item for item in self._cuisines if item.ontology_id == ontology_id), None)

    def list_all(self) -> tuple[Cuisine, ...]:
        return self._cuisines


def write_csv(tmp_path: Path, content: str) -> Path:
    """Writes a cuisine alias CSV fixture for adapter tests."""
    path = tmp_path / "cuisine_aliases.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_maps_normalized_aliases_to_cuisines(tmp_path: Path) -> None:
    catalog = CsvCuisinePreferenceCatalog(
        write_csv(
            tmp_path,
            "alias_id,cuisine_id,alias,alias_type,language,match_priority,verified\n"
            "hu1,hungarian,Hungarian,canonical,en,100,true\n"
            "hu2,hungarian,Hungarian food,phrase,en,95,true\n"
            "it1,italian,Italian cuisine,phrase,en,95,true\n",
        ),
        InMemoryCuisineCatalog(),
    )
    assert catalog.find_by_alias("  HUNGARIAN   FOOD ").cuisine_id == "hungarian"
    assert catalog.find_by_alias("italian cuisine").cuisine_id == "italian"
    assert "hungarian" in tuple(catalog.aliases())


def test_rejects_unknown_cuisine_id(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "cuisine_id,alias\nunknown,unknown food\n",
    )
    with pytest.raises(ValueError, match="Unknown cuisine_id"):
        CsvCuisinePreferenceCatalog(path, InMemoryCuisineCatalog())


def test_rejects_conflicting_alias_mapping(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "cuisine_id,alias\nhungarian,european food\nitalian,european food\n",
    )
    with pytest.raises(ValueError, match="multiple cuisines"):
        CsvCuisinePreferenceCatalog(path, InMemoryCuisineCatalog())
