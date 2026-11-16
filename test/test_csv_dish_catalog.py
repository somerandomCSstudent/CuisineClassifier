from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.catalog.csv_dish_catalog import CsvDishCatalog


def write_csv(tmp_path: Path, content: str) -> Path:
    """Writes a dish CSV fixture for adapter tests."""
    path = tmp_path / "dishes.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_dishes_and_optional_description(tmp_path: Path) -> None:
    catalog = CsvDishCatalog(
        write_csv(
            tmp_path,
            "dish_id,name,dish_category,cuisine_id,description,ontology_id\n"
            "chicken_paprikash,Chicken Paprikash,MainCourse,hungarian,"
            "Chicken with paprika and sour cream,ChickenPaprikash\n",
        )
    )
    dish = catalog.get_by_id("chicken_paprikash")
    assert dish is not None
    assert dish.ontology_id == "ChickenPaprikash"
    assert catalog.get_by_ontology_id("ChickenPaprikash") is dish
    assert len(catalog.list_all()) == 1


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "dish_id,name,dish_category,cuisine_id\n"
        "dish,Dish,MainCourse,hungarian\n",
    )
    with pytest.raises(ValueError, match="ontology_id"):
        CsvDishCatalog(path)


def test_rejects_duplicate_dish_id(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "dish_id,name,dish_category,cuisine_id,ontology_id\n"
        "dish,Dish One,MainCourse,hungarian,DishOne\n"
        "dish,Dish Two,Soup,hungarian,DishTwo\n",
    )
    with pytest.raises(ValueError, match="Duplicate dish_id"):
        CsvDishCatalog(path)
