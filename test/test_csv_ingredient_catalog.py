from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.catalog.csv_ingredient_catalog import CsvIngredientCatalog
from cuisine_assistant.domain.models import IngredientCategory


def write_csv(tmp_path: Path, content: str) -> Path:
    csv_path = tmp_path / "ingredients.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_loads_and_maps_ingredients(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        "ingredient_id,canonical_name,ingredient_category,description,ontology_id\n"
        "tomato,tomato,Vegetable,Fresh or processed tomato,Tomato\n"
        "paprika,paprika,Spice,,Paprika\n",
    )

    catalog = CsvIngredientCatalog(csv_path)

    tomato = catalog.get_by_id("tomato")
    assert tomato is not None
    assert tomato.category is IngredientCategory.VEGETABLE
    assert tomato.description == "Fresh or processed tomato"
    assert catalog.get_by_ontology_id("Paprika").ingredient_id == "paprika"
    assert len(catalog.list_all()) == 2


def test_supports_utf8_bom(tmp_path: Path) -> None:
    csv_path = tmp_path / "ingredients.csv"
    csv_path.write_text(
        "ingredient_id,canonical_name,ingredient_category,ontology_id\n"
        "sour_cream,sour cream,DairyProduct,SourCream\n",
        encoding="utf-8-sig",
    )

    catalog = CsvIngredientCatalog(csv_path)

    assert catalog.get_by_id("sour_cream").ontology_id == "SourCream"


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        "ingredient_id,canonical_name,ingredient_category\n"
        "tomato,tomato,Vegetable\n",
    )

    with pytest.raises(ValueError, match="ontology_id"):
        CsvIngredientCatalog(csv_path)


def test_rejects_unknown_category(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        "ingredient_id,canonical_name,ingredient_category,ontology_id\n"
        "tomato,tomato,UnknownCategory,Tomato\n",
    )

    with pytest.raises(ValueError, match="CSV line 2"):
        CsvIngredientCatalog(csv_path)


def test_rejects_duplicate_ingredient_id(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        "ingredient_id,canonical_name,ingredient_category,ontology_id\n"
        "tomato,tomato,Vegetable,Tomato\n"
        "tomato,red tomato,Vegetable,RedTomato\n",
    )

    with pytest.raises(ValueError, match="Duplicate ingredient_id"):
        CsvIngredientCatalog(csv_path)


def test_rejects_duplicate_ontology_id(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        "ingredient_id,canonical_name,ingredient_category,ontology_id\n"
        "tomato,tomato,Vegetable,Tomato\n"
        "red_tomato,red tomato,Vegetable,Tomato\n",
    )

    with pytest.raises(ValueError, match="Duplicate ontology_id"):
        CsvIngredientCatalog(csv_path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CsvIngredientCatalog(tmp_path / "missing.csv")
