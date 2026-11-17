from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.retrieval import CsvRecipeDocumentRepository


def write_csv(tmp_path: Path, content: str, *, bom: bool = False) -> Path:
    """Writes one recipe CSV fixture for repository tests."""
    path = tmp_path / "recipes.csv"
    path.write_text(
        content,
        encoding="utf-8-sig" if bom else "utf-8",
    )
    return path


def test_loads_complete_recipe_document(tmp_path: Path) -> None:
    """Verifies all searchable and provenance fields are mapped correctly."""
    path = write_csv(
        tmp_path,
        "recipe_id,dish_id,title,raw_text,ingredient_text,instruction_text,"
        "cuisine_id,data_split,source,source_url,license\n"
        "r001,chicken_paprikash,Traditional Chicken Paprikash,"
        "Cook chicken with paprika,chicken; paprika,Finish with sour cream,"
        "hungarian,train,manual,https://example.org/r001,CC BY 4.0\n",
    )

    repository = CsvRecipeDocumentRepository(path)
    document = repository.get_by_id("r001")

    assert document is not None
    assert document.dish_id == "chicken_paprikash"
    assert document.ingredient_text == "chicken; paprika"
    assert document.instruction_text == "Finish with sour cream"
    assert document.cuisine_id == "hungarian"
    assert document.source_url == "https://example.org/r001"
    assert document.license_name == "CC BY 4.0"
    assert repository.list_all() == (document,)


def test_supports_utf8_bom_and_unicode_content(tmp_path: Path) -> None:
    """Verifies Windows-compatible UTF-8 CSV files preserve Unicode text."""
    path = write_csv(
        tmp_path,
        "recipe_id,dish_id,title,raw_text,source\n"
        "r1,goulash,Gulyás,Gulyás paprikával,manual\n",
        bom=True,
    )

    document = CsvRecipeDocumentRepository(path).get_by_id("r1")

    assert document is not None
    assert document.title == "Gulyás"


def test_maps_missing_optional_columns_to_empty_or_none(tmp_path: Path) -> None:
    """Verifies the minimal supported recipe schema remains searchable."""
    path = write_csv(
        tmp_path,
        "recipe_id,dish_id,title,raw_text,source\n"
        "r1,tomato_dish,Tomato Dish,Cook tomatoes,manual\n",
    )

    document = CsvRecipeDocumentRepository(path).get_by_id("r1")

    assert document is not None
    assert document.ingredient_text == ""
    assert document.instruction_text == ""
    assert document.cuisine_id is None
    assert document.source_url is None
    assert document.license_name is None


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    """Verifies malformed bundle recipe schemas fail immediately."""
    path = write_csv(
        tmp_path,
        "recipe_id,dish_id,title,source\n"
        "r1,dish,Dish,manual\n",
    )

    with pytest.raises(ValueError, match="raw_text"):
        CsvRecipeDocumentRepository(path)


def test_rejects_empty_required_value_with_line_number(tmp_path: Path) -> None:
    """Verifies invalid rows identify the source CSV line."""
    path = write_csv(
        tmp_path,
        "recipe_id,dish_id,title,raw_text,source\n"
        "r1,dish,,Recipe text,manual\n",
    )

    with pytest.raises(ValueError, match="CSV line 2"):
        CsvRecipeDocumentRepository(path)


def test_rejects_duplicate_recipe_identifier(tmp_path: Path) -> None:
    """Verifies stable document identifiers are unique within a bundle."""
    path = write_csv(
        tmp_path,
        "recipe_id,dish_id,title,raw_text,source\n"
        "r1,dish_one,Dish One,Text one,manual\n"
        "r1,dish_two,Dish Two,Text two,manual\n",
    )

    with pytest.raises(ValueError, match="Duplicate recipe_id"):
        CsvRecipeDocumentRepository(path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Verifies a bundle without recipes.csv cannot construct the repository."""
    with pytest.raises(FileNotFoundError, match="Recipe CSV file"):
        CsvRecipeDocumentRepository(tmp_path / "missing.csv")
