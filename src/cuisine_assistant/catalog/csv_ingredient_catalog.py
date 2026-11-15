from __future__ import annotations

import csv
from pathlib import Path

from cuisine_assistant.domain.models import Ingredient, IngredientCategory
from cuisine_assistant.domain.ports import IngredientCatalog


class CsvIngredientCatalog(IngredientCatalog):
    """Loads canonical ingredients from a UTF-8 CSV file into memory."""

    _REQUIRED_COLUMNS = {
        "ingredient_id",
        "canonical_name",
        "ingredient_category",
        "ontology_id",
    }

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)
        ingredients = self._load_ingredients()
        self._ingredients = tuple(ingredients)
        self._by_id = self._build_unique_index(
            ingredients,
            key_name="ingredient_id",
            key_selector=lambda ingredient: ingredient.ingredient_id,
        )
        self._by_ontology_id = self._build_unique_index(
            ingredients,
            key_name="ontology_id",
            key_selector=lambda ingredient: ingredient.ontology_id,
        )

    def get_by_id(self, ingredient_id: str) -> Ingredient | None:
        return self._by_id.get(ingredient_id)

    def get_by_ontology_id(self, ontology_id: str) -> Ingredient | None:
        return self._by_ontology_id.get(ontology_id)

    def list_all(self) -> tuple[Ingredient, ...]:
        return self._ingredients

    def _load_ingredients(self) -> list[Ingredient]:
        if not self._csv_path.is_file():
            raise FileNotFoundError(
                f"Ingredient CSV file does not exist: {self._csv_path}"
            )

        with self._csv_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            self._validate_header(reader.fieldnames)
            return [
                self._map_row(row, line_number)
                for line_number, row in enumerate(reader, start=2)
                if not self._is_empty_row(row)
            ]

    def _validate_header(self, fieldnames: list[str] | None) -> None:
        available = set(fieldnames or [])
        missing = self._REQUIRED_COLUMNS - available
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"Ingredient CSV is missing required columns: {columns}")

    def _map_row(self, row: dict[str, str | None], line_number: int) -> Ingredient:
        try:
            category = IngredientCategory(self._required(row, "ingredient_category"))
            description = self._optional(row.get("description"))
            return Ingredient(
                ingredient_id=self._required(row, "ingredient_id"),
                canonical_name=self._required(row, "canonical_name"),
                category=category,
                ontology_id=self._required(row, "ontology_id"),
                description=description,
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Invalid ingredient data at CSV line {line_number}: {error}"
            ) from error

    @staticmethod
    def _required(row: dict[str, str | None], column: str) -> str:
        value = row.get(column)
        if value is None or not value.strip():
            raise ValueError(f"{column} must not be empty")
        return value.strip()

    @staticmethod
    def _optional(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _is_empty_row(row: dict[str, str | None]) -> bool:
        return not any((value or "").strip() for value in row.values())

    @staticmethod
    def _build_unique_index(
        ingredients: list[Ingredient],
        key_name: str,
        key_selector,
    ) -> dict[str, Ingredient]:
        index: dict[str, Ingredient] = {}
        for ingredient in ingredients:
            key = key_selector(ingredient)
            if key in index:
                raise ValueError(f"Duplicate {key_name} in ingredient CSV: {key}")
            index[key] = ingredient
        return index
