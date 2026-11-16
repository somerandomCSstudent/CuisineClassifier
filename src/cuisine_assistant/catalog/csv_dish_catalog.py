from __future__ import annotations

import csv
from pathlib import Path

from cuisine_assistant.domain.recommendation_models import Dish
from cuisine_assistant.domain.recommendation_ports import DishCatalog


class CsvDishCatalog(DishCatalog):
    """Loads canonical dish records from a UTF-8 CSV file into memory."""

    _REQUIRED_COLUMNS = {
        "dish_id",
        "name",
        "dish_category",
        "cuisine_id",
        "ontology_id",
    }

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)
        dishes = self._load_dishes()
        self._dishes = tuple(dishes)
        self._by_id = self._build_unique_index(
            dishes,
            key_name="dish_id",
            key_selector=lambda dish: dish.dish_id,
        )
        self._by_ontology_id = self._build_unique_index(
            dishes,
            key_name="ontology_id",
            key_selector=lambda dish: dish.ontology_id,
        )

    def get_by_id(self, dish_id: str) -> Dish | None:
        return self._by_id.get(dish_id)

    def get_by_ontology_id(self, ontology_id: str) -> Dish | None:
        return self._by_ontology_id.get(ontology_id)

    def list_all(self) -> tuple[Dish, ...]:
        return self._dishes

    def _load_dishes(self) -> list[Dish]:
        if not self._csv_path.is_file():
            raise FileNotFoundError(f"Dish CSV file does not exist: {self._csv_path}")

        with self._csv_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            self._validate_header(reader.fieldnames)
            return [
                self._map_row(row, line_number)
                for line_number, row in enumerate(reader, start=2)
                if not self._is_empty_row(row)
            ]

    def _validate_header(self, fieldnames: list[str] | None) -> None:
        missing = self._REQUIRED_COLUMNS - set(fieldnames or ())
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"Dish CSV is missing required columns: {columns}")

    def _map_row(self, row: dict[str, str | None], line_number: int) -> Dish:
        try:
            return Dish(
                dish_id=self._required(row, "dish_id"),
                name=self._required(row, "name"),
                ontology_id=self._required(row, "ontology_id"),
                dish_category=self._required(row, "dish_category"),
                cuisine_id=self._required(row, "cuisine_id"),
                description=self._optional(row.get("description")),
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid dish data at CSV line {line_number}: {error}"
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
    def _build_unique_index(dishes, key_name: str, key_selector) -> dict[str, Dish]:
        index: dict[str, Dish] = {}
        for dish in dishes:
            key = key_selector(dish)
            if key in index:
                raise ValueError(f"Duplicate {key_name} in dish CSV: {key}")
            index[key] = dish
        return index
