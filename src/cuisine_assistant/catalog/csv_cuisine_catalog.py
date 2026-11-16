from __future__ import annotations

import csv
from pathlib import Path

from cuisine_assistant.domain.models import Cuisine
from cuisine_assistant.domain.ports import CuisineCatalog


class CsvCuisineCatalog(CuisineCatalog):
    """Loads canonical cuisine records from a UTF-8 CSV file into memory."""

    _REQUIRED_COLUMNS = {
        "cuisine_id",
        "name",
        "cuisine_group",
        "ontology_id",
    }

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)
        cuisines = self._load_cuisines()
        self._cuisines = tuple(cuisines)
        self._by_id = self._build_unique_index(
            cuisines,
            key_name="cuisine_id",
            key_selector=lambda cuisine: cuisine.cuisine_id,
        )
        self._by_ontology_id = self._build_unique_index(
            cuisines,
            key_name="ontology_id",
            key_selector=lambda cuisine: cuisine.ontology_id,
        )

    def get_by_id(self, cuisine_id: str) -> Cuisine | None:
        return self._by_id.get(cuisine_id)

    def get_by_ontology_id(self, ontology_id: str) -> Cuisine | None:
        return self._by_ontology_id.get(ontology_id)

    def list_all(self) -> tuple[Cuisine, ...]:
        return self._cuisines

    def _load_cuisines(self) -> list[Cuisine]:
        if not self._csv_path.is_file():
            raise FileNotFoundError(
                f"Cuisine CSV file does not exist: {self._csv_path}"
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
        missing = self._REQUIRED_COLUMNS - set(fieldnames or ())
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"Cuisine CSV is missing required columns: {columns}")

    def _map_row(self, row: dict[str, str | None], line_number: int) -> Cuisine:
        try:
            return Cuisine(
                cuisine_id=self._required(row, "cuisine_id"),
                name=self._required(row, "name"),
                ontology_id=self._required(row, "ontology_id"),
                cuisine_group=self._required(row, "cuisine_group"),
                origin_place_id=self._optional(row.get("origin_place_id")),
                description=self._optional(row.get("description")),
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid cuisine data at CSV line {line_number}: {error}"
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
    def _build_unique_index(cuisines, key_name: str, key_selector) -> dict[str, Cuisine]:
        index: dict[str, Cuisine] = {}
        for cuisine in cuisines:
            key = key_selector(cuisine)
            if key in index:
                raise ValueError(f"Duplicate {key_name} in cuisine CSV: {key}")
            index[key] = cuisine
        return index
