from __future__ import annotations

import csv
from pathlib import Path

from cuisine_assistant.domain.ports import CuisineTypicalityRepository


class CsvCuisineTypicalityRepository(CuisineTypicalityRepository):
    """Loads cuisine-to-ingredient typicality weights from a UTF-8 CSV file."""

    _REQUIRED_COLUMNS = {
        "cuisine_id",
        "ingredient_id",
        "typicality_score",
    }

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)
        self._weights = self._load_weights()
        self._ingredient_ids_by_cuisine = self._build_cuisine_index()

    def get_weight(self, cuisine_id: str, ingredient_id: str) -> float:
        return self._weights.get((cuisine_id, ingredient_id), 0.0)

    def get_typical_ingredient_ids(self, cuisine_id: str) -> tuple[str, ...]:
        return self._ingredient_ids_by_cuisine.get(cuisine_id, ())

    def _load_weights(self) -> dict[tuple[str, str], float]:
        if not self._csv_path.is_file():
            raise FileNotFoundError(
                f"Cuisine typicality CSV file does not exist: {self._csv_path}"
            )
        with self._csv_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            self._validate_header(reader.fieldnames)
            weights: dict[tuple[str, str], float] = {}
            for line_number, row in enumerate(reader, start=2):
                if self._is_empty_row(row):
                    continue
                cuisine_id = self._required(row, "cuisine_id")
                ingredient_id = self._required(row, "ingredient_id")
                score = self._parse_score(row, line_number)
                key = (cuisine_id, ingredient_id)
                if key in weights:
                    raise ValueError(
                        "Duplicate cuisine-ingredient typicality relation at "
                        f"CSV line {line_number}: {cuisine_id}, {ingredient_id}"
                    )
                weights[key] = score
            return weights

    def _build_cuisine_index(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for (cuisine_id, ingredient_id), score in self._weights.items():
            if score > 0.0:
                grouped.setdefault(cuisine_id, []).append(ingredient_id)
        return {
            cuisine_id: tuple(sorted(set(ingredient_ids)))
            for cuisine_id, ingredient_ids in grouped.items()
        }

    def _validate_header(self, fieldnames: list[str] | None) -> None:
        missing = self._REQUIRED_COLUMNS - set(fieldnames or ())
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(
                f"Cuisine typicality CSV is missing required columns: {columns}"
            )

    def _parse_score(self, row: dict[str, str | None], line_number: int) -> float:
        raw_score = self._required(row, "typicality_score")
        try:
            score = float(raw_score)
        except ValueError as error:
            raise ValueError(
                f"Invalid typicality_score at CSV line {line_number}: {raw_score}"
            ) from error
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"typicality_score must be between 0.0 and 1.0 at CSV line {line_number}"
            )
        return score

    @staticmethod
    def _required(row: dict[str, str | None], column: str) -> str:
        value = row.get(column)
        if value is None or not value.strip():
            raise ValueError(f"{column} must not be empty")
        return value.strip()

    @staticmethod
    def _is_empty_row(row: dict[str, str | None]) -> bool:
        return not any((value or "").strip() for value in row.values())
