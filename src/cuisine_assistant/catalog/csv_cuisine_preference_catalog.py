from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from cuisine_assistant.domain.models import Cuisine
from cuisine_assistant.domain.ports import CuisineCatalog
from cuisine_assistant.domain.recommendation_ports import CuisinePreferenceCatalog


class CsvCuisinePreferenceCatalog(CuisinePreferenceCatalog):
    """Maps normalized cuisine aliases from CSV to canonical cuisine records."""

    _REQUIRED_COLUMNS = {
        "cuisine_id",
        "alias",
    }

    def __init__(
        self,
        csv_path: str | Path,
        cuisine_catalog: CuisineCatalog,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._cuisine_catalog = cuisine_catalog
        self._aliases = self._load_aliases()

    def find_by_alias(self, normalized_alias: str) -> Cuisine | None:
        return self._aliases.get(self._normalize(normalized_alias))

    def aliases(self) -> Iterable[str]:
        return self._aliases.keys()

    def _load_aliases(self) -> dict[str, Cuisine]:
        if not self._csv_path.is_file():
            raise FileNotFoundError(
                f"Cuisine alias CSV file does not exist: {self._csv_path}"
            )

        aliases: dict[str, Cuisine] = {}
        priorities: dict[str, int] = {}
        with self._csv_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            self._validate_header(reader.fieldnames)
            for line_number, row in enumerate(reader, start=2):
                if self._is_empty_row(row):
                    continue
                cuisine_id = self._required(row, "cuisine_id")
                alias = self._normalize(self._required(row, "alias"))
                priority = self._parse_priority(row, line_number)
                cuisine = self._cuisine_catalog.get_by_id(cuisine_id)
                if cuisine is None:
                    raise ValueError(
                        f"Unknown cuisine_id at CSV line {line_number}: {cuisine_id}"
                    )

                existing = aliases.get(alias)
                if existing is not None and existing.cuisine_id != cuisine_id:
                    raise ValueError(
                        f"Cuisine alias maps to multiple cuisines at CSV line "
                        f"{line_number}: {alias}"
                    )
                if priority >= priorities.get(alias, -1):
                    aliases[alias] = cuisine
                    priorities[alias] = priority
        return aliases

    def _validate_header(self, fieldnames: list[str] | None) -> None:
        missing = self._REQUIRED_COLUMNS - set(fieldnames or ())
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(
                f"Cuisine alias CSV is missing required columns: {columns}"
            )

    @staticmethod
    def _parse_priority(row: dict[str, str | None], line_number: int) -> int:
        raw_value = (row.get("match_priority") or "0").strip()
        try:
            return int(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Invalid match_priority at CSV line {line_number}: {raw_value}"
            ) from error

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.lower().split())

    @staticmethod
    def _required(row: dict[str, str | None], column: str) -> str:
        value = row.get(column)
        if value is None or not value.strip():
            raise ValueError(f"{column} must not be empty")
        return value.strip()

    @staticmethod
    def _is_empty_row(row: dict[str, str | None]) -> bool:
        return not any((value or "").strip() for value in row.values())
