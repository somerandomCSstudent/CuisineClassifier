from __future__ import annotations

import csv
from pathlib import Path

from cuisine_assistant.retrieval.models import RecipeDocument
from cuisine_assistant.retrieval.ports import RecipeDocumentRepository


class CsvRecipeDocumentRepository(RecipeDocumentRepository):
    """Loads searchable recipe documents from one bundle-scoped UTF-8 CSV file."""

    _REQUIRED_COLUMNS = {
        "recipe_id",
        "dish_id",
        "title",
        "raw_text",
        "source",
    }

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)
        documents = self._load_documents()
        self._documents = tuple(documents)
        self._by_id = self._build_unique_index(documents)

    def get_by_id(self, recipe_id: str) -> RecipeDocument | None:
        """Returns one recipe document by its stable identifier."""
        return self._by_id.get(recipe_id.strip())

    def list_all(self) -> tuple[RecipeDocument, ...]:
        """Returns all recipe documents in deterministic CSV order."""
        return self._documents

    def _load_documents(self) -> list[RecipeDocument]:
        if not self._csv_path.is_file():
            raise FileNotFoundError(
                f"Recipe CSV file does not exist: {self._csv_path}"
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
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"Recipe CSV is missing required columns: {names}"
            )

    def _map_row(
        self,
        row: dict[str, str | None],
        line_number: int,
    ) -> RecipeDocument:
        try:
            return RecipeDocument(
                recipe_id=self._required(row, "recipe_id"),
                dish_id=self._required(row, "dish_id"),
                title=self._required(row, "title"),
                raw_text=self._required(row, "raw_text"),
                ingredient_text=self._optional(row.get("ingredient_text")) or "",
                instruction_text=self._optional(row.get("instruction_text")) or "",
                cuisine_id=self._optional(row.get("cuisine_id")),
                source=self._required(row, "source"),
                source_url=self._optional(row.get("source_url")),
                license_name=self._optional(row.get("license")),
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid recipe data at CSV line {line_number}: {error}"
            ) from error

    @staticmethod
    def _required(
        row: dict[str, str | None],
        column: str,
    ) -> str:
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
        documents: list[RecipeDocument],
    ) -> dict[str, RecipeDocument]:
        index: dict[str, RecipeDocument] = {}
        for document in documents:
            if document.recipe_id in index:
                raise ValueError(
                    "Duplicate recipe_id in recipe CSV: "
                    f"{document.recipe_id}"
                )
            index[document.recipe_id] = document
        return index
