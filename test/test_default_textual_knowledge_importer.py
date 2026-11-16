from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cuisine_assistant.importing import (
    DefaultTextualKnowledgeImporter,
    ImportedDishRecord,
    ImportSourceType,
    TextualImportOptions,
    TextualImportSource,
)
from cuisine_assistant.importing.ports import (
    TextualRecordMapper,
    TextualRecordSource,
)


class InMemoryTextualRecordSource(TextualRecordSource):
    """Streams deterministic raw records for importer tests."""

    def __init__(self, records: list[Any]) -> None:
        self._records = records

    def read(
        self,
        source: TextualImportSource,
        options: TextualImportOptions,
    ) -> Iterable[Mapping[str, Any]]:
        return iter(self._records)


class CapturingTextualRecordMapper(TextualRecordMapper):
    """Maps basic dish fields and captures sanitized records for assertions."""

    def __init__(self) -> None:
        self.received_records: list[Mapping[str, Any]] = []

    def map_record(
        self,
        raw_record: Mapping[str, Any],
        source: TextualImportSource,
        options: TextualImportOptions,
    ) -> ImportedDishRecord | None:
        self.received_records.append(raw_record)
        if raw_record.get("skip"):
            return None
        return ImportedDishRecord(
            external_id=str(raw_record["id"]),
            name=str(raw_record["name"]),
            description=str(raw_record.get("description") or "") or None,
            aliases=(),
            categories=(),
            cuisines=(),
            places=(),
            ingredients=(),
            source_url=str(raw_record.get("source_url") or "") or None,
        )


def source() -> TextualImportSource:
    """Creates a Hugging Face textual source fixture."""
    return TextualImportSource(
        ImportSourceType.HUGGING_FACE,
        "worldcuisines/food-kb",
        "food-kb",
    )


def test_imports_mapped_records_and_honors_limit() -> None:
    mapper = CapturingTextualRecordMapper()
    importer = DefaultTextualKnowledgeImporter(
        InMemoryTextualRecordSource(
            [
                {"id": "1", "name": "Dish One"},
                {"id": "2", "name": "Dish Two"},
                {"id": "3", "name": "Dish Three"},
            ]
        ),
        mapper,
    )

    batch = importer.import_records(
        source(),
        TextualImportOptions(maximum_records=2),
    )

    assert tuple(item.external_id for item in batch.records) == ("1", "2")
    assert len(mapper.received_records) == 2


def test_removes_image_fields_recursively_before_mapping() -> None:
    mapper = CapturingTextualRecordMapper()
    importer = DefaultTextualKnowledgeImporter(
        InMemoryTextualRecordSource(
            [
                {
                    "id": "1",
                    "name": "Dish",
                    "image_url": "https://example.invalid/dish.jpg",
                    "nested": {
                        "thumbnail": "https://example.invalid/thumb.png",
                        "description": "Text remains available",
                    },
                    "items": [
                        {
                            "images": ["ignored.jpg"],
                            "value": "kept",
                        }
                    ],
                }
            ]
        ),
        mapper,
    )

    importer.import_records(source(), TextualImportOptions())

    sanitized = mapper.received_records[0]
    assert "image_url" not in sanitized
    assert "thumbnail" not in sanitized["nested"]
    assert sanitized["nested"]["description"] == "Text remains available"
    assert "images" not in sanitized["items"][0]
    assert sanitized["items"][0]["value"] == "kept"


def test_deduplicates_records_by_external_identifier() -> None:
    importer = DefaultTextualKnowledgeImporter(
        InMemoryTextualRecordSource(
            [
                {"id": "same", "name": "Dish One"},
                {"id": "same", "name": "Dish Duplicate"},
            ]
        ),
        CapturingTextualRecordMapper(),
    )

    batch = importer.import_records(source(), TextualImportOptions())

    assert len(batch.records) == 1
    assert batch.duplicate_record_count == 1


def test_can_preserve_duplicates_when_disabled() -> None:
    importer = DefaultTextualKnowledgeImporter(
        InMemoryTextualRecordSource(
            [
                {"id": "same", "name": "Dish One"},
                {"id": "same", "name": "Dish Duplicate"},
            ]
        ),
        CapturingTextualRecordMapper(),
    )

    batch = importer.import_records(
        source(),
        TextualImportOptions(deduplicate_records=False),
    )

    assert len(batch.records) == 2
    assert batch.duplicate_record_count == 0


def test_counts_skipped_and_invalid_records() -> None:
    importer = DefaultTextualKnowledgeImporter(
        InMemoryTextualRecordSource(
            [
                "not a mapping",
                {"id": "skip", "name": "Skipped", "skip": True},
                {"id": "invalid"},
                {"id": "valid", "name": "Valid Dish"},
            ]
        ),
        CapturingTextualRecordMapper(),
    )

    batch = importer.import_records(source(), TextualImportOptions())

    assert tuple(item.external_id for item in batch.records) == ("valid",)
    assert batch.skipped_record_count == 3
