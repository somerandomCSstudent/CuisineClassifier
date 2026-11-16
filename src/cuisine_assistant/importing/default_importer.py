from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cuisine_assistant.importing.models import (
    ImportedDishRecord,
    TextualImportBatch,
    TextualImportOptions,
    TextualImportSource,
)
from cuisine_assistant.importing.ports import (
    TextualKnowledgeImporter,
    TextualRecordMapper,
    TextualRecordSource,
)


class DefaultTextualKnowledgeImporter(TextualKnowledgeImporter):
    """Imports bounded, deduplicated, image-free records through source and mapper ports."""

    _FORBIDDEN_IMAGE_KEYS = {
        "image",
        "images",
        "image_url",
        "image_urls",
        "image_file",
        "image_files",
        "image_path",
        "image_paths",
        "image_embedding",
        "image_embeddings",
        "thumbnail",
        "thumbnails",
    }

    def __init__(
        self,
        record_source: TextualRecordSource,
        record_mapper: TextualRecordMapper,
    ) -> None:
        self._record_source = record_source
        self._record_mapper = record_mapper

    def import_records(
        self,
        source: TextualImportSource,
        options: TextualImportOptions,
    ) -> TextualImportBatch:
        """Reads, sanitizes, maps, deduplicates, and limits external records."""
        imported: list[ImportedDishRecord] = []
        seen_keys: set[str] = set()
        skipped_count = 0
        duplicate_count = 0

        for raw_record in self._record_source.read(source, options):
            if len(imported) >= options.maximum_records:
                break
            if not isinstance(raw_record, Mapping):
                skipped_count += 1
                continue

            sanitized_record = self._remove_image_fields(raw_record)
            try:
                mapped = self._record_mapper.map_record(
                    sanitized_record,
                    source,
                    options,
                )
            except (KeyError, TypeError, ValueError):
                skipped_count += 1
                continue

            if mapped is None:
                skipped_count += 1
                continue

            deduplication_key = self._deduplication_key(mapped)
            if options.deduplicate_records and deduplication_key in seen_keys:
                duplicate_count += 1
                continue

            seen_keys.add(deduplication_key)
            imported.append(mapped)

        return TextualImportBatch(
            source=source,
            options=options,
            records=tuple(imported),
            skipped_record_count=skipped_count,
            duplicate_record_count=duplicate_count,
        )

    @classmethod
    def _remove_image_fields(
        cls,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            key: cls._sanitize_value(child)
            for key, child in value.items()
            if cls._normalize_key(key) not in cls._FORBIDDEN_IMAGE_KEYS
        }

    @classmethod
    def _sanitize_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls._remove_image_fields(value)
        if isinstance(value, list):
            return [cls._sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._sanitize_value(item) for item in value)
        return value

    @staticmethod
    def _normalize_key(key: object) -> str:
        return str(key).strip().lower().replace("-", "_")

    @staticmethod
    def _deduplication_key(record: ImportedDishRecord) -> str:
        external_id = record.external_id.strip().lower()
        if external_id:
            return f"id:{external_id}"
        normalized_name = " ".join(record.name.lower().split())
        cuisine_ids = ",".join(
            sorted(cuisine.cuisine_id for cuisine in record.cuisines)
        )
        return f"name:{normalized_name}|cuisines:{cuisine_ids}"
