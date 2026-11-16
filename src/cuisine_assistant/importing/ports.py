from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from cuisine_assistant.importing.models import (
    ImportedDishRecord,
    TextualImportBatch,
    TextualImportOptions,
    TextualImportSource,
)


class TextualRecordSource(ABC):
    """Streams raw structured records without retrieving any image content."""

    @abstractmethod
    def read(
        self,
        source: TextualImportSource,
        options: TextualImportOptions,
    ) -> Iterable[Mapping[str, Any]]:
        """Yields raw text and metadata records up to the configured limit."""
        raise NotImplementedError


class TextualRecordMapper(ABC):
    """Maps one external structured record into the normalized import model."""

    @abstractmethod
    def map_record(
        self,
        raw_record: Mapping[str, Any],
        source: TextualImportSource,
        options: TextualImportOptions,
    ) -> ImportedDishRecord | None:
        """Returns a normalized record or None when the input must be skipped."""
        raise NotImplementedError


class TextualKnowledgeImporter(ABC):
    """Imports bounded text-only records for later knowledge base generation."""

    @abstractmethod
    def import_records(
        self,
        source: TextualImportSource,
        options: TextualImportOptions,
    ) -> TextualImportBatch:
        """Returns normalized image-free records from one supported source."""
        raise NotImplementedError
