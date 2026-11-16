from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ImportSourceType(StrEnum):
    """Defines supported external textual knowledge sources."""

    HUGGING_FACE = "hugging_face"
    LOCAL_JSON = "local_json"
    LOCAL_CSV = "local_csv"


@dataclass(frozen=True)
class TextualImportSource:
    """Describes a textual or structured source used to build a knowledge base."""

    source_type: ImportSourceType
    source_id: str
    dataset_name: str
    split: str = "train"
    language: str = "en"
    revision: str | None = None
    license_name: str | None = None

    def __post_init__(self) -> None:
        _require_text("source_id", self.source_id)
        _require_text("dataset_name", self.dataset_name)
        _require_text("split", self.split)
        _require_text("language", self.language)


@dataclass(frozen=True)
class TextualImportOptions:
    """Controls record limits and text-only behavior for one import operation."""

    maximum_records: int = 500
    extract_ingredients_from_description: bool = True
    include_descriptions: bool = True
    include_aliases: bool = True
    deduplicate_records: bool = True
    download_images: bool = False
    process_images: bool = False
    create_image_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.maximum_records < 1:
            raise ValueError("maximum_records must be at least 1")
        if self.download_images:
            raise ValueError("Image download is not allowed")
        if self.process_images:
            raise ValueError("Image processing is not allowed")
        if self.create_image_embeddings:
            raise ValueError("Image embeddings are not allowed")


@dataclass(frozen=True)
class ImportedPlace:
    """Represents a normalized place extracted from an external textual record."""

    place_id: str
    name: str
    place_type: str
    parent_place_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("place_id", self.place_id)
        _require_text("name", self.name)
        _require_text("place_type", self.place_type)


@dataclass(frozen=True)
class ImportedCuisine:
    """Represents a normalized cuisine extracted from an external textual record."""

    cuisine_id: str
    name: str
    ontology_id: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    origin_place_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("cuisine_id", self.cuisine_id)
        _require_text("name", self.name)
        _require_text("ontology_id", self.ontology_id)


@dataclass(frozen=True)
class ImportedIngredientMention:
    """Represents one normalized ingredient mention derived from textual content."""

    canonical_text: str
    surface_text: str
    confidence: float

    def __post_init__(self) -> None:
        _require_text("canonical_text", self.canonical_text)
        _require_text("surface_text", self.surface_text)
        _require_score("confidence", self.confidence)


@dataclass(frozen=True)
class ImportedDishRecord:
    """Represents one image-free normalized dish record ready for bundle building."""

    external_id: str
    name: str
    description: str | None
    aliases: tuple[str, ...]
    categories: tuple[str, ...]
    cuisines: tuple[ImportedCuisine, ...]
    places: tuple[ImportedPlace, ...]
    ingredients: tuple[ImportedIngredientMention, ...]
    source_url: str | None = None
    source_license: str | None = None

    def __post_init__(self) -> None:
        _require_text("external_id", self.external_id)
        _require_text("name", self.name)
        _reject_image_content(self)


@dataclass(frozen=True)
class TextualImportBatch:
    """Contains normalized text-only records produced by one import operation."""

    source: TextualImportSource
    options: TextualImportOptions
    records: tuple[ImportedDishRecord, ...]
    skipped_record_count: int = 0
    duplicate_record_count: int = 0

    def __post_init__(self) -> None:
        if len(self.records) > self.options.maximum_records:
            raise ValueError("Imported records exceed maximum_records")
        if self.skipped_record_count < 0:
            raise ValueError("skipped_record_count must not be negative")
        if self.duplicate_record_count < 0:
            raise ValueError("duplicate_record_count must not be negative")


@dataclass(frozen=True)
class TextualImportResult:
    """Summarizes a completed textual import before knowledge base generation."""

    batch: TextualImportBatch
    imported_record_count: int
    cuisine_count: int
    place_count: int
    ingredient_mention_count: int

    def __post_init__(self) -> None:
        expected = len(self.batch.records)
        if self.imported_record_count != expected:
            raise ValueError(
                "imported_record_count must match the number of batch records"
            )
        for field_name in (
            "cuisine_count",
            "place_count",
            "ingredient_mention_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")


def _require_text(field_name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_score(field_name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


def _reject_image_content(record: ImportedDishRecord) -> None:
    values = (
        record.description,
        record.source_url,
        *record.aliases,
        *record.categories,
    )
    forbidden_extensions = (
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".svg",
        ".webp",
    )
    for value in values:
        if not value:
            continue
        normalized = value.lower().split("?", 1)[0]
        if normalized.endswith(forbidden_extensions):
            raise ValueError("Image references are not allowed in imported records")
