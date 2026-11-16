from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from cuisine_assistant.domain.ports import IngredientExtractor
from cuisine_assistant.importing.models import (
    ImportedCuisine,
    ImportedDishRecord,
    ImportedIngredientMention,
    ImportedPlace,
    TextualImportOptions,
    TextualImportSource,
)
from cuisine_assistant.importing.ports import TextualRecordMapper


class WorldCuisinesRecordMapper(TextualRecordMapper):
    """Maps World Cuisines food-kb rows into normalized image-free dish records."""

    _IMAGE_FIELD_PATTERN = re.compile(
        r"^(image|images|thumbnail)(\d+)?(_url|_license|_file|_path)?$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        ingredient_extractor: IngredientExtractor | None = None,
    ) -> None:
        self._ingredient_extractor = ingredient_extractor

    def map_record(
        self,
        raw_record: Mapping[str, Any],
        source: TextualImportSource,
        options: TextualImportOptions,
    ) -> ImportedDishRecord | None:
        """Maps one food-kb row while ignoring every image-related field."""
        sanitized = {
            key: value
            for key, value in raw_record.items()
            if not self._is_image_field(key)
        }
        name = self._optional_text(sanitized.get("name"))
        if not name:
            return None

        cuisines = self._map_cuisines(sanitized)
        places = self._map_places(sanitized)
        description = (
            self._optional_text(sanitized.get("text_description"))
            if options.include_descriptions
            else None
        )
        aliases = (
            self._map_aliases(sanitized.get("alias"))
            if options.include_aliases
            else ()
        )
        categories = self._unique_texts(
            self._as_texts(sanitized.get("coarse_categories"))
            + self._as_texts(sanitized.get("fine_categories"))
        )
        ingredient_mentions = self._extract_ingredients(
            description,
            options,
        )
        external_id = self._external_id(name, cuisines)

        return ImportedDishRecord(
            external_id=external_id,
            name=name,
            description=description,
            aliases=aliases,
            categories=categories,
            cuisines=cuisines,
            places=places,
            ingredients=ingredient_mentions,
            source_url=(
                "https://huggingface.co/datasets/"
                f"{source.source_id}"
            ),
            source_license=source.license_name,
        )

    def _map_cuisines(
        self,
        record: Mapping[str, Any],
    ) -> tuple[ImportedCuisine, ...]:
        names = self._unique_texts(
            self._as_texts(record.get("cuisines"))
            + self._as_texts(record.get("associated_cuisines"))
        )
        country_ids = tuple(
            self._slugify(name)
            for name in self._as_texts(record.get("countries"))
        )
        origin_place_id = country_ids[0] if country_ids else None
        return tuple(
            ImportedCuisine(
                cuisine_id=self._slugify(name),
                name=self._cuisine_name(name),
                ontology_id=self._ontology_id(name, "Cuisine"),
                aliases=self._cuisine_aliases(name),
                origin_place_id=origin_place_id,
            )
            for name in names
        )

    def _map_places(
        self,
        record: Mapping[str, Any],
    ) -> tuple[ImportedPlace, ...]:
        places: list[ImportedPlace] = []
        seen: set[str] = set()
        parent_id: str | None = None

        regions = [
            self._optional_text(record.get(f"region{index}"))
            for index in range(5, 0, -1)
        ]
        for region in regions:
            if not region:
                continue
            place_id = self._slugify(region)
            if place_id not in seen:
                places.append(
                    ImportedPlace(
                        place_id=place_id,
                        name=region,
                        place_type="Region",
                        parent_place_id=parent_id,
                    )
                )
                seen.add(place_id)
            parent_id = place_id

        for country in self._as_texts(record.get("countries")):
            place_id = self._slugify(country)
            if place_id not in seen:
                places.append(
                    ImportedPlace(
                        place_id=place_id,
                        name=country,
                        place_type="Country",
                        parent_place_id=parent_id,
                    )
                )
                seen.add(place_id)

        for area in self._as_texts(record.get("area")):
            place_id = self._slugify(area)
            if place_id not in seen:
                places.append(
                    ImportedPlace(
                        place_id=place_id,
                        name=area,
                        place_type="Area",
                    )
                )
                seen.add(place_id)
        return tuple(places)

    def _extract_ingredients(
        self,
        description: str | None,
        options: TextualImportOptions,
    ) -> tuple[ImportedIngredientMention, ...]:
        if (
            not description
            or not options.extract_ingredients_from_description
            or self._ingredient_extractor is None
        ):
            return ()

        mentions: list[ImportedIngredientMention] = []
        seen: set[str] = set()
        for extracted in self._ingredient_extractor.extract(description):
            canonical = extracted.matched_ingredient_id or extracted.normalized_text
            canonical = self._slugify(canonical)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            mentions.append(
                ImportedIngredientMention(
                    canonical_text=canonical,
                    surface_text=extracted.surface_form,
                    confidence=extracted.match_score,
                )
            )
        return tuple(mentions)

    @classmethod
    def _map_aliases(cls, value: Any) -> tuple[str, ...]:
        aliases: list[str] = []
        if isinstance(value, str):
            aliases.append(value)
        elif isinstance(value, Mapping):
            aliases.extend(str(key) for key in value if key != "no_alias")
        elif isinstance(value, Iterable):
            for item in value:
                if isinstance(item, str):
                    aliases.append(item)
                elif isinstance(item, Mapping):
                    aliases.extend(
                        str(key) for key in item if key != "no_alias"
                    )
        return cls._unique_texts(aliases)

    @classmethod
    def _as_texts(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = cls._optional_text(value)
            return [text] if text else []
        if isinstance(value, Iterable) and not isinstance(value, Mapping):
            result = []
            for item in value:
                text = cls._optional_text(item)
                if text:
                    result.append(text)
            return result
        return []

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return " ".join(text.split())

    @classmethod
    def _unique_texts(cls, values: Iterable[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = cls._optional_text(value)
            if not text:
                continue
            key = text.casefold()
            if key not in seen:
                seen.add(key)
                result.append(text)
        return tuple(result)

    @classmethod
    def _external_id(
        cls,
        name: str,
        cuisines: tuple[ImportedCuisine, ...],
    ) -> str:
        cuisine_part = cuisines[0].cuisine_id if cuisines else "uncategorized"
        return f"{cls._slugify(name)}__{cuisine_part}"

    @staticmethod
    def _cuisine_name(name: str) -> str:
        return name if name.lower().endswith("cuisine") else f"{name} cuisine"

    @classmethod
    def _cuisine_aliases(cls, name: str) -> tuple[str, ...]:
        base = name.strip()
        return cls._unique_texts(
            (base, f"{base} food", cls._cuisine_name(base))
        )

    @classmethod
    def _ontology_id(cls, value: str, suffix: str = "") -> str:
        words = re.findall(r"[a-zA-Z0-9]+", cls._ascii(value))
        identifier = "".join(word[:1].upper() + word[1:] for word in words)
        if suffix and not identifier.lower().endswith(suffix.lower()):
            identifier += suffix
        return identifier or suffix

    @classmethod
    def _slugify(cls, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", cls._ascii(value).lower())
        return normalized.strip("_")

    @staticmethod
    def _ascii(value: str) -> str:
        return unicodedata.normalize("NFKD", value).encode(
            "ascii", "ignore"
        ).decode("ascii")

    @classmethod
    def _is_image_field(cls, key: object) -> bool:
        normalized = str(key).strip().lower().replace("-", "_")
        return bool(cls._IMAGE_FIELD_PATTERN.match(normalized))
