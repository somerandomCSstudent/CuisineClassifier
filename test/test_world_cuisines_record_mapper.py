from __future__ import annotations

from cuisine_assistant.importing import (
    ImportSourceType,
    TextualImportOptions,
    TextualImportSource,
    WorldCuisinesRecordMapper,
)


def source() -> TextualImportSource:
    """Creates the World Cuisines source metadata used by mapper tests."""
    return TextualImportSource(
        ImportSourceType.HUGGING_FACE,
        "worldcuisines/food-kb",
        "default",
        split="main",
        license_name="CC BY-SA 4.0",
    )


def rawon_record() -> dict:
    """Creates a representative food-kb row without relying on network access."""
    return {
        "name": "Rawon",
        "alias": [{"ꦫꦮꦺꦴꦤ꧀": "Javanese"}],
        "coarse_categories": ["soup"],
        "fine_categories": ["beef soup"],
        "cuisines": ["Indonesian"],
        "associated_cuisines": ["Indonesian", "Singaporean"],
        "area": ["Surabaya", "Malang", "East Java"],
        "countries": ["Indonesia"],
        "region1": "South Eastern Asia",
        "region2": "nan",
        "region3": None,
        "region4": None,
        "region5": None,
        "text_description": (
            "The soup is composed of garlic, shallots, ginger and turmeric."
        ),
        "image1": {"bytes": "must be ignored"},
        "image1_url": "https://example.invalid/rawon.jpg",
        "image1_license": "CC BY-SA 4.0",
    }


def test_maps_world_cuisines_schema_to_normalized_record() -> None:
    """Verifies names, aliases, categories, cuisines, places, and source metadata."""
    mapped = WorldCuisinesRecordMapper().map_record(
        rawon_record(),
        source(),
        TextualImportOptions(maximum_records=500),
    )

    assert mapped is not None
    assert mapped.external_id == "rawon__indonesian"
    assert mapped.name == "Rawon"
    assert mapped.aliases == ("ꦫꦮꦺꦴꦤ꧀",)
    assert mapped.categories == ("soup", "beef soup")
    assert tuple(item.cuisine_id for item in mapped.cuisines) == (
        "indonesian",
        "singaporean",
    )
    assert mapped.cuisines[0].ontology_id == "IndonesianCuisine"
    assert mapped.cuisines[0].origin_place_id == "indonesia"
    assert any(item.place_id == "indonesia" for item in mapped.places)
    assert any(item.place_id == "south_eastern_asia" for item in mapped.places)
    assert mapped.source_url == (
        "https://huggingface.co/datasets/worldcuisines/food-kb"
    )
    assert mapped.source_license == "CC BY-SA 4.0"


def test_ignores_all_image_columns_even_when_mapper_is_used_directly() -> None:
    """Verifies image fields cannot leak into the normalized record."""
    record = rawon_record()
    record["image8_url"] = "https://example.invalid/other.png"
    record["thumbnail"] = "https://example.invalid/thumb.webp"

    mapped = WorldCuisinesRecordMapper().map_record(
        record,
        source(),
        TextualImportOptions(),
    )

    assert mapped is not None
    assert mapped.source_url.endswith("worldcuisines/food-kb")
    assert ".jpg" not in str(mapped)
    assert ".png" not in str(mapped)
    assert ".webp" not in str(mapped)


def test_returns_none_when_name_is_missing() -> None:
    """Verifies unnamed rows are skipped by the default importer."""
    assert (
        WorldCuisinesRecordMapper().map_record(
            {"text_description": "No name"},
            source(),
            TextualImportOptions(),
        )
        is None
    )


def test_respects_alias_and_description_options() -> None:
    """Verifies optional text fields can be excluded from generated bundles."""
    mapped = WorldCuisinesRecordMapper().map_record(
        rawon_record(),
        source(),
        TextualImportOptions(
            include_aliases=False,
            include_descriptions=False,
        ),
    )

    assert mapped is not None
    assert mapped.aliases == ()
    assert mapped.description is None


def test_handles_no_alias_marker_and_nan_regions() -> None:
    """Verifies dataset sentinel values do not become aliases or places."""
    record = rawon_record()
    record["alias"] = [{"no_alias": True}]
    record["region1"] = "nan"

    mapped = WorldCuisinesRecordMapper().map_record(
        record,
        source(),
        TextualImportOptions(),
    )

    assert mapped is not None
    assert mapped.aliases == ()
    assert all(item.name.lower() != "nan" for item in mapped.places)
