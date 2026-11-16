from __future__ import annotations

import pytest

from cuisine_assistant.importing import (
    ImportedCuisine,
    ImportedDishRecord,
    ImportedIngredientMention,
    ImportedPlace,
    ImportSourceType,
    TextualImportBatch,
    TextualImportOptions,
    TextualImportResult,
    TextualImportSource,
)


def create_record() -> ImportedDishRecord:
    """Creates one image-free imported dish fixture."""
    return ImportedDishRecord(
        external_id="rawon",
        name="Rawon",
        description="A beef soup with garlic and ginger.",
        aliases=("Nasi Rawon",),
        categories=("Soup",),
        cuisines=(
            ImportedCuisine(
                cuisine_id="indonesian",
                name="Indonesian cuisine",
                ontology_id="IndonesianCuisine",
                aliases=("Indonesian food",),
                origin_place_id="indonesia",
            ),
        ),
        places=(
            ImportedPlace(
                place_id="indonesia",
                name="Indonesia",
                place_type="Country",
                parent_place_id="south_eastern_asia",
            ),
        ),
        ingredients=(
            ImportedIngredientMention("beef", "beef", 1.0),
            ImportedIngredientMention("garlic", "garlic", 0.95),
            ImportedIngredientMention("ginger", "ginger", 0.95),
        ),
        source_url="https://example.org/rawon",
        source_license="CC BY-SA 4.0",
    )


def test_creates_bounded_textual_import_batch() -> None:
    source = TextualImportSource(
        ImportSourceType.HUGGING_FACE,
        "worldcuisines/food-kb",
        "food-kb",
        revision="main",
        license_name="CC BY-SA 4.0",
    )
    options = TextualImportOptions(maximum_records=500)
    batch = TextualImportBatch(source, options, (create_record(),))
    result = TextualImportResult(
        batch=batch,
        imported_record_count=1,
        cuisine_count=1,
        place_count=1,
        ingredient_mention_count=3,
    )

    assert result.imported_record_count == 1
    assert batch.options.download_images is False
    assert batch.options.process_images is False
    assert batch.options.create_image_embeddings is False


def test_rejects_all_image_operations() -> None:
    with pytest.raises(ValueError, match="Image download"):
        TextualImportOptions(download_images=True)
    with pytest.raises(ValueError, match="Image processing"):
        TextualImportOptions(process_images=True)
    with pytest.raises(ValueError, match="Image embeddings"):
        TextualImportOptions(create_image_embeddings=True)


def test_rejects_image_references_in_imported_record() -> None:
    with pytest.raises(ValueError, match="Image references"):
        ImportedDishRecord(
            external_id="dish",
            name="Dish",
            description="Textual description",
            aliases=(),
            categories=(),
            cuisines=(),
            places=(),
            ingredients=(),
            source_url="https://example.org/dish.jpg",
        )


def test_rejects_more_records_than_configured_limit() -> None:
    source = TextualImportSource(
        ImportSourceType.LOCAL_JSON,
        "local.json",
        "local",
    )
    options = TextualImportOptions(maximum_records=1)

    with pytest.raises(ValueError, match="exceed maximum_records"):
        TextualImportBatch(
            source,
            options,
            (create_record(), create_record()),
        )


def test_result_count_must_match_batch_size() -> None:
    source = TextualImportSource(
        ImportSourceType.LOCAL_CSV,
        "local.csv",
        "local",
    )
    batch = TextualImportBatch(
        source,
        TextualImportOptions(maximum_records=10),
        (create_record(),),
    )

    with pytest.raises(ValueError, match="must match"):
        TextualImportResult(
            batch=batch,
            imported_record_count=2,
            cuisine_count=1,
            place_count=1,
            ingredient_mention_count=3,
        )
