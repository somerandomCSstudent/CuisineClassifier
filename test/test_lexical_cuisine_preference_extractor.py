from __future__ import annotations

from collections.abc import Iterable

from cuisine_assistant.domain.models import Cuisine
from cuisine_assistant.domain.recommendation_ports import CuisinePreferenceCatalog
from cuisine_assistant.nlp.lexical_cuisine_preference_extractor import (
    LexicalCuisinePreferenceExtractor,
)


class InMemoryCuisinePreferenceCatalog(CuisinePreferenceCatalog):
    """Provides deterministic cuisine aliases for lexical extractor tests."""

    def __init__(self) -> None:
        hungarian = Cuisine(
            cuisine_id="hungarian",
            name="Hungarian cuisine",
            ontology_id="HungarianCuisine",
            cuisine_group="EuropeanCuisine",
        )
        italian = Cuisine(
            cuisine_id="italian",
            name="Italian cuisine",
            ontology_id="ItalianCuisine",
            cuisine_group="EuropeanCuisine",
        )
        self._aliases = {
            "hungarian": hungarian,
            "hungarian food": hungarian,
            "hungarian cuisine": hungarian,
            "italian": italian,
            "italian food": italian,
            "italian cuisine": italian,
        }

    def find_by_alias(self, normalized_alias: str) -> Cuisine | None:
        return self._aliases.get(normalized_alias)

    def aliases(self) -> Iterable[str]:
        return self._aliases.keys()


def create_extractor() -> LexicalCuisinePreferenceExtractor:
    """Creates the lexical cuisine preference extractor for tests."""
    return LexicalCuisinePreferenceExtractor(
        InMemoryCuisinePreferenceCatalog()
    )


def test_extracts_explicit_cuisine_preference() -> None:
    preference = create_extractor().extract(
        "I have chicken and paprika and I would like Hungarian food."
    )

    assert preference is not None
    assert preference.cuisine_id == "hungarian"
    assert preference.ontology_id == "HungarianCuisine"
    assert preference.source_text == "Hungarian food"
    assert preference.match_score == 1.0
    assert preference.strict is False


def test_prefers_longest_alias() -> None:
    preference = create_extractor().extract(
        "I would like Hungarian cuisine tonight."
    )

    assert preference is not None
    assert preference.source_text == "Hungarian cuisine"


def test_is_case_insensitive() -> None:
    preference = create_extractor().extract(
        "Please recommend ITALIAN FOOD."
    )

    assert preference is not None
    assert preference.cuisine_id == "italian"


def test_avoids_partial_word_matches() -> None:
    preference = create_extractor().extract(
        "This ingredient was prepared in an Italianate style."
    )

    assert preference is None


def test_returns_none_when_no_cuisine_is_mentioned() -> None:
    preference = create_extractor().extract(
        "I have chicken, paprika, onion and sour cream."
    )

    assert preference is None


def test_detects_strict_cuisine_request() -> None:
    preference = create_extractor().extract(
        "Only recommend Hungarian dishes."
    )

    assert preference is not None
    assert preference.cuisine_id == "hungarian"
    assert preference.strict is True


def test_empty_input_returns_none() -> None:
    assert create_extractor().extract("   ") is None
