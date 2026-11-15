from __future__ import annotations

from cuisine_assistant.domain.models import SearchCandidate
from cuisine_assistant.domain.ports import EntitySearch
from cuisine_assistant.evaluation.metrics import calculate_metrics
from cuisine_assistant.nlp.ingredient_line_parser import EnglishIngredientLineParser
from cuisine_assistant.nlp.normalizer import EnglishTextNormalizer
from cuisine_assistant.nlp.spacy_extractor import SpacyPhraseIngredientExtractor
from cuisine_assistant.nlp.structured_extractor import StructuredIngredientExtractor


class InMemorySearch(EntitySearch):
    def __init__(self) -> None:
        self._values = {
            "tomato": "tomato",
            "tomatoes": "tomato",
            "fresh tomatoes": "tomato",
            "tomato paste": "tomato_paste",
            "olive oil": "olive_oil",
            "extra virgin olive oil": "olive_oil",
            "chickpeas": "chickpea",
            "garbanzo beans": "chickpea",
            "onion": "onion",
        }

    def find(self, normalized_text: str) -> list[SearchCandidate]:
        entity_id = self._values.get(normalized_text)
        if entity_id is None:
            return []
        return [SearchCandidate(entity_id, normalized_text, "alias", 1.0)]

    def known_terms(self):
        return self._values.keys()


def build_structured_extractor():
    normalizer = EnglishTextNormalizer()
    return StructuredIngredientExtractor(
        EnglishIngredientLineParser(normalizer),
        InMemorySearch(),
    )


def test_parser_removes_quantity_unit_and_preparation() -> None:
    result = build_structured_extractor().extract("2 cups freshly chopped tomatoes")
    assert result[0].normalized_text == "tomatoes"
    assert result[0].matched_ingredient_id == "tomato"


def test_parser_handles_fraction_and_hyphenated_unit_line() -> None:
    result = build_structured_extractor().extract("1/2 tbsp extra virgin olive oil")
    assert result[0].normalized_text == "extra virgin olive oil"
    assert result[0].matched_ingredient_id == "olive_oil"


def test_parser_preserves_unknown_ingredient() -> None:
    result = build_structured_extractor().extract("2 tbsp gochujang")
    assert result[0].normalized_text == "gochujang"
    assert result[0].matched_ingredient_id is None
    assert result[0].match_method == "unmatched"
    assert result[0].match_score == 0.0


def test_parser_handles_multiple_lines() -> None:
    result = build_structured_extractor().extract("400 g chickpeas\n1 medium onion")
    assert [item.matched_ingredient_id for item in result] == ["chickpea", "onion"]


def test_spacy_prefers_longest_overlapping_phrase() -> None:
    normalizer = EnglishTextNormalizer()
    extractor = SpacyPhraseIngredientExtractor(normalizer, InMemorySearch())
    result = extractor.extract("Use tomato paste, not tomato.")
    assert [item.matched_ingredient_id for item in result] == ["tomato_paste", "tomato"]


def test_spacy_is_case_insensitive() -> None:
    normalizer = EnglishTextNormalizer()
    extractor = SpacyPhraseIngredientExtractor(normalizer, InMemorySearch())
    result = extractor.extract("Add EXTRA VIRGIN OLIVE OIL.")
    assert result[0].matched_ingredient_id == "olive_oil"


def test_metrics_are_calculated_correctly() -> None:
    metrics = calculate_metrics({"a", "b", "c"}, {"b", "c", "d"})
    assert metrics.true_positive == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert round(metrics.f1, 3) == 0.667
