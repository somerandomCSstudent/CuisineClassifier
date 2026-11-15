from pathlib import Path

from cuisine_assistant.application.extract_ingredients import ExtractIngredientsUseCase
from cuisine_assistant.nlp.alias_extractor import AliasIngredientExtractor
from cuisine_assistant.nlp.normalizer import EnglishTextNormalizer
from cuisine_assistant.search.lexical_search import CsvLexicalEntitySearch


def build_use_case() -> ExtractIngredientsUseCase:
    root = Path(__file__).resolve().parents[1]
    normalizer = EnglishTextNormalizer()
    search = CsvLexicalEntitySearch(
        root / "data" / "processed" / "ingredient_aliases.csv",
        normalizer,
    )
    return ExtractIngredientsUseCase(AliasIngredientExtractor(normalizer, search))


def test_extracts_longest_aliases() -> None:
    result = build_use_case().execute(
        "Fresh tomatoes, mozzarella and basil with extra virgin olive oil."
    )
    assert [item.matched_ingredient_id for item in result] == [
        "tomato", "mozzarella", "basil", "olive_oil"
    ]
    assert result[0].surface_form == "fresh tomatoes"
    assert result[-1].surface_form == "extra virgin olive oil"


def test_empty_input_returns_empty_list() -> None:
    assert build_use_case().execute("   ") == []
