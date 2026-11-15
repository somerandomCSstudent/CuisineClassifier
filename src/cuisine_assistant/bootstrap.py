from __future__ import annotations

from pathlib import Path

from cuisine_assistant.application.extract_ingredients import ExtractIngredientsUseCase
from cuisine_assistant.cli.console import ConsoleApplication
from cuisine_assistant.nlp.alias_extractor import AliasIngredientExtractor
from cuisine_assistant.nlp.normalizer import EnglishTextNormalizer
from cuisine_assistant.search.lexical_search import CsvLexicalEntitySearch


def build_application(project_root: Path) -> ConsoleApplication:
    normalizer = EnglishTextNormalizer()
    search = CsvLexicalEntitySearch(
        project_root / "data" / "processed" / "ingredient_aliases.csv",
        normalizer,
    )
    extractor = AliasIngredientExtractor(normalizer, search)
    use_case = ExtractIngredientsUseCase(extractor)
    return ConsoleApplication(use_case)
