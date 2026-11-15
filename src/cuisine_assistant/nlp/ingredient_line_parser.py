from __future__ import annotations

import re

from cuisine_assistant.domain.models import IngredientMention
from cuisine_assistant.domain.ports import IngredientMentionParser, TextNormalizer


class EnglishIngredientLineParser(IngredientMentionParser):
    """Parses structured English ingredient-list lines into clean mentions."""

    _LINE_SPLIT = re.compile(r"[\n;]+")
    _LEADING_BULLET = re.compile(r"^\s*[-*•]\s*")
    _QUANTITY = re.compile(
        r"^\s*(?:\d+\s*\/\s*\d+|\d+(?:[.,]\d+)?|[¼½¾⅓⅔⅛⅜⅝⅞])"
        r"(?:\s*[-–]\s*(?:\d+(?:[.,]\d+)?|\d+\s*\/\s*\d+))?\s*"
    )
    _UNITS = re.compile(
        r"^(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|grams?|g|kilograms?|kg|"
        r"millilit(?:er|re)s?|ml|lit(?:er|re)s?|l|ounces?|oz|pounds?|lb|cloves?|cans?|"
        r"packages?|pinches?|slices?|pieces?|bunch(?:es)?|heads?)\b\.?\s*",
        re.IGNORECASE,
    )
    _LEADING_CONNECTOR = re.compile(r"^(?:of\s+)", re.IGNORECASE)
    _TRAILING_NOTE = re.compile(r"\s*(?:,|\(|\[).*?$")
    _PREPARATION_WORDS = {
        "fresh", "freshly", "chopped", "finely", "roughly", "diced", "sliced",
        "minced", "ground", "crushed", "grated", "peeled", "seeded", "drained",
        "rinsed", "melted", "softened", "divided", "optional", "large", "medium",
        "small", "thinly", "coarsely",
    }

    def __init__(self, normalizer: TextNormalizer) -> None:
        self._normalizer = normalizer

    def parse(self, text: str) -> list[IngredientMention]:
        mentions: list[IngredientMention] = []
        cursor = 0
        for raw_line in self._LINE_SPLIT.split(text):
            if not raw_line.strip():
                cursor += len(raw_line) + 1
                continue
            cleaned = self._clean_line(raw_line)
            if cleaned:
                start = text.find(raw_line, cursor)
                end = start + len(raw_line)
                mentions.append(
                    IngredientMention(
                        surface_form=raw_line.strip(),
                        normalized_text=cleaned,
                        start_char=max(start, 0),
                        end_char=max(end, 0),
                    )
                )
            cursor += len(raw_line) + 1
        return mentions

    def _clean_line(self, line: str) -> str:
        value = self._LEADING_BULLET.sub("", line)
        value = self._QUANTITY.sub("", value)
        value = self._UNITS.sub("", value)
        value = self._LEADING_CONNECTOR.sub("", value)
        value = self._TRAILING_NOTE.sub("", value)
        tokens = self._normalizer.normalize(value).split()
        while tokens and tokens[0] in self._PREPARATION_WORDS:
            tokens.pop(0)
        return " ".join(tokens)
