from __future__ import annotations

import re

from cuisine_assistant.domain.models import ExtractedIngredient
from cuisine_assistant.domain.ports import EntitySearch, IngredientExtractor, TextNormalizer


class AliasIngredientExtractor(IngredientExtractor):
    def __init__(self, normalizer: TextNormalizer, search: EntitySearch) -> None:
        self._normalizer = normalizer
        self._search = search
        self._pattern = self._build_pattern(search.known_terms())

    @staticmethod
    def _build_pattern(terms) -> re.Pattern[str]:
        ordered = sorted(set(terms), key=lambda value: (-len(value.split()), -len(value)))
        alternatives = "|".join(re.escape(term) for term in ordered)
        return re.compile(
            rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])",
            re.IGNORECASE,
        ) if alternatives else re.compile(r"(?!x)x")

    def extract(self, recipe_text: str) -> list[ExtractedIngredient]:
        normalized_recipe = self._normalizer.normalize(recipe_text)
        results: list[ExtractedIngredient] = []

        for match in self._pattern.finditer(normalized_recipe):
            surface = match.group(0)
            normalized = self._normalizer.normalize(surface)
            candidates = self._search.find(normalized)
            if not candidates:
                continue
            best = candidates[0]
            results.append(
                ExtractedIngredient(
                    surface_form=surface,
                    normalized_text=normalized,
                    start_char=match.start(),
                    end_char=match.end(),
                    matched_ingredient_id=best.entity_id,
                    match_method=best.method,
                    match_score=best.score,
                    candidate_ids=[candidate.entity_id for candidate in candidates],
                )
            )
        return results
