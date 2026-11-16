from __future__ import annotations

import re

from cuisine_assistant.domain.recommendation_models import CuisinePreference
from cuisine_assistant.domain.recommendation_ports import (
    CuisinePreferenceCatalog,
    CuisinePreferenceExtractor,
)


class LexicalCuisinePreferenceExtractor(CuisinePreferenceExtractor):
    """Extracts the longest explicit cuisine alias from English user input."""

    _STRICT_PATTERNS = (
        re.compile(r"\bonly\b", re.IGNORECASE),
        re.compile(r"\bstrictly\b", re.IGNORECASE),
        re.compile(r"\bmust be\b", re.IGNORECASE),
        re.compile(r"\bnothing but\b", re.IGNORECASE),
    )

    def __init__(self, catalog: CuisinePreferenceCatalog) -> None:
        self._catalog = catalog
        self._patterns = self._build_patterns(catalog)

    def extract(self, text: str) -> CuisinePreference | None:
        if not text or not text.strip():
            return None

        matches: list[tuple[int, int, str, str]] = []
        for alias, pattern in self._patterns:
            for match in pattern.finditer(text):
                matches.append(
                    (
                        match.end() - match.start(),
                        match.start(),
                        alias,
                        match.group(0),
                    )
                )

        if not matches:
            return None

        matches.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, _, alias, source_text = matches[0]
        cuisine = self._catalog.find_by_alias(alias)
        if cuisine is None:
            return None

        return CuisinePreference(
            cuisine_id=cuisine.cuisine_id,
            ontology_id=cuisine.ontology_id,
            source_text=source_text,
            match_score=1.0,
            strict=self._is_strict_request(text),
        )

    @classmethod
    def _build_patterns(
        cls,
        catalog: CuisinePreferenceCatalog,
    ) -> tuple[tuple[str, re.Pattern[str]], ...]:
        aliases = sorted(
            {cls._normalize(alias) for alias in catalog.aliases()},
            key=lambda value: (-len(value.split()), -len(value), value),
        )
        return tuple(
            (
                alias,
                re.compile(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                    re.IGNORECASE,
                ),
            )
            for alias in aliases
            if alias
        )

    @classmethod
    def _is_strict_request(cls, text: str) -> bool:
        return any(pattern.search(text) for pattern in cls._STRICT_PATTERNS)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.lower().split())
