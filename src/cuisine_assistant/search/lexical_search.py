from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from cuisine_assistant.domain.models import SearchCandidate
from cuisine_assistant.domain.ports import EntitySearch, TextNormalizer


class CsvLexicalEntitySearch(EntitySearch):
    def __init__(self, aliases_csv: Path, normalizer: TextNormalizer) -> None:
        self._normalizer = normalizer
        self._index = self._load_index(aliases_csv)

    def _load_index(self, path: Path) -> dict[str, list[SearchCandidate]]:
        index: dict[str, list[tuple[bool, int, SearchCandidate]]] = {}
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                alias = self._normalizer.normalize(row["alias"])
                if not alias:
                    continue
                verified = str(row.get("verified", "")).lower() == "true"
                priority = int(row.get("match_priority") or 0)
                candidate = SearchCandidate(
                    entity_id=row["ingredient_id"],
                    matched_text=alias,
                    method="exact" if row.get("alias_type") == "canonical" else "alias",
                    score=1.0,
                )
                index.setdefault(alias, []).append((verified, priority, candidate))

        result: dict[str, list[SearchCandidate]] = {}
        for alias, ranked in index.items():
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            result[alias] = [item[2] for item in ranked]
        return result

    def find(self, normalized_text: str) -> list[SearchCandidate]:
        return list(self._index.get(normalized_text, []))

    def known_terms(self) -> Iterable[str]:
        return self._index.keys()
