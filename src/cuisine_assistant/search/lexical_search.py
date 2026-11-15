from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from cuisine_assistant.domain.models import MatchMethod, SearchCandidate
from cuisine_assistant.domain.ports import EntitySearch, TextNormalizer


class CsvLexicalEntitySearch(EntitySearch):
    """Provides exact and alias-based ingredient lookup from a CSV file."""

    def __init__(self, aliases_csv: str | Path, normalizer: TextNormalizer) -> None:
        self._normalizer = normalizer
        self._index = self._load_index(Path(aliases_csv))

    def find(self, normalized_text: str) -> list[SearchCandidate]:
        return list(self._index.get(normalized_text, ()))

    def known_terms(self) -> Iterable[str]:
        return self._index.keys()

    def _load_index(
        self,
        path: Path,
    ) -> dict[str, tuple[SearchCandidate, ...]]:
        if not path.is_file():
            raise FileNotFoundError(f"Ingredient alias CSV does not exist: {path}")

        ranked_index: dict[
            str,
            list[tuple[bool, int, SearchCandidate]],
        ] = {}
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            self._validate_header(reader.fieldnames)
            for line_number, row in enumerate(reader, start=2):
                alias = self._normalizer.normalize(row.get("alias") or "")
                ingredient_id = (row.get("ingredient_id") or "").strip()
                if not alias or not ingredient_id:
                    raise ValueError(
                        f"Invalid ingredient alias data at CSV line {line_number}"
                    )
                verified = (row.get("verified") or "").strip().lower() == "true"
                priority = int((row.get("match_priority") or "0").strip())
                alias_type = (row.get("alias_type") or "").strip().lower()
                method = (
                    MatchMethod.EXACT
                    if alias_type == "canonical"
                    else MatchMethod.ALIAS
                )
                candidate = SearchCandidate(
                    entity_id=ingredient_id,
                    matched_text=alias,
                    method=method,
                    score=1.0,
                )
                ranked_index.setdefault(alias, []).append(
                    (verified, priority, candidate)
                )

        index: dict[str, tuple[SearchCandidate, ...]] = {}
        for alias, ranked in ranked_index.items():
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            index[alias] = tuple(item[2] for item in ranked)
        return index

    @staticmethod
    def _validate_header(fieldnames: list[str] | None) -> None:
        required = {"ingredient_id", "alias", "alias_type"}
        missing = required - set(fieldnames or ())
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(
                f"Ingredient alias CSV is missing required columns: {columns}"
            )
