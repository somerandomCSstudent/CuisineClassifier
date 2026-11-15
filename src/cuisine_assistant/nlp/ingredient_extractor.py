from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExtractedIngredient:
    surface_form: str
    normalized_text: str
    start_char: int
    end_char: int
    matched_ingredient_id: str | None
    match_method: str
    match_score: float
    candidate_ids: list[str]


class IngredientExtractor:
    """Dictionary-based English ingredient extractor.

    This is the baseline for part 3. It links recipe spans to the canonical
    ingredient IDs used by the CSV data model and the OWL ontology.
    """

    def __init__(self, aliases_csv: str | Path) -> None:
        self.alias_index = self._load_aliases(Path(aliases_csv))
        self.pattern = self._build_pattern(self.alias_index.keys())

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower().replace("’", "'")
        text = re.sub(r"[^a-z0-9\s'-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _load_aliases(cls, path: Path) -> dict[str, list[dict]]:
        index: dict[str, list[dict]] = {}
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                alias = cls.normalize(row["alias"])
                if not alias:
                    continue
                entry = {
                    "ingredient_id": row["ingredient_id"],
                    "priority": int(row.get("match_priority") or 0),
                    "verified": str(row.get("verified", "")).lower() == "true",
                }
                index.setdefault(alias, []).append(entry)

        for entries in index.values():
            entries.sort(
                key=lambda item: (item["verified"], item["priority"]),
                reverse=True,
            )
        return index

    @staticmethod
    def _build_pattern(aliases: Iterable[str]) -> re.Pattern[str]:
        # Longest aliases first, so "tomato paste" wins over "tomato".
        ordered = sorted(set(aliases), key=lambda value: (-len(value.split()), -len(value)))
        alternatives = "|".join(re.escape(alias) for alias in ordered)
        if not alternatives:
            return re.compile(r"(?!x)x")
        return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])", re.IGNORECASE)

    def extract(self, recipe_text: str) -> list[ExtractedIngredient]:
        normalized_recipe = self.normalize(recipe_text)
        results: list[ExtractedIngredient] = []
        occupied: list[tuple[int, int]] = []

        for match in self.pattern.finditer(normalized_recipe):
            start, end = match.span()
            if any(start < old_end and end > old_start for old_start, old_end in occupied):
                continue

            surface = match.group(0)
            normalized = self.normalize(surface)
            candidates = self.alias_index[normalized]
            best = candidates[0]
            method = "exact" if normalized == self.normalize(best["ingredient_id"].replace("_", " ")) else "alias"

            results.append(
                ExtractedIngredient(
                    surface_form=surface,
                    normalized_text=normalized,
                    start_char=start,
                    end_char=end,
                    matched_ingredient_id=best["ingredient_id"],
                    match_method=method,
                    match_score=1.0,
                    candidate_ids=[item["ingredient_id"] for item in candidates],
                )
            )
            occupied.append((start, end))

        return sorted(results, key=lambda item: item.start_char)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    aliases_file = project_dir / "ingredient_aliases.csv"
    extractor = IngredientExtractor(aliases_file)

    print("Cuisine Assistant - ingredient extraction baseline")
    print("Enter an English recipe or ingredient list. Type 'exit' to quit.\n")

    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            print("Please enter some text.\n")
            continue

        result = extractor.extract(text)
        print(json.dumps([asdict(item) for item in result], indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
