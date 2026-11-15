from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchCandidate:
    entity_id: str
    matched_text: str
    method: str
    score: float


@dataclass(frozen=True)
class IngredientMention:
    surface_form: str
    normalized_text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ExtractedIngredient:
    surface_form: str
    normalized_text: str
    start_char: int
    end_char: int
    matched_ingredient_id: str | None
    match_method: str
    match_score: float
    candidate_ids: list[str] = field(default_factory=list)
