from __future__ import annotations

from dataclasses import dataclass, field

from cuisine_assistant.domain.models import LinkedIngredient, CuisinePrediction


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_score(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class CuisinePreference:
    """Represents an optional cuisine preference explicitly stated by the user."""

    cuisine_id: str
    ontology_id: str
    source_text: str
    match_score: float
    strict: bool = False

    def __post_init__(self) -> None:
        _require_text("cuisine_id", self.cuisine_id)
        _require_text("ontology_id", self.ontology_id)
        _require_text("source_text", self.source_text)
        _require_score(self.match_score)


@dataclass(frozen=True)
class Dish:
    """Represents a canonical dish available for knowledge-based recommendation."""

    dish_id: str
    name: str
    ontology_id: str
    dish_category: str
    cuisine_id: str
    description: str | None = None

    def __post_init__(self) -> None:
        _require_text("dish_id", self.dish_id)
        _require_text("name", self.name)
        _require_text("ontology_id", self.ontology_id)
        _require_text("dish_category", self.dish_category)
        _require_text("cuisine_id", self.cuisine_id)


@dataclass(frozen=True)
class RecommendationRequest:
    """Contains linked ingredients, inferred cuisines, and an optional user preference."""

    linked_ingredients: tuple[LinkedIngredient, ...]
    cuisine_predictions: tuple[CuisinePrediction, ...] = field(default_factory=tuple)
    preferred_cuisine: CuisinePreference | None = None

    @property
    def linked_ingredient_ids(self) -> tuple[str, ...]:
        """Returns unique canonical ingredient identifiers in input order."""
        seen: set[str] = set()
        identifiers: list[str] = []
        for item in self.linked_ingredients:
            if item.ingredient is None:
                continue
            identifier = item.ingredient.ingredient_id
            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)
        return tuple(identifiers)

    @property
    def linked_ontology_ids(self) -> tuple[str, ...]:
        """Returns unique ingredient ontology identifiers in input order."""
        seen: set[str] = set()
        identifiers: list[str] = []
        for item in self.linked_ingredients:
            if item.ingredient is None:
                continue
            identifier = item.ingredient.ontology_id
            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)
        return tuple(identifiers)


@dataclass(frozen=True)
class RecommendationEvidence:
    """Explains ingredient and cuisine contributions to one dish recommendation."""

    matched_ingredient_ids: tuple[str, ...]
    missing_ingredient_ids: tuple[str, ...]
    ingredient_match_score: float
    cuisine_match_score: float
    used_explicit_cuisine_preference: bool = False

    def __post_init__(self) -> None:
        _require_score(self.ingredient_match_score)
        _require_score(self.cuisine_match_score)


@dataclass(frozen=True)
class DishRecommendation:
    """Represents one ranked dish recommendation with its scoring evidence."""

    dish: Dish
    score: float
    rank: int
    evidence: RecommendationEvidence

    def __post_init__(self) -> None:
        _require_score(self.score)
        if self.rank < 1:
            raise ValueError("rank must be at least 1")


@dataclass(frozen=True)
class RecommendationResult:
    """Contains ranked dish recommendations and the request used to produce them."""

    request: RecommendationRequest
    recommendations: tuple[DishRecommendation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ranks = tuple(item.rank for item in self.recommendations)
        if ranks != tuple(range(1, len(self.recommendations) + 1)):
            raise ValueError("recommendation ranks must be consecutive and start at 1")
        scores = tuple(item.score for item in self.recommendations)
        if scores != tuple(sorted(scores, reverse=True)):
            raise ValueError("recommendations must be ordered by descending score")

    @property
    def best_recommendation(self) -> DishRecommendation | None:
        """Returns the highest-ranked dish recommendation when available."""
        return self.recommendations[0] if self.recommendations else None
