from __future__ import annotations

from dataclasses import dataclass, field


def _require_text(field_name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_score(field_name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


@dataclass(frozen=True)
class RecipeDocument:
    """Represents one searchable recipe document from the active bundle."""

    recipe_id: str
    dish_id: str
    title: str
    raw_text: str
    ingredient_text: str
    instruction_text: str
    cuisine_id: str | None
    source: str
    source_url: str | None = None
    license_name: str | None = None

    def __post_init__(self) -> None:
        _require_text("recipe_id", self.recipe_id)
        _require_text("dish_id", self.dish_id)
        _require_text("title", self.title)
        _require_text("raw_text", self.raw_text)
        _require_text("source", self.source)

    @property
    def searchable_text(self) -> str:
        """Returns normalized document fields used by lexical retrieval."""
        return " ".join(
            part.strip()
            for part in (
                self.title,
                self.raw_text,
                self.ingredient_text,
                self.instruction_text,
                self.cuisine_id or "",
            )
            if part and part.strip()
        )


@dataclass(frozen=True)
class RecipeSearchQuery:
    """Contains one textual retrieval request and optional semantic filters."""

    text: str
    maximum_results: int = 5
    cuisine_id: str | None = None
    required_ingredient_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_text("text", self.text)
        if self.maximum_results < 1:
            raise ValueError("maximum_results must be at least 1")


@dataclass(frozen=True)
class RecipeSearchEvidence:
    """Explains why one recipe document matched a retrieval query."""

    matched_terms: tuple[str, ...]
    matched_ingredient_ids: tuple[str, ...]
    title_match_score: float
    content_match_score: float
    cuisine_match_score: float

    def __post_init__(self) -> None:
        _require_score("title_match_score", self.title_match_score)
        _require_score("content_match_score", self.content_match_score)
        _require_score("cuisine_match_score", self.cuisine_match_score)


@dataclass(frozen=True)
class RecipeSearchHit:
    """Represents one ranked and explainable recipe retrieval result."""

    document: RecipeDocument
    score: float
    rank: int
    evidence: RecipeSearchEvidence

    def __post_init__(self) -> None:
        _require_score("score", self.score)
        if self.rank < 1:
            raise ValueError("rank must be at least 1")


@dataclass(frozen=True)
class RecipeSearchResult:
    """Contains ranked recipe documents returned for one search query."""

    query: RecipeSearchQuery
    hits: tuple[RecipeSearchHit, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ranks = tuple(hit.rank for hit in self.hits)
        if ranks != tuple(range(1, len(self.hits) + 1)):
            raise ValueError("hit ranks must be consecutive and start at 1")
        scores = tuple(hit.score for hit in self.hits)
        if scores != tuple(sorted(scores, reverse=True)):
            raise ValueError("hits must be ordered by descending score")
        if len(self.hits) > self.query.maximum_results:
            raise ValueError("hits exceed query maximum_results")

    @property
    def best_hit(self) -> RecipeSearchHit | None:
        """Returns the highest-ranked recipe document when available."""
        return self.hits[0] if self.hits else None
