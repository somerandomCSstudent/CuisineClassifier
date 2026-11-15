from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MatchMethod(StrEnum):
    EXACT = "exact"
    ALIAS = "alias"
    LEMMA = "lemma"
    EMBEDDING = "embedding"
    UNMATCHED = "unmatched"


class IngredientCategory(StrEnum):
    DAIRY_PRODUCT = "DairyProduct"
    DAIRY_ALTERNATIVE = "DairyAlternative"
    EGG_PRODUCT = "EggProduct"
    FRUIT = "Fruit"
    GRAIN = "Grain"
    HERB = "Herb"
    LEGUME = "Legume"
    MEAT = "Meat"
    OIL = "Oil"
    SEAFOOD = "Seafood"
    SPICE = "Spice"
    VEGETABLE = "Vegetable"
    OTHER = "OtherIngredient"


@dataclass(frozen=True)
class SearchCandidate:
    entity_id: str
    matched_text: str
    method: MatchMethod
    score: float

    def __post_init__(self) -> None:
        _validate_non_empty("entity_id", self.entity_id)
        _validate_non_empty("matched_text", self.matched_text)
        _validate_score(self.score)


@dataclass(frozen=True)
class IngredientMention:
    surface_form: str
    normalized_text: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        _validate_non_empty("surface_form", self.surface_form)
        _validate_non_empty("normalized_text", self.normalized_text)
        _validate_span(self.start_char, self.end_char)


@dataclass(frozen=True)
class ExtractedIngredient:
    surface_form: str
    normalized_text: str
    start_char: int
    end_char: int
    matched_ingredient_id: str | None
    match_method: MatchMethod
    match_score: float
    candidate_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_non_empty("surface_form", self.surface_form)
        _validate_non_empty("normalized_text", self.normalized_text)
        _validate_span(self.start_char, self.end_char)
        _validate_score(self.match_score)

        if self.match_method is MatchMethod.UNMATCHED:
            if self.matched_ingredient_id is not None:
                raise ValueError(
                    "matched_ingredient_id must be None for unmatched ingredients"
                )
            if self.match_score != 0.0:
                raise ValueError("match_score must be 0.0 for unmatched ingredients")
        elif self.matched_ingredient_id is None:
            raise ValueError(
                "matched_ingredient_id is required for a successful match"
            )


@dataclass(frozen=True)
class Ingredient:
    ingredient_id: str
    canonical_name: str
    category: IngredientCategory
    ontology_id: str
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty("ingredient_id", self.ingredient_id)
        _validate_non_empty("canonical_name", self.canonical_name)
        _validate_non_empty("ontology_id", self.ontology_id)


@dataclass(frozen=True)
class OntologyEntityReference:
    ontology_id: str
    iri: str
    rdf_types: tuple[str, ...] = field(default_factory=tuple)
    exists_in_knowledge_graph: bool = False

    def __post_init__(self) -> None:
        _validate_non_empty("ontology_id", self.ontology_id)
        _validate_non_empty("iri", self.iri)


@dataclass(frozen=True)
class LinkedIngredient:
    extracted: ExtractedIngredient
    ingredient: Ingredient | None
    ontology_entity: OntologyEntityReference | None

    def __post_init__(self) -> None:
        if self.extracted.match_method is MatchMethod.UNMATCHED:
            if self.ingredient is not None or self.ontology_entity is not None:
                raise ValueError(
                    "Unmatched extraction cannot have ingredient or ontology links"
                )
            return

        if self.ingredient is None:
            raise ValueError("Matched extraction requires an Ingredient")
        if self.extracted.matched_ingredient_id != self.ingredient.ingredient_id:
            raise ValueError(
                "Extracted ingredient ID must match the linked Ingredient ID"
            )
        if self.ontology_entity is not None:
            if self.ingredient.ontology_id != self.ontology_entity.ontology_id:
                raise ValueError(
                    "Ingredient ontology_id must match the ontology reference"
                )

    @property
    def is_linked_to_knowledge_graph(self) -> bool:
        return bool(
            self.ontology_entity
            and self.ontology_entity.exists_in_knowledge_graph
        )


def _validate_non_empty(field_name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _validate_score(score: float) -> None:
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0.0 and 1.0")


def _validate_span(start_char: int, end_char: int) -> None:
    if start_char < 0:
        raise ValueError("start_char must not be negative")
    if end_char <= start_char:
        raise ValueError("end_char must be greater than start_char")
