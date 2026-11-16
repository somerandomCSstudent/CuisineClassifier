from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from cuisine_assistant.domain.models import Cuisine
from cuisine_assistant.domain.recommendation_models import (
    CuisinePreference,
    Dish,
    RecommendationRequest,
    RecommendationResult,
)


class DishCatalog(ABC):
    """Provides technology-independent access to canonical dishes."""

    @abstractmethod
    def get_by_id(self, dish_id: str) -> Dish | None:
        """Returns a dish by its stable application identifier."""
        raise NotImplementedError

    @abstractmethod
    def get_by_ontology_id(self, ontology_id: str) -> Dish | None:
        """Returns a dish by its local OWL entity identifier."""
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> tuple[Dish, ...]:
        """Returns all known dishes as an immutable collection."""
        raise NotImplementedError


class DishRecommender(ABC):
    """Produces ranked dish recommendations from a prepared request."""

    @abstractmethod
    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        """Returns ranked and explained dish recommendations."""
        raise NotImplementedError


class CuisinePreferenceExtractor(ABC):
    """Extracts an optional explicit cuisine preference from English text."""

    @abstractmethod
    def extract(self, text: str) -> CuisinePreference | None:
        """Returns the detected cuisine preference or None when absent."""
        raise NotImplementedError


class CuisinePreferenceCatalog(ABC):
    """Provides canonical cuisines and searchable aliases for preference extraction."""

    @abstractmethod
    def find_by_alias(self, normalized_alias: str) -> Cuisine | None:
        """Returns the cuisine mapped to an exact normalized alias."""
        raise NotImplementedError

    @abstractmethod
    def aliases(self) -> Iterable[str]:
        """Returns all normalized aliases available for phrase matching."""
        raise NotImplementedError
