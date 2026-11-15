from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .models import (
    ExtractedIngredient,
    Ingredient,
    IngredientMention,
    OntologyEntityReference,
    SearchCandidate,
)


class TextNormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        raise NotImplementedError


class IngredientMentionParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> list[IngredientMention]:
        raise NotImplementedError


class EntitySearch(ABC):
    @abstractmethod
    def find(self, normalized_text: str) -> list[SearchCandidate]:
        raise NotImplementedError

    @abstractmethod
    def known_terms(self) -> Iterable[str]:
        raise NotImplementedError


class IngredientExtractor(ABC):
    @abstractmethod
    def extract(self, recipe_text: str) -> list[ExtractedIngredient]:
        raise NotImplementedError


class IngredientCatalog(ABC):
    """Provides technology-independent access to canonical ingredients."""

    @abstractmethod
    def get_by_id(self, ingredient_id: str) -> Ingredient | None:
        """Returns an ingredient by its stable application identifier."""
        raise NotImplementedError

    @abstractmethod
    def get_by_ontology_id(self, ontology_id: str) -> Ingredient | None:
        """Returns an ingredient by its local OWL entity identifier."""
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> tuple[Ingredient, ...]:
        """Returns all known ingredients as an immutable collection."""
        raise NotImplementedError


class KnowledgeGraphRepository(ABC):
    """Provides technology-independent read access to the cuisine knowledge graph."""

    @abstractmethod
    def get_entity_reference(
        self,
        ontology_id: str,
    ) -> OntologyEntityReference | None:
        """Returns an ontology reference when the entity exists in the graph."""
        raise NotImplementedError

    @abstractmethod
    def get_entity_types(self, ontology_id: str) -> tuple[str, ...]:
        """Returns the local RDF type identifiers assigned to an entity."""
        raise NotImplementedError

    @abstractmethod
    def find_dish_ids_by_ingredient_ids(
        self,
        ingredient_ontology_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Returns dishes connected to at least one supplied ingredient entity."""
        raise NotImplementedError

    @abstractmethod
    def get_dish_ingredient_ids(self, dish_ontology_id: str) -> tuple[str, ...]:
        """Returns ingredient ontology identifiers linked to a dish."""
        raise NotImplementedError

    @abstractmethod
    def get_dish_cuisine_id(self, dish_ontology_id: str) -> str | None:
        """Returns the cuisine ontology identifier linked to a dish."""
        raise NotImplementedError
