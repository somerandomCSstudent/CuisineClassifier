from __future__ import annotations

from abc import ABC, abstractmethod

from cuisine_assistant.retrieval.models import (
    RecipeDocument,
    RecipeSearchQuery,
    RecipeSearchResult,
)


class RecipeDocumentRepository(ABC):
    """Provides technology-independent access to searchable recipe documents."""

    @abstractmethod
    def get_by_id(self, recipe_id: str) -> RecipeDocument | None:
        """Returns one recipe document by its stable identifier."""
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> tuple[RecipeDocument, ...]:
        """Returns all recipe documents from the active knowledge base bundle."""
        raise NotImplementedError


class RecipeRetriever(ABC):
    """Returns ranked recipe documents for lexical or embedding-based search."""

    @abstractmethod
    def search(self, query: RecipeSearchQuery) -> RecipeSearchResult:
        """Returns ranked and explained documents for the search query."""
        raise NotImplementedError
