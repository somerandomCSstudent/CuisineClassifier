from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .models import ExtractedIngredient, IngredientMention, SearchCandidate


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
