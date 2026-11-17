from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.application.link_ingredients import (
    LinkIngredientsResult,
    LinkIngredientsUseCase,
)
from cuisine_assistant.domain.recommendation_models import CuisinePreference
from cuisine_assistant.domain.recommendation_ports import CuisinePreferenceExtractor
from cuisine_assistant.retrieval import (
    RecipeRetriever,
    RecipeSearchQuery,
    RecipeSearchResult,
)


@dataclass(frozen=True)
class SearchRecipesResult:
    """Combines query interpretation and ranked recipe document retrieval."""

    query_text: str
    linking: LinkIngredientsResult
    cuisine_preference: CuisinePreference | None
    search: RecipeSearchResult

    @property
    def has_results(self) -> bool:
        """Returns whether at least one recipe document was retrieved."""
        return bool(self.search.hits)


class SearchRecipesUseCase:
    """Links query ingredients, detects cuisine preference, and retrieves recipes."""

    def __init__(
        self,
        link_ingredients: LinkIngredientsUseCase,
        cuisine_preference_extractor: CuisinePreferenceExtractor,
        recipe_retriever: RecipeRetriever,
        *,
        maximum_results: int = 5,
    ) -> None:
        if maximum_results < 1:
            raise ValueError("maximum_results must be at least 1")
        self._link_ingredients = link_ingredients
        self._cuisine_preference_extractor = cuisine_preference_extractor
        self._recipe_retriever = recipe_retriever
        self._maximum_results = maximum_results

    def execute(self, query_text: str) -> SearchRecipesResult:
        """Returns explainable top-k recipes for one English search query."""
        normalized_query = query_text.strip()
        if not normalized_query:
            raise ValueError("query_text must not be empty")

        linking = self._link_ingredients.execute(normalized_query)
        cuisine_preference = self._cuisine_preference_extractor.extract(
            normalized_query
        )
        search_query = RecipeSearchQuery(
            text=normalized_query,
            maximum_results=self._maximum_results,
            cuisine_id=(
                cuisine_preference.cuisine_id
                if cuisine_preference is not None
                else None
            ),
            required_ingredient_ids=self._linked_ingredient_ids(linking),
        )
        search = self._recipe_retriever.search(search_query)
        return SearchRecipesResult(
            query_text=normalized_query,
            linking=linking,
            cuisine_preference=cuisine_preference,
            search=search,
        )

    @staticmethod
    def _linked_ingredient_ids(
        linking: LinkIngredientsResult,
    ) -> tuple[str, ...]:
        identifiers: list[str] = []
        seen: set[str] = set()
        for linked in linking.linked_ingredients:
            ingredient = linked.ingredient
            if ingredient is None:
                continue
            ingredient_id = ingredient.ingredient_id
            if ingredient_id not in seen:
                seen.add(ingredient_id)
                identifiers.append(ingredient_id)
        return tuple(identifiers)
