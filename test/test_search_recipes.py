from __future__ import annotations

import pytest

from cuisine_assistant.application import LinkIngredientsResult
from cuisine_assistant.application.search_recipes import SearchRecipesUseCase
from cuisine_assistant.domain.models import (
    ExtractedIngredient,
    Ingredient,
    IngredientCategory,
    LinkedIngredient,
    MatchMethod,
)
from cuisine_assistant.domain.recommendation_models import CuisinePreference
from cuisine_assistant.domain.recommendation_ports import CuisinePreferenceExtractor
from cuisine_assistant.retrieval import (
    RecipeRetriever,
    RecipeSearchResult,
)


class StubLinkIngredientsUseCase:
    """Returns deterministic linked ingredients for recipe search tests."""

    def __init__(self, result: LinkIngredientsResult) -> None:
        self._result = result
        self.received_text: str | None = None

    def execute(self, text: str) -> LinkIngredientsResult:
        self.received_text = text
        return self._result


class StubCuisinePreferenceExtractor(CuisinePreferenceExtractor):
    """Returns a configured cuisine preference for recipe search tests."""

    def __init__(self, preference: CuisinePreference | None) -> None:
        self._preference = preference
        self.received_text: str | None = None

    def extract(self, text: str) -> CuisinePreference | None:
        self.received_text = text
        return self._preference


class CapturingRecipeRetriever(RecipeRetriever):
    """Captures the generated search query and returns an empty result."""

    def __init__(self) -> None:
        self.received_query = None

    def search(self, query):
        self.received_query = query
        return RecipeSearchResult(query=query)


def linked_ingredient(
    ingredient_id: str,
    ontology_id: str,
    category: IngredientCategory,
) -> LinkedIngredient:
    """Creates one linked ingredient fixture."""
    surface = ingredient_id.replace("_", " ")
    extracted = ExtractedIngredient(
        surface_form=surface,
        normalized_text=surface,
        start_char=0,
        end_char=len(surface),
        matched_ingredient_id=ingredient_id,
        match_method=MatchMethod.EXACT,
        match_score=1.0,
        candidate_ids=(ingredient_id,),
    )
    ingredient = Ingredient(
        ingredient_id=ingredient_id,
        canonical_name=surface,
        category=category,
        ontology_id=ontology_id,
    )
    return LinkedIngredient(
        extraction=extracted,
        ingredient=ingredient,
        ontology_reference=None,
    )


def linking_result() -> LinkIngredientsResult:
    """Creates linked chicken and paprika query ingredients."""
    return LinkIngredientsResult(
        linked_ingredients=(
            linked_ingredient(
                "chicken",
                "Chicken",
                IngredientCategory.MEAT,
            ),
            linked_ingredient(
                "paprika",
                "Paprika",
                IngredientCategory.SPICE,
            ),
        )
    )


def test_builds_search_query_from_linked_ingredients_and_cuisine() -> None:
    """Verifies semantic query interpretation reaches the recipe retriever."""
    preference = CuisinePreference(
        cuisine_id="hungarian",
        ontology_id="HungarianCuisine",
        source_text="Hungarian food",
        match_score=1.0,
    )
    linker = StubLinkIngredientsUseCase(linking_result())
    extractor = StubCuisinePreferenceExtractor(preference)
    retriever = CapturingRecipeRetriever()
    use_case = SearchRecipesUseCase(
        linker,
        extractor,
        retriever,
        maximum_results=3,
    )

    result = use_case.execute(
        "  Find Hungarian food with chicken and paprika.  "
    )

    assert result.query_text == (
        "Find Hungarian food with chicken and paprika."
    )
    assert linker.received_text == result.query_text
    assert extractor.received_text == result.query_text
    assert retriever.received_query.text == result.query_text
    assert retriever.received_query.maximum_results == 3
    assert retriever.received_query.cuisine_id == "hungarian"
    assert retriever.received_query.required_ingredient_ids == (
        "chicken",
        "paprika",
    )
    assert result.cuisine_preference is preference
    assert result.has_results is False


def test_supports_query_without_explicit_cuisine_preference() -> None:
    """Verifies lexical retrieval still works when no cuisine is mentioned."""
    retriever = CapturingRecipeRetriever()
    use_case = SearchRecipesUseCase(
        StubLinkIngredientsUseCase(linking_result()),
        StubCuisinePreferenceExtractor(None),
        retriever,
    )

    result = use_case.execute("chicken and paprika recipe")

    assert result.cuisine_preference is None
    assert retriever.received_query.cuisine_id is None
    assert retriever.received_query.required_ingredient_ids == (
        "chicken",
        "paprika",
    )


def test_deduplicates_linked_ingredient_identifiers() -> None:
    """Verifies repeated mentions become one semantic ingredient constraint."""
    chicken = linked_ingredient(
        "chicken",
        "Chicken",
        IngredientCategory.MEAT,
    )
    retriever = CapturingRecipeRetriever()
    use_case = SearchRecipesUseCase(
        StubLinkIngredientsUseCase(
            LinkIngredientsResult((chicken, chicken))
        ),
        StubCuisinePreferenceExtractor(None),
        retriever,
    )

    use_case.execute("chicken chicken recipe")

    assert retriever.received_query.required_ingredient_ids == ("chicken",)


def test_rejects_blank_query_before_running_dependencies() -> None:
    """Verifies empty recipe search requests fail at the application boundary."""
    linker = StubLinkIngredientsUseCase(linking_result())
    extractor = StubCuisinePreferenceExtractor(None)
    retriever = CapturingRecipeRetriever()
    use_case = SearchRecipesUseCase(linker, extractor, retriever)

    with pytest.raises(ValueError, match="query_text"):
        use_case.execute("   ")

    assert linker.received_text is None
    assert extractor.received_text is None
    assert retriever.received_query is None


def test_rejects_invalid_result_limit() -> None:
    """Verifies the configured top-k retrieval limit is positive."""
    with pytest.raises(ValueError, match="maximum_results"):
        SearchRecipesUseCase(
            StubLinkIngredientsUseCase(linking_result()),
            StubCuisinePreferenceExtractor(None),
            CapturingRecipeRetriever(),
            maximum_results=0,
        )
