from __future__ import annotations

from collections.abc import Iterable

from cuisine_assistant.domain.models import Cuisine
from cuisine_assistant.domain.recommendation_models import (
    CuisinePreference,
    Dish,
    RecommendationRequest,
    RecommendationResult,
)
from cuisine_assistant.domain.recommendation_ports import (
    CuisinePreferenceCatalog,
    CuisinePreferenceExtractor,
    DishCatalog,
    DishRecommender,
)


class InMemoryDishCatalog(DishCatalog):
    """Provides deterministic dish records for recommendation-port tests."""

    def __init__(self) -> None:
        self._dishes = (
            Dish(
                dish_id="chicken_paprikash",
                name="Chicken Paprikash",
                ontology_id="ChickenPaprikash",
                dish_category="MainCourse",
                cuisine_id="hungarian",
            ),
        )

    def get_by_id(self, dish_id: str) -> Dish | None:
        return next((dish for dish in self._dishes if dish.dish_id == dish_id), None)

    def get_by_ontology_id(self, ontology_id: str) -> Dish | None:
        return next((dish for dish in self._dishes if dish.ontology_id == ontology_id), None)

    def list_all(self) -> tuple[Dish, ...]:
        return self._dishes


class EmptyDishRecommender(DishRecommender):
    """Returns an empty recommendation result for port-contract tests."""

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        return RecommendationResult(request=request)


class InMemoryCuisinePreferenceCatalog(CuisinePreferenceCatalog):
    """Maps normalized cuisine phrases to canonical cuisine records."""

    def __init__(self) -> None:
        self._cuisine = Cuisine(
            cuisine_id="hungarian",
            name="Hungarian cuisine",
            ontology_id="HungarianCuisine",
            cuisine_group="EuropeanCuisine",
        )
        self._aliases = {
            "hungarian": self._cuisine,
            "hungarian food": self._cuisine,
            "hungarian cuisine": self._cuisine,
        }

    def find_by_alias(self, normalized_alias: str) -> Cuisine | None:
        return self._aliases.get(normalized_alias)

    def aliases(self) -> Iterable[str]:
        return self._aliases.keys()


class StubCuisinePreferenceExtractor(CuisinePreferenceExtractor):
    """Returns a fixed preference for extractor-port tests."""

    def extract(self, text: str) -> CuisinePreference | None:
        if "hungarian" not in text.lower():
            return None
        return CuisinePreference(
            cuisine_id="hungarian",
            ontology_id="HungarianCuisine",
            source_text="Hungarian food",
            match_score=1.0,
        )


def test_dish_catalog_supports_both_identifiers() -> None:
    catalog: DishCatalog = InMemoryDishCatalog()

    assert catalog.get_by_id("chicken_paprikash") is not None
    assert catalog.get_by_ontology_id("ChickenPaprikash") is not None
    assert len(catalog.list_all()) == 1


def test_recommender_returns_result_for_request() -> None:
    request = RecommendationRequest(linked_ingredients=())
    recommender: DishRecommender = EmptyDishRecommender()

    result = recommender.recommend(request)

    assert result.request is request
    assert result.recommendations == ()


def test_preference_catalog_exposes_aliases() -> None:
    catalog: CuisinePreferenceCatalog = InMemoryCuisinePreferenceCatalog()

    assert catalog.find_by_alias("hungarian food") is not None
    assert "hungarian cuisine" in tuple(catalog.aliases())


def test_preference_extractor_returns_optional_preference() -> None:
    extractor: CuisinePreferenceExtractor = StubCuisinePreferenceExtractor()

    preference = extractor.extract("I would like Hungarian food")

    assert preference is not None
    assert preference.cuisine_id == "hungarian"
    assert extractor.extract("I have chicken and paprika") is None
