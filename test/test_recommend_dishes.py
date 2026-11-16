from __future__ import annotations

from cuisine_assistant.application import (
    ClassifyRecipeResult,
    LinkIngredientsResult,
    RecommendDishesUseCase,
)
from cuisine_assistant.domain.models import ClassificationResult
from cuisine_assistant.domain.recommendation_models import (
    CuisinePreference,
    RecommendationResult,
)
from cuisine_assistant.domain.recommendation_ports import (
    CuisinePreferenceExtractor,
    DishRecommender,
)


class StubClassifyRecipeUseCase:
    """Returns a deterministic classification result for recommendation tests."""

    def __init__(self, result: ClassifyRecipeResult) -> None:
        self._result = result
        self.received_text: str | None = None

    def execute(self, recipe_text: str) -> ClassifyRecipeResult:
        self.received_text = recipe_text
        return self._result


class StubCuisinePreferenceExtractor(CuisinePreferenceExtractor):
    """Returns an optional deterministic cuisine preference for tests."""

    def __init__(self, preference: CuisinePreference | None) -> None:
        self._preference = preference
        self.received_text: str | None = None

    def extract(self, text: str) -> CuisinePreference | None:
        self.received_text = text
        return self._preference


class CapturingDishRecommender(DishRecommender):
    """Captures the recommendation request and returns an empty result."""

    def __init__(self) -> None:
        self.received_request = None

    def recommend(self, request):
        self.received_request = request
        return RecommendationResult(request=request)


def create_classification_result() -> ClassifyRecipeResult:
    """Creates an empty but valid classification result for tests."""
    return ClassifyRecipeResult(
        linking=LinkIngredientsResult(linked_ingredients=()),
        classification=ClassificationResult(),
    )


def test_propagates_text_to_classification_and_preference_extraction() -> None:
    classification_use_case = StubClassifyRecipeUseCase(
        create_classification_result()
    )
    preference_extractor = StubCuisinePreferenceExtractor(None)
    recommender = CapturingDishRecommender()
    use_case = RecommendDishesUseCase(
        classification_use_case,
        preference_extractor,
        recommender,
    )

    result = use_case.execute(
        "I have chicken and paprika and I would like Hungarian food."
    )

    assert classification_use_case.received_text is not None
    assert preference_extractor.received_text == (
        "I have chicken and paprika and I would like Hungarian food."
    )
    assert recommender.received_request is result.recommendation.request
    assert result.has_recommendations is False


def test_adds_explicit_cuisine_preference_to_request() -> None:
    preference = CuisinePreference(
        cuisine_id="hungarian",
        ontology_id="HungarianCuisine",
        source_text="Hungarian food",
        match_score=1.0,
    )
    recommender = CapturingDishRecommender()
    use_case = RecommendDishesUseCase(
        StubClassifyRecipeUseCase(create_classification_result()),
        StubCuisinePreferenceExtractor(preference),
        recommender,
    )

    result = use_case.execute("I would like Hungarian food.")

    assert result.cuisine_preference is preference
    assert recommender.received_request.preferred_cuisine is preference


def test_supports_input_without_explicit_cuisine_preference() -> None:
    recommender = CapturingDishRecommender()
    use_case = RecommendDishesUseCase(
        StubClassifyRecipeUseCase(create_classification_result()),
        StubCuisinePreferenceExtractor(None),
        recommender,
    )

    result = use_case.execute("I have chicken and paprika.")

    assert result.cuisine_preference is None
    assert recommender.received_request.preferred_cuisine is None
