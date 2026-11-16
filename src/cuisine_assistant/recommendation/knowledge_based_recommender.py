from __future__ import annotations

from cuisine_assistant.domain.ports import (
    IngredientCatalog,
    KnowledgeGraphRepository,
)
from cuisine_assistant.domain.recommendation_models import (
    DishRecommendation,
    RecommendationEvidence,
    RecommendationRequest,
    RecommendationResult,
)
from cuisine_assistant.domain.recommendation_ports import (
    DishCatalog,
    DishRecommender,
)


class KnowledgeBasedDishRecommender(DishRecommender):
    """Ranks knowledge-graph dishes by ingredient and cuisine compatibility."""

    def __init__(
        self,
        dish_catalog: DishCatalog,
        ingredient_catalog: IngredientCatalog,
        knowledge_graph: KnowledgeGraphRepository,
        *,
        maximum_recommendations: int = 3,
        ingredient_weight: float = 0.70,
        inferred_cuisine_weight: float = 0.30,
        explicit_ingredient_weight: float = 0.65,
        explicit_cuisine_weight: float = 0.35,
        minimum_score: float = 0.0,
    ) -> None:
        if maximum_recommendations < 1:
            raise ValueError("maximum_recommendations must be at least 1")
        self._validate_weight_pair(
            ingredient_weight,
            inferred_cuisine_weight,
            "inferred cuisine",
        )
        self._validate_weight_pair(
            explicit_ingredient_weight,
            explicit_cuisine_weight,
            "explicit cuisine",
        )
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0.0 and 1.0")

        self._dish_catalog = dish_catalog
        self._ingredient_catalog = ingredient_catalog
        self._knowledge_graph = knowledge_graph
        self._maximum_recommendations = maximum_recommendations
        self._ingredient_weight = ingredient_weight
        self._inferred_cuisine_weight = inferred_cuisine_weight
        self._explicit_ingredient_weight = explicit_ingredient_weight
        self._explicit_cuisine_weight = explicit_cuisine_weight
        self._minimum_score = minimum_score

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        if not request.linked_ontology_ids:
            return RecommendationResult(request=request)

        candidate_ontology_ids = self._knowledge_graph.find_dish_ids_by_ingredient_ids(
            request.linked_ontology_ids
        )
        scored_candidates = []
        for dish_ontology_id in candidate_ontology_ids:
            dish = self._dish_catalog.get_by_ontology_id(dish_ontology_id)
            if dish is None:
                continue
            if self._is_excluded_by_strict_preference(request, dish.cuisine_id):
                continue

            score, evidence = self._score_dish(request, dish)
            if score > self._minimum_score:
                scored_candidates.append((dish, score, evidence))

        scored_candidates.sort(
            key=lambda item: (-item[1], item[0].dish_id)
        )
        selected = scored_candidates[: self._maximum_recommendations]
        recommendations = tuple(
            DishRecommendation(
                dish=dish,
                score=score,
                rank=rank,
                evidence=evidence,
            )
            for rank, (dish, score, evidence) in enumerate(selected, start=1)
        )
        return RecommendationResult(
            request=request,
            recommendations=recommendations,
        )

    def _score_dish(self, request, dish):
        input_ids = set(request.linked_ingredient_ids)
        dish_ontology_ids = self._knowledge_graph.get_dish_ingredient_ids(
            dish.ontology_id
        )
        dish_ids = {
            ingredient.ingredient_id
            for ontology_id in dish_ontology_ids
            if (
                ingredient := self._ingredient_catalog.get_by_ontology_id(
                    ontology_id
                )
            ) is not None
        }
        matched_ids = tuple(sorted(input_ids & dish_ids))
        missing_ids = tuple(sorted(dish_ids - input_ids))
        ingredient_score = len(matched_ids) / len(input_ids)

        cuisine_score, used_explicit = self._cuisine_score(
            request,
            dish.cuisine_id,
        )
        if used_explicit:
            score = (
                self._explicit_ingredient_weight * ingredient_score
                + self._explicit_cuisine_weight * cuisine_score
            )
        else:
            score = (
                self._ingredient_weight * ingredient_score
                + self._inferred_cuisine_weight * cuisine_score
            )

        evidence = RecommendationEvidence(
            matched_ingredient_ids=matched_ids,
            missing_ingredient_ids=missing_ids,
            ingredient_match_score=ingredient_score,
            cuisine_match_score=cuisine_score,
            used_explicit_cuisine_preference=used_explicit,
        )
        return score, evidence

    @staticmethod
    def _cuisine_score(
        request: RecommendationRequest,
        dish_cuisine_id: str,
    ) -> tuple[float, bool]:
        preference = request.preferred_cuisine
        if preference is not None:
            score = (
                preference.match_score
                if preference.cuisine_id == dish_cuisine_id
                else 0.0
            )
            return score, True

        for prediction in request.cuisine_predictions:
            if prediction.cuisine.cuisine_id == dish_cuisine_id:
                return prediction.score, False
        return 0.0, False

    @staticmethod
    def _is_excluded_by_strict_preference(
        request: RecommendationRequest,
        dish_cuisine_id: str,
    ) -> bool:
        preference = request.preferred_cuisine
        return bool(
            preference
            and preference.strict
            and preference.cuisine_id != dish_cuisine_id
        )

    @staticmethod
    def _validate_weight_pair(
        first: float,
        second: float,
        label: str,
    ) -> None:
        if not 0.0 <= first <= 1.0 or not 0.0 <= second <= 1.0:
            raise ValueError(f"{label} weights must be between 0.0 and 1.0")
        if abs((first + second) - 1.0) > 1e-9:
            raise ValueError(f"{label} weights must add up to 1.0")
