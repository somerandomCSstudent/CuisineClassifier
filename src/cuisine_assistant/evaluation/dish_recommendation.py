from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class DishRecommendationUseCase(Protocol):
    """Defines the application-level recommendation operation required by evaluation."""

    def execute(self, user_text: str):
        """Returns dish recommendations for one English input text."""
        ...


@dataclass(frozen=True)
class LabeledRecommendationExample:
    """Represents one manually labeled dish-recommendation evaluation example."""

    example_id: str
    text: str
    relevant_dish_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not self.relevant_dish_ids:
            raise ValueError("relevant_dish_ids must not be empty")


@dataclass(frozen=True)
class RecommendationEvaluation:
    """Stores aggregate retrieval metrics for ranked dish recommendations."""

    total_examples: int
    examples_with_recommendations: int
    top_1_hit_rate: float
    top_3_hit_rate: float
    mean_reciprocal_rank: float
    mean_top_ingredient_match: float


class DishRecommendationEvaluator:
    """Evaluates dish recommendations with top-k hit rate and reciprocal rank."""

    def __init__(
        self,
        recommender: DishRecommendationUseCase,
        *,
        top_k: int = 3,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._recommender = recommender
        self._top_k = top_k

    def evaluate(
        self,
        examples: tuple[LabeledRecommendationExample, ...],
    ) -> RecommendationEvaluation:
        if not examples:
            raise ValueError("at least one recommendation example is required")

        top_1_hits = 0
        top_k_hits = 0
        reciprocal_rank_sum = 0.0
        ingredient_match_sum = 0.0
        examples_with_recommendations = 0

        for example in examples:
            result = self._recommender.execute(example.text)
            recommendations = result.recommendation.recommendations
            relevant = set(example.relevant_dish_ids)
            if recommendations:
                examples_with_recommendations += 1
                ingredient_match_sum += (
                    recommendations[0].evidence.ingredient_match_score
                )

            ranked_ids = tuple(item.dish.dish_id for item in recommendations)
            if ranked_ids and ranked_ids[0] in relevant:
                top_1_hits += 1
            if relevant.intersection(ranked_ids[: self._top_k]):
                top_k_hits += 1

            first_relevant_rank = next(
                (
                    rank
                    for rank, dish_id in enumerate(ranked_ids, start=1)
                    if dish_id in relevant
                ),
                None,
            )
            if first_relevant_rank is not None:
                reciprocal_rank_sum += 1.0 / first_relevant_rank

        total = len(examples)
        mean_match = (
            ingredient_match_sum / examples_with_recommendations
            if examples_with_recommendations
            else 0.0
        )
        return RecommendationEvaluation(
            total_examples=total,
            examples_with_recommendations=examples_with_recommendations,
            top_1_hit_rate=top_1_hits / total,
            top_3_hit_rate=top_k_hits / total,
            mean_reciprocal_rank=reciprocal_rank_sum / total,
            mean_top_ingredient_match=mean_match,
        )
