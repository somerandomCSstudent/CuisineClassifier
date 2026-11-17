from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class RecipeSearchOperation(Protocol):
    """Defines the application recipe-search operation required by evaluation."""

    def execute(self, query_text: str):
        """Returns ranked recipe documents for one English query."""
        ...


@dataclass(frozen=True)
class LabeledRecipeSearchExample:
    """Represents one manually labeled recipe-retrieval query."""

    example_id: str
    query_text: str
    relevant_recipe_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must not be empty")
        if not self.query_text.strip():
            raise ValueError("query_text must not be empty")
        if not self.relevant_recipe_ids:
            raise ValueError("relevant_recipe_ids must not be empty")


@dataclass(frozen=True)
class RecipeRetrievalEvaluation:
    """Stores aggregate metrics for ranked recipe-document retrieval."""

    total_examples: int
    examples_with_results: int
    precision_at_1: float
    precision_at_3: float
    recall_at_3: float
    mean_reciprocal_rank: float
    ndcg_at_5: float
    coverage: float


class RecipeRetrievalEvaluator:
    """Evaluates recipe retrieval with precision, recall, MRR, nDCG, and coverage."""

    def __init__(self, search_operation: RecipeSearchOperation) -> None:
        self._search_operation = search_operation

    def evaluate(
        self,
        examples: tuple[LabeledRecipeSearchExample, ...],
    ) -> RecipeRetrievalEvaluation:
        """Calculates standard ranked-retrieval metrics over labeled queries."""
        if not examples:
            raise ValueError("at least one recipe search example is required")

        precision_1_sum = 0.0
        precision_3_sum = 0.0
        recall_3_sum = 0.0
        reciprocal_rank_sum = 0.0
        ndcg_5_sum = 0.0
        examples_with_results = 0

        for example in examples:
            result = self._search_operation.execute(example.query_text)
            ranked_ids = tuple(
                hit.document.recipe_id for hit in result.search.hits
            )
            relevant = set(example.relevant_recipe_ids)
            if ranked_ids:
                examples_with_results += 1

            precision_1_sum += self._precision_at_k(ranked_ids, relevant, 1)
            precision_3_sum += self._precision_at_k(ranked_ids, relevant, 3)
            recall_3_sum += self._recall_at_k(ranked_ids, relevant, 3)
            reciprocal_rank_sum += self._reciprocal_rank(ranked_ids, relevant)
            ndcg_5_sum += self._ndcg_at_k(ranked_ids, relevant, 5)

        total = len(examples)
        return RecipeRetrievalEvaluation(
            total_examples=total,
            examples_with_results=examples_with_results,
            precision_at_1=precision_1_sum / total,
            precision_at_3=precision_3_sum / total,
            recall_at_3=recall_3_sum / total,
            mean_reciprocal_rank=reciprocal_rank_sum / total,
            ndcg_at_5=ndcg_5_sum / total,
            coverage=examples_with_results / total,
        )

    @staticmethod
    def _precision_at_k(
        ranked_ids: tuple[str, ...],
        relevant: set[str],
        k: int,
    ) -> float:
        retrieved = ranked_ids[:k]
        if not retrieved:
            return 0.0
        hits = sum(recipe_id in relevant for recipe_id in retrieved)
        return hits / k

    @staticmethod
    def _recall_at_k(
        ranked_ids: tuple[str, ...],
        relevant: set[str],
        k: int,
    ) -> float:
        hits = sum(recipe_id in relevant for recipe_id in ranked_ids[:k])
        return hits / len(relevant)

    @staticmethod
    def _reciprocal_rank(
        ranked_ids: tuple[str, ...],
        relevant: set[str],
    ) -> float:
        rank = next(
            (
                index
                for index, recipe_id in enumerate(ranked_ids, start=1)
                if recipe_id in relevant
            ),
            None,
        )
        return 1.0 / rank if rank is not None else 0.0

    @staticmethod
    def _ndcg_at_k(
        ranked_ids: tuple[str, ...],
        relevant: set[str],
        k: int,
    ) -> float:
        import math

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, recipe_id in enumerate(ranked_ids[:k], start=1)
            if recipe_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )
        return dcg / idcg if idcg > 0 else 0.0
