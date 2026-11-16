from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class RecipeClassifier(Protocol):
    """Defines the application-level operation required by the evaluator."""

    def execute(self, recipe_text: str):
        """Classifies one English recipe text."""
        ...


@dataclass(frozen=True)
class LabeledRecipe:
    """Represents one manually labeled recipe used for evaluation."""

    recipe_id: str
    text: str
    expected_cuisine_id: str

    def __post_init__(self) -> None:
        if not self.recipe_id.strip():
            raise ValueError("recipe_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not self.expected_cuisine_id.strip():
            raise ValueError("expected_cuisine_id must not be empty")


@dataclass(frozen=True)
class CuisineClassMetrics:
    """Stores precision, recall, and F1 metrics for one cuisine class."""

    cuisine_id: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Stores aggregate and per-class cuisine-classification measurements."""

    total_examples: int
    evaluated_examples: int
    no_prediction_count: int
    accuracy: float
    top_k_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: tuple[CuisineClassMetrics, ...]
    confusion_matrix: dict[str, dict[str, int]]


class CuisineClassificationEvaluator:
    """Evaluates a recipe classifier with accuracy, macro F1, and top-k accuracy."""

    NO_PREDICTION = "__no_prediction__"

    def __init__(self, classifier: RecipeClassifier, *, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._classifier = classifier
        self._top_k = top_k

    def evaluate(
        self,
        examples: tuple[LabeledRecipe, ...],
    ) -> ClassificationEvaluation:
        if not examples:
            raise ValueError("at least one labeled recipe is required")

        expected_labels = sorted({item.expected_cuisine_id for item in examples})
        matrix = self._empty_matrix(expected_labels)
        correct = 0
        top_k_correct = 0
        no_prediction_count = 0

        for example in examples:
            result = self._classifier.execute(example.text)
            predictions = result.classification.predictions
            predicted_id = (
                predictions[0].cuisine.cuisine_id
                if predictions
                else self.NO_PREDICTION
            )
            if predicted_id not in matrix[example.expected_cuisine_id]:
                for row in matrix.values():
                    row[predicted_id] = 0
            matrix[example.expected_cuisine_id][predicted_id] += 1

            if predicted_id == self.NO_PREDICTION:
                no_prediction_count += 1
            if predicted_id == example.expected_cuisine_id:
                correct += 1
            if example.expected_cuisine_id in {
                prediction.cuisine.cuisine_id
                for prediction in predictions[: self._top_k]
            }:
                top_k_correct += 1

        per_class = tuple(
            self._calculate_class_metrics(cuisine_id, matrix)
            for cuisine_id in expected_labels
        )
        total = len(examples)
        return ClassificationEvaluation(
            total_examples=total,
            evaluated_examples=total - no_prediction_count,
            no_prediction_count=no_prediction_count,
            accuracy=correct / total,
            top_k_accuracy=top_k_correct / total,
            macro_precision=sum(item.precision for item in per_class) / len(per_class),
            macro_recall=sum(item.recall for item in per_class) / len(per_class),
            macro_f1=sum(item.f1 for item in per_class) / len(per_class),
            per_class=per_class,
            confusion_matrix=matrix,
        )

    @classmethod
    def _empty_matrix(cls, labels: list[str]) -> dict[str, dict[str, int]]:
        prediction_labels = labels + [cls.NO_PREDICTION]
        return {
            expected: {predicted: 0 for predicted in prediction_labels}
            for expected in labels
        }

    @staticmethod
    def _calculate_class_metrics(
        cuisine_id: str,
        matrix: dict[str, dict[str, int]],
    ) -> CuisineClassMetrics:
        true_positive = matrix[cuisine_id].get(cuisine_id, 0)
        false_negative = sum(matrix[cuisine_id].values()) - true_positive
        false_positive = sum(
            row.get(cuisine_id, 0)
            for expected, row in matrix.items()
            if expected != cuisine_id
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return CuisineClassMetrics(
            cuisine_id=cuisine_id,
            support=sum(matrix[cuisine_id].values()),
            precision=precision,
            recall=recall,
            f1=f1,
        )
