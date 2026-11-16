from __future__ import annotations

from cuisine_assistant.evaluation.cuisine_classification import (
    ClassificationEvaluation,
)


class ClassificationEvaluationFormatter:
    """Formats classification metrics and a confusion matrix as plain text."""

    def format(self, evaluation: ClassificationEvaluation) -> str:
        lines = [
            "Cuisine classification evaluation",
            f"Examples: {evaluation.total_examples}",
            f"No prediction: {evaluation.no_prediction_count}",
            f"Accuracy: {evaluation.accuracy:.2%}",
            f"Top-k accuracy: {evaluation.top_k_accuracy:.2%}",
            f"Macro precision: {evaluation.macro_precision:.2%}",
            f"Macro recall: {evaluation.macro_recall:.2%}",
            f"Macro F1: {evaluation.macro_f1:.2%}",
            "",
            "Per-class metrics:",
        ]
        for item in evaluation.per_class:
            lines.append(
                f"- {item.cuisine_id}: support={item.support}, "
                f"precision={item.precision:.2%}, recall={item.recall:.2%}, "
                f"F1={item.f1:.2%}"
            )
        lines.extend(["", "Confusion matrix:"])
        labels = sorted(
            {
                predicted
                for row in evaluation.confusion_matrix.values()
                for predicted in row
            }
        )
        lines.append("expected\\predicted," + ",".join(labels))
        for expected, row in sorted(evaluation.confusion_matrix.items()):
            lines.append(
                expected + "," + ",".join(str(row.get(label, 0)) for label in labels)
            )
        return "\n".join(lines)
