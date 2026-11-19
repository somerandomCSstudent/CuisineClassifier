from __future__ import annotations

from cuisine_assistant.rag.answer_models import (
    GroundedAnswer,
    GroundedCitation,
)
from cuisine_assistant.rag.answer_ports import GroundedAnswerGenerator
from cuisine_assistant.rag.models import RagContext, RagDocumentEvidence


class TemplateGroundedAnswerGenerator(GroundedAnswerGenerator):
    """Builds deterministic answers exclusively from retrieved documents and RDF facts."""

    _INSUFFICIENT_TEXT = (
        "I could not find enough grounded evidence to answer the question."
    )

    def __init__(self, maximum_graph_facts: int = 4) -> None:
        if maximum_graph_facts < 1:
            raise ValueError("maximum_graph_facts must be at least 1")
        self._maximum_graph_facts = maximum_graph_facts

    def generate(self, context: RagContext) -> GroundedAnswer:
        """Returns a concise answer whose claims point to context evidence IDs."""
        if context.is_empty or not context.documents:
            return GroundedAnswer(
                text=self._INSUFFICIENT_TEXT,
                insufficient_evidence=True,
            )

        best_document = context.documents[0]
        statements = [self._document_statement(best_document)]
        citations = [
            GroundedCitation(
                reference_id="D1",
                reference_type="document",
                description=best_document.title,
            )
        ]

        selected_facts = context.graph_facts[: self._maximum_graph_facts]
        for index, fact in enumerate(selected_facts, start=1):
            reference_id = f"G{index}"
            statements.append(
                f"The knowledge graph states that {fact.subject_id} "
                f"{self._humanize_predicate(fact.predicate)} "
                f"{fact.object_value}. [{reference_id}]"
            )
            citations.append(
                GroundedCitation(
                    reference_id=reference_id,
                    reference_type="graph_fact",
                    description=(
                        f"{fact.subject_id} {fact.predicate} "
                        f"{fact.object_value}"
                    ),
                )
            )

        source_recipe_ids = tuple(
            source.recipe_id for source in context.sources
        )
        return GroundedAnswer(
            text=" ".join(statements),
            citations=tuple(citations),
            source_recipe_ids=source_recipe_ids,
        )

    @staticmethod
    def _document_statement(document: RagDocumentEvidence) -> str:
        cuisine = (
            f" from {document.cuisine_id} cuisine"
            if document.cuisine_id
            else ""
        )
        ingredient_text = (
            ", ".join(document.ingredients)
            if document.ingredients
            else "the ingredients described in the retrieved recipe"
        )
        return (
            f"{document.title} is the strongest retrieved recipe{cuisine}. "
            f"The retrieved document lists {ingredient_text}. [D1]"
        )

    @staticmethod
    def _humanize_predicate(predicate: str) -> str:
        mapping = {
            "belongsToCuisine": "belongs to cuisine",
            "hasIngredient": "has ingredient",
            "type": "has RDF type",
            "label": "has label",
        }
        return mapping.get(predicate, predicate)
