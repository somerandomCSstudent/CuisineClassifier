from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.application import AskCuisineQuestionUseCase
from cuisine_assistant.knowledge_base.models import KnowledgeBaseBundle
from cuisine_assistant.rag import (
    RagContextBuilder,
    RdfLibRagFactProvider,
    TemplateGroundedAnswerGenerator,
)


@dataclass(frozen=True)
class QuestionAnswerSettings:
    """Defines bounded RAG context and deterministic answer-generation limits."""

    maximum_rag_documents: int = 3
    maximum_rag_graph_facts: int = 20
    maximum_rag_excerpt_characters: int = 700
    maximum_answer_graph_facts: int = 4

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_rag_documents",
            "maximum_rag_graph_facts",
            "maximum_answer_graph_facts",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1")
        if self.maximum_rag_excerpt_characters < 50:
            raise ValueError(
                "maximum_rag_excerpt_characters must be at least 50"
            )


class QuestionAnswerUseCaseFactory:
    """Builds question answering from the selected bundle and recipe search."""

    def create(
        self,
        bundle: KnowledgeBaseBundle,
        search_recipes,
        settings: QuestionAnswerSettings,
    ) -> AskCuisineQuestionUseCase:
        """Creates bundle-scoped grounding and deterministic answer generation."""
        fact_provider = RdfLibRagFactProvider(
            ontology_path=bundle.paths.ontology,
            dishes_csv=bundle.paths.dishes,
            ontology_base_iri=bundle.metadata.ontology_base_iri,
        )
        context_builder = RagContextBuilder(
            fact_provider=fact_provider,
            maximum_documents=settings.maximum_rag_documents,
            maximum_graph_facts=settings.maximum_rag_graph_facts,
            maximum_excerpt_characters=(
                settings.maximum_rag_excerpt_characters
            ),
        )
        answer_generator = TemplateGroundedAnswerGenerator(
            maximum_graph_facts=settings.maximum_answer_graph_facts,
        )
        return AskCuisineQuestionUseCase(
            search_recipes=search_recipes,
            context_builder=context_builder,
            answer_generator=answer_generator,
        )
