from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.application.search_recipes import (
    SearchRecipesResult,
    SearchRecipesUseCase,
)
from cuisine_assistant.rag.answer_models import (
    GroundedAnswer,
)
from cuisine_assistant.rag.answer_ports import (
    GroundedAnswerGenerator,
)
from cuisine_assistant.rag.context_builder import (
    RagContextBuilder,
)
from cuisine_assistant.rag.models import (
    RagContext,
)

@dataclass(frozen=True)
class AskCuisineQuestionResult:
    """Contains retrieval, grounding context, and the final supported answer."""

    question: str
    search_result: SearchRecipesResult
    context: RagContext
    answer: GroundedAnswer

    @property
    def has_grounded_answer(self) -> bool:
        """Returns whether the answer is supported by retrieved evidence."""
        return not self.answer.insufficient_evidence


class AskCuisineQuestionUseCase:
    """Searches recipes, builds RDF-grounded context, and generates an answer."""

    def __init__(
        self,
        search_recipes: SearchRecipesUseCase,
        context_builder: RagContextBuilder,
        answer_generator: GroundedAnswerGenerator,
    ) -> None:
        self._search_recipes = search_recipes
        self._context_builder = context_builder
        self._answer_generator = answer_generator

    def execute(self, question: str) -> AskCuisineQuestionResult:
        """Answers one English cuisine question from bundle evidence only."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        search_result = self._search_recipes.execute(normalized_question)
        context = self._context_builder.build(search_result)
        answer = self._answer_generator.generate(context)
        return AskCuisineQuestionResult(
            question=normalized_question,
            search_result=search_result,
            context=context,
            answer=answer,
        )
