from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuisine_assistant.application.ask_cuisine_question import (
    AskCuisineQuestionUseCase,
)
from cuisine_assistant.rag import GroundedAnswer, RagContext


class StubSearchRecipesUseCase:
    """Returns a configured recipe-search result and captures the question."""

    def __init__(self, result) -> None:
        self._result = result
        self.received_question: str | None = None

    def execute(self, question: str):
        self.received_question = question
        return self._result


class StubRagContextBuilder:
    """Returns a configured RAG context and captures the search result."""

    def __init__(self, context: RagContext) -> None:
        self._context = context
        self.received_search_result = None

    def build(self, search_result):
        self.received_search_result = search_result
        return self._context


class StubGroundedAnswerGenerator:
    """Returns a configured answer and captures the supplied RAG context."""

    def __init__(self, answer: GroundedAnswer) -> None:
        self._answer = answer
        self.received_context: RagContext | None = None

    def generate(self, context: RagContext) -> GroundedAnswer:
        self.received_context = context
        return self._answer


def test_executes_search_context_and_answer_pipeline() -> None:
    """Verifies the question moves through retrieval, grounding, and generation."""
    search_result = SimpleNamespace(query_text="question")
    context = RagContext(question="What Hungarian dish uses paprika?")
    answer = GroundedAnswer(
        "Chicken Paprikash is supported by the retrieved evidence. [D1]",
        source_recipe_ids=("r001",),
    )
    search = StubSearchRecipesUseCase(search_result)
    builder = StubRagContextBuilder(context)
    generator = StubGroundedAnswerGenerator(answer)
    use_case = AskCuisineQuestionUseCase(search, builder, generator)

    result = use_case.execute(
        "  What Hungarian dish uses paprika?  "
    )

    assert result.question == "What Hungarian dish uses paprika?"
    assert search.received_question == result.question
    assert builder.received_search_result is search_result
    assert generator.received_context is context
    assert result.search_result is search_result
    assert result.context is context
    assert result.answer is answer
    assert result.has_grounded_answer is True


def test_preserves_insufficient_evidence_answer() -> None:
    """Verifies the use case exposes generator abstention without inventing facts."""
    search_result = SimpleNamespace(query_text="unknown")
    context = RagContext(question="Unknown dish question")
    answer = GroundedAnswer(
        "I could not find enough grounded evidence to answer the question.",
        insufficient_evidence=True,
    )
    use_case = AskCuisineQuestionUseCase(
        StubSearchRecipesUseCase(search_result),
        StubRagContextBuilder(context),
        StubGroundedAnswerGenerator(answer),
    )

    result = use_case.execute("Unknown dish question")

    assert result.answer.insufficient_evidence is True
    assert result.has_grounded_answer is False


def test_rejects_blank_question_before_running_dependencies() -> None:
    """Verifies an empty question cannot invoke retrieval or generation."""
    search = StubSearchRecipesUseCase(SimpleNamespace())
    builder = StubRagContextBuilder(RagContext(question="placeholder"))
    generator = StubGroundedAnswerGenerator(GroundedAnswer("placeholder"))
    use_case = AskCuisineQuestionUseCase(search, builder, generator)

    with pytest.raises(ValueError, match="question"):
        use_case.execute("   ")

    assert search.received_question is None
    assert builder.received_search_result is None
    assert generator.received_context is None
