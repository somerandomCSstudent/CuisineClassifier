from __future__ import annotations

from types import SimpleNamespace

from cuisine_assistant.cli.console import ConsoleApplication


class StubQuestionUseCase:
    """Captures the CLI question and returns a configured result."""

    def __init__(self, result) -> None:
        self.result = result
        self.received_question = None

    def execute(self, question: str):
        self.received_question = question
        return self.result


class StubFormatter:
    """Returns fixed text and captures the formatted result."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.received = None

    def format(self, result):
        self.received = result
        return self.text


def test_cli_option_five_runs_grounded_question_answering(
    monkeypatch,
    capsys,
) -> None:
    """Verifies menu option five sends the question through the QA pipeline."""
    qa_result = SimpleNamespace()
    qa_use_case = StubQuestionUseCase(qa_result)
    qa_formatter = StubFormatter("Grounded answer [D1]")
    inputs = iter(("5", "What dish uses paprika?", "6"))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    application = ConsoleApplication(
        knowledge_base=SimpleNamespace(name="Test KB", version="1.0"),
        link_ingredients=SimpleNamespace(),
        classify_recipe=SimpleNamespace(),
        recommend_dishes=SimpleNamespace(),
        search_recipes=SimpleNamespace(),
        ask_cuisine_question=qa_use_case,
        link_formatter=SimpleNamespace(),
        classification_formatter=SimpleNamespace(),
        recommendation_formatter=SimpleNamespace(),
        recipe_search_formatter=SimpleNamespace(),
        grounded_answer_formatter=qa_formatter,
    )

    application.run()

    output = capsys.readouterr().out
    assert "5 - Ask a grounded cuisine question" in output
    assert "Grounded answer [D1]" in output
    assert qa_use_case.received_question == "What dish uses paprika?"
    assert qa_formatter.received is qa_result
