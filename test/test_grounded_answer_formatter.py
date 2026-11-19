from __future__ import annotations

from types import SimpleNamespace

from cuisine_assistant.cli.grounded_answer_formatter import (
    GroundedAnswerFormatter,
)
from cuisine_assistant.rag import (
    GroundedAnswer,
    GroundedCitation,
    RagContext,
    RagDocumentEvidence,
    RagSourceReference,
)


def grounded_result():
    """Creates one complete grounded answer result for formatter tests."""
    context = RagContext(
        question="What Hungarian dish uses paprika?",
        documents=(
            RagDocumentEvidence(
                recipe_id="r001",
                title="Traditional Chicken Paprikash",
                excerpt="Chicken with paprika and sour cream.",
                ingredients=("chicken", "paprika", "sour cream"),
                cuisine_id="hungarian",
                retrieval_score=0.95,
            ),
        ),
        sources=(
            RagSourceReference(
                recipe_id="r001",
                dish_id="chicken_paprikash",
                title="Traditional Chicken Paprikash",
                source="manual",
                source_url="https://example.org/r001",
                license_name="CC BY 4.0",
                retrieval_score=0.95,
            ),
        ),
        linked_ingredient_ids=("chicken", "paprika"),
        cuisine_id="hungarian",
    )
    answer = GroundedAnswer(
        text="Chicken Paprikash is supported by the evidence. [D1] [G1]",
        citations=(
            GroundedCitation(
                "D1",
                "document",
                "Traditional Chicken Paprikash",
            ),
            GroundedCitation(
                "G1",
                "graph_fact",
                "ChickenPaprikash belongsToCuisine Hungarian cuisine",
            ),
        ),
        source_recipe_ids=("r001",),
    )
    return SimpleNamespace(answer=answer, context=context)


def test_formats_answer_evidence_sources_and_entities() -> None:
    """Verifies the CLI report exposes all grounded output sections."""
    output = GroundedAnswerFormatter().format(grounded_result())

    assert "Grounded cuisine answer:" in output
    assert "[D1] document: Traditional Chicken Paprikash" in output
    assert "[G1] graph_fact:" in output
    assert "[S1] Traditional Chicken Paprikash" in output
    assert "License: CC BY 4.0" in output
    assert "chicken, paprika" in output
    assert "Detected cuisine preference:\nhungarian" in output


def test_formats_insufficient_evidence_without_fake_sources() -> None:
    """Verifies abstaining answers do not display nonexistent evidence."""
    result = SimpleNamespace(
        answer=GroundedAnswer(
            "I could not find enough grounded evidence to answer the question.",
            insufficient_evidence=True,
        ),
        context=RagContext(question="Unknown question"),
    )

    output = GroundedAnswerFormatter().format(result)

    assert "insufficient grounded evidence" in output
    assert "Evidence:" not in output
    assert "Sources:" not in output
