from __future__ import annotations

import pytest

from cuisine_assistant.rag import (
    RagContext,
    RagDocumentEvidence,
    RagGraphFact,
    RagSourceReference,
    TemplateGroundedAnswerGenerator,
)


def complete_context() -> RagContext:
    """Creates one grounded context with document, graph, and source evidence."""
    return RagContext(
        question="What Hungarian dish uses chicken and paprika?",
        documents=(
            RagDocumentEvidence(
                recipe_id="r001",
                title="Traditional Chicken Paprikash",
                excerpt="Chicken is cooked with paprika and sour cream.",
                ingredients=("chicken", "paprika", "sour cream"),
                cuisine_id="hungarian",
                retrieval_score=0.94,
            ),
        ),
        graph_facts=(
            RagGraphFact(
                "ChickenPaprikash",
                "belongsToCuisine",
                "Hungarian cuisine",
                "urn:test#ChickenPaprikash",
            ),
            RagGraphFact(
                "ChickenPaprikash",
                "hasIngredient",
                "paprika",
                "urn:test#ChickenPaprikash",
            ),
        ),
        sources=(
            RagSourceReference(
                recipe_id="r001",
                dish_id="chicken_paprikash",
                title="Traditional Chicken Paprikash",
                source="manual",
                source_url=None,
                license_name=None,
                retrieval_score=0.94,
            ),
        ),
        linked_ingredient_ids=("chicken", "paprika"),
        cuisine_id="hungarian",
    )


def test_generates_answer_only_from_context_evidence() -> None:
    """Verifies document and graph claims carry deterministic citations."""
    answer = TemplateGroundedAnswerGenerator().generate(complete_context())

    assert "Traditional Chicken Paprikash" in answer.text
    assert "chicken, paprika, sour cream" in answer.text
    assert "[D1]" in answer.text
    assert "[G1]" in answer.text
    assert "[G2]" in answer.text
    assert tuple(citation.reference_id for citation in answer.citations) == (
        "D1",
        "G1",
        "G2",
    )
    assert answer.source_recipe_ids == ("r001",)
    assert answer.insufficient_evidence is False


def test_returns_explicit_insufficient_evidence_answer() -> None:
    """Verifies empty retrieval context never produces an invented answer."""
    context = RagContext(question="Unknown cuisine question")

    answer = TemplateGroundedAnswerGenerator().generate(context)

    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert answer.source_recipe_ids == ()
    assert "could not find enough grounded evidence" in answer.text


def test_limits_number_of_graph_claims() -> None:
    """Verifies generated answers remain bounded for large RDF contexts."""
    answer = TemplateGroundedAnswerGenerator(
        maximum_graph_facts=1
    ).generate(complete_context())

    assert "[G1]" in answer.text
    assert "[G2]" not in answer.text
    assert tuple(citation.reference_id for citation in answer.citations) == (
        "D1",
        "G1",
    )


def test_rejects_invalid_graph_fact_limit() -> None:
    """Verifies answer generation limits remain positive."""
    with pytest.raises(ValueError, match="maximum_graph_facts"):
        TemplateGroundedAnswerGenerator(maximum_graph_facts=0)
