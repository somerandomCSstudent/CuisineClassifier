from __future__ import annotations

import pytest

from cuisine_assistant.rag import GroundedAnswer, GroundedCitation


def test_rejects_duplicate_citation_identifiers() -> None:
    """Verifies each evidence marker identifies one unique citation."""
    citation = GroundedCitation("D1", "document", "Recipe")

    with pytest.raises(ValueError, match="unique"):
        GroundedAnswer("Answer [D1]", (citation, citation))


def test_rejects_citations_for_insufficient_evidence() -> None:
    """Verifies abstaining answers cannot cite nonexistent evidence."""
    citation = GroundedCitation("D1", "document", "Recipe")

    with pytest.raises(ValueError, match="must not contain citations"):
        GroundedAnswer(
            "Insufficient evidence",
            (citation,),
            insufficient_evidence=True,
        )


def test_rejects_unknown_citation_type() -> None:
    """Verifies citation types stay within supported grounded evidence kinds."""
    with pytest.raises(ValueError, match="reference_type"):
        GroundedCitation("X1", "unknown", "Unsupported")
