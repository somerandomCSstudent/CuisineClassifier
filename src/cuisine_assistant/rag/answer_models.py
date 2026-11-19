from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GroundedCitation:
    """Identifies one document or graph fact supporting a generated statement."""

    reference_id: str
    reference_type: str
    description: str

    def __post_init__(self) -> None:
        if not self.reference_id or not self.reference_id.strip():
            raise ValueError("reference_id must not be empty")
        if self.reference_type not in {"document", "graph_fact", "source"}:
            raise ValueError(
                "reference_type must be document, graph_fact, or source"
            )
        if not self.description or not self.description.strip():
            raise ValueError("description must not be empty")


@dataclass(frozen=True)
class GroundedAnswer:
    """Contains a context-grounded answer and its explicit evidence references."""

    text: str
    citations: tuple[GroundedCitation, ...] = field(default_factory=tuple)
    source_recipe_ids: tuple[str, ...] = field(default_factory=tuple)
    insufficient_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("text must not be empty")
        citation_ids = tuple(citation.reference_id for citation in self.citations)
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("citation reference IDs must be unique")
        if self.insufficient_evidence and self.citations:
            raise ValueError(
                "insufficient-evidence answers must not contain citations"
            )
