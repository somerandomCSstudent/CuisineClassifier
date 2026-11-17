from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RagSourceReference:
    """Identifies one recipe document used as grounded RAG evidence."""

    recipe_id: str
    dish_id: str
    title: str
    source: str
    source_url: str | None
    license_name: str | None
    retrieval_score: float

    def __post_init__(self) -> None:
        for field_name in ("recipe_id", "dish_id", "title", "source"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not 0.0 <= self.retrieval_score <= 1.0:
            raise ValueError("retrieval_score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class RagGraphFact:
    """Represents one human-readable fact grounded in the active RDF graph."""

    subject_id: str
    predicate: str
    object_value: str
    source_iri: str

    def __post_init__(self) -> None:
        for field_name in (
            "subject_id",
            "predicate",
            "object_value",
            "source_iri",
        ):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class RagDocumentEvidence:
    """Contains one retrieved recipe excerpt prepared for grounded generation."""

    recipe_id: str
    title: str
    excerpt: str
    ingredients: tuple[str, ...]
    cuisine_id: str | None
    retrieval_score: float

    def __post_init__(self) -> None:
        if not self.recipe_id.strip():
            raise ValueError("recipe_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.excerpt.strip():
            raise ValueError("excerpt must not be empty")
        if not 0.0 <= self.retrieval_score <= 1.0:
            raise ValueError("retrieval_score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class RagContext:
    """Contains bounded document evidence, RDF facts, and source references."""

    question: str
    documents: tuple[RagDocumentEvidence, ...] = field(default_factory=tuple)
    graph_facts: tuple[RagGraphFact, ...] = field(default_factory=tuple)
    sources: tuple[RagSourceReference, ...] = field(default_factory=tuple)
    linked_ingredient_ids: tuple[str, ...] = field(default_factory=tuple)
    cuisine_id: str | None = None

    def __post_init__(self) -> None:
        if not self.question or not self.question.strip():
            raise ValueError("question must not be empty")
        source_ids = {source.recipe_id for source in self.sources}
        document_ids = {document.recipe_id for document in self.documents}
        if source_ids != document_ids:
            raise ValueError(
                "RAG source references must match document evidence"
            )

    @property
    def is_empty(self) -> bool:
        """Returns whether no document and no graph evidence is available."""
        return not self.documents and not self.graph_facts

    def to_prompt_context(self) -> str:
        """Renders deterministic grounded context for a later answer generator."""
        sections: list[str] = [f"Question: {self.question}"]
        if self.linked_ingredient_ids:
            sections.append(
                "Linked ingredients: "
                + ", ".join(self.linked_ingredient_ids)
            )
        if self.cuisine_id:
            sections.append(f"Cuisine preference: {self.cuisine_id}")

        if self.documents:
            sections.append("\nRetrieved recipe documents:")
            for index, document in enumerate(self.documents, start=1):
                sections.append(
                    f"[D{index}] {document.title} "
                    f"(recipe_id={document.recipe_id}, "
                    f"score={document.retrieval_score:.4f})"
                )
                if document.cuisine_id:
                    sections.append(f"Cuisine: {document.cuisine_id}")
                if document.ingredients:
                    sections.append(
                        "Ingredients: " + ", ".join(document.ingredients)
                    )
                sections.append(f"Excerpt: {document.excerpt}")

        if self.graph_facts:
            sections.append("\nKnowledge graph facts:")
            for index, fact in enumerate(self.graph_facts, start=1):
                sections.append(
                    f"[G{index}] {fact.subject_id} "
                    f"{fact.predicate} {fact.object_value}"
                )

        if self.sources:
            sections.append("\nSources:")
            for index, source in enumerate(self.sources, start=1):
                location = source.source_url or source.source
                sections.append(
                    f"[S{index}] {source.title} | {location} | "
                    f"recipe_id={source.recipe_id}"
                )
        return "\n".join(sections)
