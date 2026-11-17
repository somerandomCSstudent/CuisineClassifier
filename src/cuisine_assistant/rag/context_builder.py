from __future__ import annotations

import re

from cuisine_assistant.application import SearchRecipesResult
from cuisine_assistant.rag.models import (
    RagContext,
    RagDocumentEvidence,
    RagSourceReference,
)
from cuisine_assistant.rag.ports import RagKnowledgeGraphFactProvider


class RagContextBuilder:
    """Builds bounded grounded context from recipe retrieval and RDF facts."""

    def __init__(
        self,
        fact_provider: RagKnowledgeGraphFactProvider,
        *,
        maximum_documents: int = 3,
        maximum_graph_facts: int = 20,
        maximum_excerpt_characters: int = 700,
    ) -> None:
        if maximum_documents < 1:
            raise ValueError("maximum_documents must be at least 1")
        if maximum_graph_facts < 1:
            raise ValueError("maximum_graph_facts must be at least 1")
        if maximum_excerpt_characters < 50:
            raise ValueError(
                "maximum_excerpt_characters must be at least 50"
            )
        self._fact_provider = fact_provider
        self._maximum_documents = maximum_documents
        self._maximum_graph_facts = maximum_graph_facts
        self._maximum_excerpt_characters = maximum_excerpt_characters

    def build(self, result: SearchRecipesResult) -> RagContext:
        """Converts top retrieval hits and linked entities into prompt context."""
        selected_hits = result.search.hits[: self._maximum_documents]
        documents = tuple(
            RagDocumentEvidence(
                recipe_id=hit.document.recipe_id,
                title=hit.document.title,
                excerpt=self._excerpt(hit.document.searchable_text),
                ingredients=self._ingredients(
                    hit.document.ingredient_text
                ),
                cuisine_id=hit.document.cuisine_id,
                retrieval_score=hit.score,
            )
            for hit in selected_hits
        )
        sources = tuple(
            RagSourceReference(
                recipe_id=hit.document.recipe_id,
                dish_id=hit.document.dish_id,
                title=hit.document.title,
                source=hit.document.source,
                source_url=hit.document.source_url,
                license_name=hit.document.license_name,
                retrieval_score=hit.score,
            )
            for hit in selected_hits
        )
        graph_facts = self._fact_provider.facts_for_dishes(
            tuple(hit.document.dish_id for hit in selected_hits),
            self._maximum_graph_facts,
        )
        linked_ingredient_ids = tuple(
            dict.fromkeys(
                linked.ingredient.ingredient_id
                for linked in result.linking.linked_ingredients
                if linked.ingredient is not None
            )
        )
        cuisine_id = (
            result.cuisine_preference.cuisine_id
            if result.cuisine_preference is not None
            else None
        )
        return RagContext(
            question=result.query_text,
            documents=documents,
            graph_facts=graph_facts,
            sources=sources,
            linked_ingredient_ids=linked_ingredient_ids,
            cuisine_id=cuisine_id,
        )

    def _excerpt(self, text: str) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= self._maximum_excerpt_characters:
            return normalized
        shortened = normalized[: self._maximum_excerpt_characters + 1]
        boundary = shortened.rfind(" ")
        if boundary >= self._maximum_excerpt_characters // 2:
            shortened = shortened[:boundary]
        return shortened.rstrip(" ,.;:") + "..."

    @staticmethod
    def _ingredients(text: str) -> tuple[str, ...]:
        values = re.split(r"[;,|]", text)
        return tuple(
            dict.fromkeys(
                value.strip()
                for value in values
                if value.strip()
            )
        )
