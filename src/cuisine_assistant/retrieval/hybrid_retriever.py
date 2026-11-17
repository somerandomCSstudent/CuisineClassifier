from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.retrieval.models import (
    RecipeSearchEvidence,
    RecipeSearchHit,
    RecipeSearchQuery,
    RecipeSearchResult,
)
from cuisine_assistant.retrieval.ports import RecipeRetriever


@dataclass(frozen=True)
class _FusedCandidate:
    """Stores accumulated reciprocal-rank evidence for one recipe document."""

    document: object
    score: float
    evidence: RecipeSearchEvidence


class HybridRecipeRetriever(RecipeRetriever):
    """Fuses BM25 and embedding rankings with weighted Reciprocal Rank Fusion."""

    def __init__(
        self,
        lexical_retriever: RecipeRetriever,
        embedding_retriever: RecipeRetriever,
        *,
        rrf_constant: int = 60,
        lexical_weight: float = 1.0,
        embedding_weight: float = 1.0,
        candidate_multiplier: int = 3,
    ) -> None:
        if rrf_constant < 1:
            raise ValueError("rrf_constant must be at least 1")
        if lexical_weight < 0:
            raise ValueError("lexical_weight must not be negative")
        if embedding_weight < 0:
            raise ValueError("embedding_weight must not be negative")
        if lexical_weight + embedding_weight == 0:
            raise ValueError("at least one retriever weight must be positive")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")

        self._lexical_retriever = lexical_retriever
        self._embedding_retriever = embedding_retriever
        self._rrf_constant = rrf_constant
        self._lexical_weight = lexical_weight
        self._embedding_weight = embedding_weight
        self._candidate_multiplier = candidate_multiplier

    def search(self, query: RecipeSearchQuery) -> RecipeSearchResult:
        """Returns a deterministic top-k ranking fused from both retrievers."""
        candidate_limit = query.maximum_results * self._candidate_multiplier
        candidate_query = RecipeSearchQuery(
            text=query.text,
            maximum_results=candidate_limit,
            cuisine_id=query.cuisine_id,
            required_ingredient_ids=query.required_ingredient_ids,
        )
        lexical_result = self._lexical_retriever.search(candidate_query)
        embedding_result = self._embedding_retriever.search(candidate_query)

        accumulated_scores: dict[str, float] = {}
        documents: dict[str, object] = {}
        evidence: dict[str, RecipeSearchEvidence] = {}

        self._accumulate(
            lexical_result,
            self._lexical_weight,
            accumulated_scores,
            documents,
            evidence,
        )
        self._accumulate(
            embedding_result,
            self._embedding_weight,
            accumulated_scores,
            documents,
            evidence,
        )

        if not accumulated_scores:
            return RecipeSearchResult(query=query)

        maximum_score = max(accumulated_scores.values())
        candidates = [
            _FusedCandidate(
                document=documents[recipe_id],
                score=(score / maximum_score if maximum_score > 0 else 0.0),
                evidence=evidence[recipe_id],
            )
            for recipe_id, score in accumulated_scores.items()
        ]
        candidates.sort(
            key=lambda item: (-item.score, item.document.recipe_id)
        )
        hits = tuple(
            RecipeSearchHit(
                document=candidate.document,
                score=candidate.score,
                rank=rank,
                evidence=candidate.evidence,
            )
            for rank, candidate in enumerate(
                candidates[: query.maximum_results],
                start=1,
            )
        )
        return RecipeSearchResult(query=query, hits=hits)

    def _accumulate(
        self,
        result: RecipeSearchResult,
        weight: float,
        accumulated_scores: dict[str, float],
        documents: dict[str, object],
        evidence: dict[str, RecipeSearchEvidence],
    ) -> None:
        if weight == 0:
            return
        for hit in result.hits:
            recipe_id = hit.document.recipe_id
            contribution = weight / (self._rrf_constant + hit.rank)
            accumulated_scores[recipe_id] = (
                accumulated_scores.get(recipe_id, 0.0) + contribution
            )
            documents.setdefault(recipe_id, hit.document)
            prior = evidence.get(recipe_id)
            evidence[recipe_id] = self._merge_evidence(
                prior,
                hit.evidence,
            )

    @staticmethod
    def _merge_evidence(
        previous: RecipeSearchEvidence | None,
        current: RecipeSearchEvidence,
    ) -> RecipeSearchEvidence:
        if previous is None:
            return current
        return RecipeSearchEvidence(
            matched_terms=tuple(
                dict.fromkeys(
                    (*previous.matched_terms, *current.matched_terms)
                )
            ),
            matched_ingredient_ids=tuple(
                dict.fromkeys(
                    (
                        *previous.matched_ingredient_ids,
                        *current.matched_ingredient_ids,
                    )
                )
            ),
            title_match_score=max(
                previous.title_match_score,
                current.title_match_score,
            ),
            content_match_score=max(
                previous.content_match_score,
                current.content_match_score,
            ),
            cuisine_match_score=max(
                previous.cuisine_match_score,
                current.cuisine_match_score,
            ),
        )
