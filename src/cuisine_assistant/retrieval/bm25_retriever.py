from __future__ import annotations

import math
import re
from collections import Counter

from cuisine_assistant.retrieval.models import (
    RecipeDocument,
    RecipeSearchEvidence,
    RecipeSearchHit,
    RecipeSearchQuery,
    RecipeSearchResult,
)
from cuisine_assistant.retrieval.ports import (
    RecipeDocumentRepository,
    RecipeRetriever,
)


class Bm25RecipeRetriever(RecipeRetriever):
    """Ranks recipe documents with BM25 and optional cuisine and ingredient signals."""

    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)

    def __init__(
        self,
        repository: RecipeDocumentRepository,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: float = 0.20,
        cuisine_weight: float = 0.15,
        ingredient_weight: float = 0.25,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0.0 and 1.0")
        for name, value in (
            ("title_weight", title_weight),
            ("cuisine_weight", cuisine_weight),
            ("ingredient_weight", ingredient_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        self._repository = repository
        self._k1 = k1
        self._b = b
        self._title_weight = title_weight
        self._cuisine_weight = cuisine_weight
        self._ingredient_weight = ingredient_weight
        self._documents = repository.list_all()
        self._document_tokens = tuple(
            self._tokenize(document.searchable_text)
            for document in self._documents
        )
        self._document_frequencies = self._build_document_frequencies(
            self._document_tokens
        )
        self._average_document_length = (
            sum(len(tokens) for tokens in self._document_tokens)
            / len(self._document_tokens)
            if self._document_tokens
            else 0.0
        )

    def search(self, query: RecipeSearchQuery) -> RecipeSearchResult:
        """Returns top-k documents using normalized BM25 and semantic boosts."""
        query_tokens = tuple(dict.fromkeys(self._tokenize(query.text)))
        if not query_tokens or not self._documents:
            return RecipeSearchResult(query=query)

        raw_candidates: list[tuple[RecipeDocument, float, tuple[str, ...]]] = []
        for document, tokens in zip(self._documents, self._document_tokens):
            if query.cuisine_id and document.cuisine_id != query.cuisine_id:
                continue
            raw_score, matched_terms = self._bm25_score(query_tokens, tokens)
            raw_candidates.append((document, raw_score, matched_terms))

        maximum_bm25 = max(
            (candidate[1] for candidate in raw_candidates),
            default=0.0,
        )
        scored = []
        for document, raw_score, matched_terms in raw_candidates:
            bm25_score = raw_score / maximum_bm25 if maximum_bm25 > 0 else 0.0
            title_score = self._term_overlap(
                query_tokens,
                self._tokenize(document.title),
            )
            matched_ingredients = self._matched_ingredients(query, document)
            ingredient_score = (
                len(matched_ingredients) / len(query.required_ingredient_ids)
                if query.required_ingredient_ids
                else 0.0
            )
            cuisine_score = (
                1.0
                if query.cuisine_id and document.cuisine_id == query.cuisine_id
                else 0.0
            )
            lexical_weight = 1.0 - (
                self._title_weight
                + (self._cuisine_weight if query.cuisine_id else 0.0)
                + (
                    self._ingredient_weight
                    if query.required_ingredient_ids
                    else 0.0
                )
            )
            lexical_weight = max(0.0, lexical_weight)
            score = (
                lexical_weight * bm25_score
                + self._title_weight * title_score
                + self._cuisine_weight * cuisine_score
                + self._ingredient_weight * ingredient_score
            )
            score = min(1.0, score)
            if score <= 0.0:
                continue
            evidence = RecipeSearchEvidence(
                matched_terms=matched_terms,
                matched_ingredient_ids=matched_ingredients,
                title_match_score=title_score,
                content_match_score=bm25_score,
                cuisine_match_score=cuisine_score,
            )
            scored.append((document, score, evidence))

        scored.sort(key=lambda item: (-item[1], item[0].recipe_id))
        hits = tuple(
            RecipeSearchHit(document, score, rank, evidence)
            for rank, (document, score, evidence) in enumerate(
                scored[: query.maximum_results],
                start=1,
            )
        )
        return RecipeSearchResult(query=query, hits=hits)

    def _bm25_score(
        self,
        query_tokens: tuple[str, ...],
        document_tokens: tuple[str, ...],
    ) -> tuple[float, tuple[str, ...]]:
        frequencies = Counter(document_tokens)
        document_length = len(document_tokens)
        matched_terms = tuple(
            token for token in query_tokens if frequencies[token] > 0
        )
        score = 0.0
        for token in matched_terms:
            document_frequency = self._document_frequencies[token]
            inverse_document_frequency = math.log(
                1.0
                + (
                    len(self._documents) - document_frequency + 0.5
                ) / (document_frequency + 0.5)
            )
            term_frequency = frequencies[token]
            length_ratio = (
                document_length / self._average_document_length
                if self._average_document_length > 0
                else 0.0
            )
            denominator = term_frequency + self._k1 * (
                1.0 - self._b + self._b * length_ratio
            )
            score += inverse_document_frequency * (
                term_frequency * (self._k1 + 1.0) / denominator
            )
        return score, matched_terms

    @classmethod
    def _tokenize(cls, text: str) -> tuple[str, ...]:
        return tuple(
            token.lower()
            for token in cls._TOKEN_PATTERN.findall(text)
        )

    @staticmethod
    def _build_document_frequencies(
        documents: tuple[tuple[str, ...], ...],
    ) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in documents:
            frequencies.update(set(tokens))
        return frequencies

    @staticmethod
    def _term_overlap(
        query_tokens: tuple[str, ...],
        field_tokens: tuple[str, ...],
    ) -> float:
        if not query_tokens:
            return 0.0
        matches = set(query_tokens).intersection(field_tokens)
        return len(matches) / len(query_tokens)

    @classmethod
    def _matched_ingredients(
        cls,
        query: RecipeSearchQuery,
        document: RecipeDocument,
    ) -> tuple[str, ...]:
        ingredient_tokens = set(cls._tokenize(document.ingredient_text))
        return tuple(
            ingredient_id
            for ingredient_id in query.required_ingredient_ids
            if set(cls._tokenize(ingredient_id.replace("_", " "))).issubset(
                ingredient_tokens
            )
        )
