from __future__ import annotations

from collections.abc import Iterable

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher

from cuisine_assistant.domain.models import ExtractedIngredient
from cuisine_assistant.domain.ports import EntitySearch, IngredientExtractor, TextNormalizer


class SpacyPhraseIngredientExtractor(IngredientExtractor):
    """Uses spaCy tokenization and PhraseMatcher with the search index terms."""

    def __init__(
        self,
        normalizer: TextNormalizer,
        search: EntitySearch,
        nlp: Language | None = None,
    ) -> None:
        self._normalizer = normalizer
        self._search = search
        self._nlp = nlp or spacy.blank("en")
        self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        self._add_patterns(search.known_terms())

    def _add_patterns(self, terms: Iterable[str]) -> None:
        patterns = [self._nlp.make_doc(term) for term in sorted(set(terms))]
        if patterns:
            self._matcher.add("INGREDIENT", patterns)

    def extract(self, recipe_text: str) -> list[ExtractedIngredient]:
        doc = self._nlp(recipe_text)
        spans = self._select_non_overlapping_spans(doc)
        results: list[ExtractedIngredient] = []
        for span in spans:
            normalized = self._normalizer.normalize(span.text)
            candidates = self._search.find(normalized)
            if not candidates:
                continue
            best = candidates[0]
            results.append(
                ExtractedIngredient(
                    surface_form=span.text,
                    normalized_text=normalized,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    matched_ingredient_id=best.entity_id,
                    match_method=best.method,
                    match_score=best.score,
                    candidate_ids=[candidate.entity_id for candidate in candidates],
                )
            )
        return sorted(results, key=lambda item: item.start_char)

    def _select_non_overlapping_spans(self, doc):
        spans = [doc[start:end] for _, start, end in self._matcher(doc)]
        spans.sort(key=lambda span: (-(span.end_char - span.start_char), span.start_char))
        selected = []
        occupied: list[tuple[int, int]] = []
        for span in spans:
            interval = (span.start_char, span.end_char)
            if any(interval[0] < end and interval[1] > start for start, end in occupied):
                continue
            selected.append(span)
            occupied.append(interval)
        return selected
