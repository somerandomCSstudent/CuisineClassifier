from __future__ import annotations

from cuisine_assistant.application import LinkIngredientsResult
from cuisine_assistant.cli import LinkIngredientsResultFormatter
from cuisine_assistant.domain.models import (
    ExtractedIngredient,
    Ingredient,
    IngredientCategory,
    LinkedIngredient,
    MatchMethod,
    OntologyEntityReference,
)


def create_linked_tomato() -> LinkedIngredient:
    """Creates a fully linked ingredient for formatter tests."""
    extracted = ExtractedIngredient(
        surface_form="fresh tomatoes",
        normalized_text="fresh tomatoes",
        start_char=0,
        end_char=14,
        matched_ingredient_id="tomato",
        match_method=MatchMethod.ALIAS,
        match_score=1.0,
        candidate_ids=("tomato",),
    )
    ingredient = Ingredient(
        ingredient_id="tomato",
        canonical_name="tomato",
        category=IngredientCategory.VEGETABLE,
        ontology_id="Tomato",
    )
    reference = OntologyEntityReference(
        ontology_id="Tomato",
        iri="http://example.org/cuisine#Tomato",
        rdf_types=("Vegetable",),
        exists_in_knowledge_graph=True,
    )
    return LinkedIngredient(extracted, ingredient, reference)


def create_unmatched_gochujang() -> LinkedIngredient:
    """Creates an unmatched ingredient for formatter tests."""
    extracted = ExtractedIngredient(
        surface_form="gochujang",
        normalized_text="gochujang",
        start_char=16,
        end_char=25,
        matched_ingredient_id=None,
        match_method=MatchMethod.UNMATCHED,
        match_score=0.0,
    )
    return LinkedIngredient(extracted, None, None)


def test_formats_linked_and_unmatched_ingredients() -> None:
    formatter = LinkIngredientsResultFormatter()
    result = LinkIngredientsResult(
        linked_ingredients=(
            create_linked_tomato(),
            create_unmatched_gochujang(),
        )
    )

    output = formatter.format(result)

    assert "Canonical ingredient: tomato" in output
    assert "OWL entity: Tomato" in output
    assert "RDF types: Vegetable" in output
    assert "Knowledge graph status: linked" in output
    assert "Status: unmatched" in output
    assert "Linked to knowledge graph: 1" in output
    assert "Unmatched: 1" in output


def test_formats_empty_result() -> None:
    formatter = LinkIngredientsResultFormatter()

    assert formatter.format(LinkIngredientsResult(())) == (
        "No ingredients were detected."
    )
