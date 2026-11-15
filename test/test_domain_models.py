import pytest

from cuisine_assistant.domain.models import (
    ExtractedIngredient,
    Ingredient,
    IngredientCategory,
    LinkedIngredient,
    MatchMethod,
    OntologyEntityReference,
)


def test_linked_ingredient_reports_existing_graph_entity() -> None:
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

    linked = LinkedIngredient(extracted, ingredient, reference)

    assert linked.is_linked_to_knowledge_graph is True


def test_unmatched_extraction_cannot_have_ingredient_link() -> None:
    extracted = ExtractedIngredient(
        surface_form="gochujang",
        normalized_text="gochujang",
        start_char=0,
        end_char=9,
        matched_ingredient_id=None,
        match_method=MatchMethod.UNMATCHED,
        match_score=0.0,
    )
    ingredient = Ingredient(
        ingredient_id="gochujang",
        canonical_name="gochujang",
        category=IngredientCategory.OTHER,
        ontology_id="Gochujang",
    )

    with pytest.raises(ValueError):
        LinkedIngredient(extracted, ingredient, None)


def test_rejects_mismatched_ontology_reference() -> None:
    extracted = ExtractedIngredient(
        surface_form="tomato",
        normalized_text="tomato",
        start_char=0,
        end_char=6,
        matched_ingredient_id="tomato",
        match_method=MatchMethod.EXACT,
        match_score=1.0,
    )
    ingredient = Ingredient(
        ingredient_id="tomato",
        canonical_name="tomato",
        category=IngredientCategory.VEGETABLE,
        ontology_id="Tomato",
    )
    wrong_reference = OntologyEntityReference(
        ontology_id="Potato",
        iri="http://example.org/cuisine#Potato",
        exists_in_knowledge_graph=True,
    )

    with pytest.raises(ValueError):
        LinkedIngredient(extracted, ingredient, wrong_reference)
