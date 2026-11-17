from __future__ import annotations

from types import SimpleNamespace

from cuisine_assistant.rag import (
    RagContextBuilder,
    RagGraphFact,
    RagKnowledgeGraphFactProvider,
)


class StubFactProvider(RagKnowledgeGraphFactProvider):
    """Returns deterministic graph facts and captures requested dishes."""

    def __init__(self) -> None:
        self.received_dish_ids = ()
        self.received_limit = 0

    def facts_for_dishes(self, dish_ids, maximum_facts):
        self.received_dish_ids = dish_ids
        self.received_limit = maximum_facts
        return (
            RagGraphFact(
                "ChickenPaprikash",
                "belongsToCuisine",
                "Hungarian cuisine",
                "urn:test#ChickenPaprikash",
            ),
            RagGraphFact(
                "ChickenPaprikash",
                "hasIngredient",
                "Paprika",
                "urn:test#ChickenPaprikash",
            ),
        )


def search_result():
    """Creates one application search result compatible with the builder."""
    document = SimpleNamespace(
        recipe_id="r001",
        dish_id="chicken_paprikash",
        title="Traditional Chicken Paprikash",
        searchable_text=(
            "Traditional Chicken Paprikash. Cook chicken with paprika, "
            "onion and sour cream."
        ),
        ingredient_text="chicken; paprika; onion; sour cream",
        cuisine_id="hungarian",
        source="manual",
        source_url="https://example.org/r001",
        license_name="CC BY 4.0",
    )
    hit = SimpleNamespace(document=document, score=0.93)
    ingredient = SimpleNamespace(ingredient_id="paprika")
    linked = SimpleNamespace(ingredient=ingredient)
    return SimpleNamespace(
        query_text="Find Hungarian chicken with paprika.",
        search=SimpleNamespace(hits=(hit,)),
        linking=SimpleNamespace(linked_ingredients=(linked,)),
        cuisine_preference=SimpleNamespace(cuisine_id="hungarian"),
    )


def test_builds_grounded_bounded_rag_context() -> None:
    """Verifies documents, RDF facts, sources, and query entities are combined."""
    provider = StubFactProvider()
    builder = RagContextBuilder(
        provider,
        maximum_documents=1,
        maximum_graph_facts=10,
    )

    context = builder.build(search_result())

    assert context.question == "Find Hungarian chicken with paprika."
    assert context.documents[0].recipe_id == "r001"
    assert context.documents[0].ingredients == (
        "chicken",
        "paprika",
        "onion",
        "sour cream",
    )
    assert context.sources[0].source_url == "https://example.org/r001"
    assert context.linked_ingredient_ids == ("paprika",)
    assert context.cuisine_id == "hungarian"
    assert provider.received_dish_ids == ("chicken_paprikash",)
    assert provider.received_limit == 10
    assert len(context.graph_facts) == 2


def test_renders_prompt_context_with_document_graph_and_source_ids() -> None:
    """Verifies deterministic labels can be referenced by an answer generator."""
    context = RagContextBuilder(StubFactProvider()).build(search_result())

    rendered = context.to_prompt_context()

    assert "[D1] Traditional Chicken Paprikash" in rendered
    assert "[G1] ChickenPaprikash belongsToCuisine" in rendered
    assert "[S1] Traditional Chicken Paprikash" in rendered
    assert "Linked ingredients: paprika" in rendered
    assert "Cuisine preference: hungarian" in rendered


def test_truncates_long_excerpt_at_word_boundary() -> None:
    """Verifies document context stays bounded for later model prompts."""
    result = search_result()
    long_document = SimpleNamespace(
        **{
            **vars(result.search.hits[0].document),
            "searchable_text": "ingredient " * 100,
        }
    )
    result.search = SimpleNamespace(
        hits=(SimpleNamespace(document=long_document, score=0.9),)
    )

    context = RagContextBuilder(
        StubFactProvider(),
        maximum_excerpt_characters=80,
    ).build(result)

    assert len(context.documents[0].excerpt) <= 83
    assert context.documents[0].excerpt.endswith("...")
