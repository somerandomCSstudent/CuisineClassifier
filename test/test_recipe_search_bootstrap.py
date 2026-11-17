from __future__ import annotations

from pathlib import Path

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.knowledge_base import JsonKnowledgeBaseManifestLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_settings_keep_five_recipe_search_results_by_default() -> None:
    """Verifies the default top-k recipe retrieval limit remains five."""
    settings = ApplicationSettings()

    assert settings.maximum_recipe_search_results == 5
    assert settings.bm25_k1 == 1.5
    assert settings.bm25_b == 0.75


def test_curated_bundle_bootstrap_exposes_recipe_search() -> None:
    """Verifies the selected bundle constructs the BM25 recipe search use case."""
    manifest = (
        PROJECT_ROOT
        / "knowledge_bases"
        / "curated"
        / "knowledge_base.json"
    )
    if not manifest.is_file():
        return

    bundle = JsonKnowledgeBaseManifestLoader().load(manifest)
    container = build_application_container(bundle, ApplicationSettings())
    result = container.search_recipes.execute(
        "Hungarian chicken with paprika"
    )

    assert container.search_recipes is not None
    assert result.search.query.maximum_results == 5
    assert result.search.query.required_ingredient_ids == (
        "chicken",
        "paprika",
    )
