from __future__ import annotations

import os
from pathlib import Path

import pytest

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.knowledge_base import (
    JsonKnowledgeBaseManifestLoader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_project_links_known_ingredients() -> None:
    """Verifies the real curated bundle links known ingredients."""
    if os.environ.get("RUN_REAL_INTEGRATION") != "1":
        pytest.skip(
            "Set RUN_REAL_INTEGRATION=1 to run real file integration"
        )

    bundle = JsonKnowledgeBaseManifestLoader().load(
        PROJECT_ROOT
        / "knowledge_bases"
        / "curated"
        / "knowledge_base.json"
    )
    container = build_application_container(
        bundle=bundle,
        settings=ApplicationSettings(),
    )

    result = container.link_ingredients.execute(
        "Tomatoes, mozzarella and basil"
    )

    assert result.linked_ingredients
    assert result.unmatched_count == 0
    assert result.catalog_only_count == 0
    assert result.linked_count == len(result.linked_ingredients)
    assert container.knowledge_base.metadata.knowledge_base_id == "curated"