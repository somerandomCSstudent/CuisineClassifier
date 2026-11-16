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
def test_real_project_recommends_chicken_paprikash() -> None:
    """Verifies the curated bundle recommends Chicken Paprikash."""
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

    result = container.recommend_dishes.execute(
        "I have chicken, paprika, onion and sour cream "
        "and I would like Hungarian food."
    )

    best = result.recommendation.best_recommendation

    assert best is not None
    assert best.dish.dish_id == "chicken_paprikash"
    assert best.score == pytest.approx(1.0)
    assert container.knowledge_base.metadata.knowledge_base_id == "curated"