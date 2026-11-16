from __future__ import annotations

import os
from pathlib import Path

import pytest

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_project_recommends_chicken_paprikash() -> None:
    """Verifies real CSV and RDF files produce the expected Hungarian dish."""
    if os.environ.get("RUN_REAL_INTEGRATION") != "1":
        pytest.skip("Set RUN_REAL_INTEGRATION=1 to run real file integration")

    base_iri = os.environ.get("CUISINE_ONTOLOGY_BASE_IRI")
    if not base_iri:
        pytest.fail("CUISINE_ONTOLOGY_BASE_IRI must be set")

    container = build_application_container(
        ApplicationSettings(
            project_root=PROJECT_ROOT,
            ontology_base_iri=base_iri,
        )
    )
    result = container.recommend_dishes.execute(
        "I have chicken, paprika, onion and sour cream and I would like Hungarian food."
    )

    assert result.recommendation.best_recommendation is not None
    assert (
        result.recommendation.best_recommendation.dish.dish_id
        == "chicken_paprikash"
    )
    assert result.recommendation.best_recommendation.score == pytest.approx(1.0)
