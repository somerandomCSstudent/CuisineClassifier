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
def test_real_project_links_known_ingredients() -> None:
    """Verifies the real CSV and OWL files when integration testing is enabled."""
    if os.environ.get("RUN_REAL_INTEGRATION") != "1":
        pytest.skip("Set RUN_REAL_INTEGRATION=1 to run real file integration")

    ontology_base_iri = os.environ.get("CUISINE_ONTOLOGY_BASE_IRI")
    if not ontology_base_iri:
        pytest.fail("CUISINE_ONTOLOGY_BASE_IRI must be set")

    container = build_application_container(
        ApplicationSettings(
        project_root=PROJECT_ROOT,
        ontology_base_iri=ontology_base_iri,
        ontology_relative_path=Path("ontology/cuisine.rdf"),
        )
    )
    result = container.link_ingredients.execute(
        "Tomatoes, mozzarella and basil"
    )

    assert result.linked_ingredients
    assert result.unmatched_count == 0
    assert result.catalog_only_count == 0
    assert result.linked_count == len(result.linked_ingredients)
