from __future__ import annotations

from pathlib import Path

from cuisine_assistant.bootstrap import ApplicationSettings


def test_settings_resolve_recommendation_data_paths(tmp_path: Path) -> None:
    """Verifies recommendation files are resolved from the project root."""
    settings = ApplicationSettings(
        project_root=tmp_path,
        ontology_base_iri="http://example.org/cuisine#",
    )

    assert settings.dishes_path == tmp_path / "data/processed/dishes.csv"
    assert settings.cuisine_aliases_path == (
        tmp_path / "data/processed/cuisine_aliases.csv"
    )


def test_settings_keep_top_three_recommendations_by_default(tmp_path: Path) -> None:
    """Verifies the default recommendation result limit remains three."""
    settings = ApplicationSettings(
        project_root=tmp_path,
        ontology_base_iri="http://example.org/cuisine#",
    )

    assert settings.maximum_recommendations == 3
