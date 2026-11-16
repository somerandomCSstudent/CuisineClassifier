from __future__ import annotations

from pathlib import Path

from scripts.build_world_cuisines_bundle import parse_arguments


def test_script_module_has_valid_default_record_limit(monkeypatch) -> None:
    """Verifies the explicit build command defaults to the approved 500 records."""
    monkeypatch.setattr(
        "sys.argv",
        ["build_world_cuisines_bundle.py"],
    )

    arguments = parse_arguments()

    assert arguments.maximum_records == 500
    assert arguments.curated_knowledge_base_id == "curated"
    assert arguments.ontology_base_iri.endswith("#")


def test_script_is_present_in_expected_location() -> None:
    """Verifies the deliverable provides the documented executable script."""
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "scripts/build_world_cuisines_bundle.py").is_file()
