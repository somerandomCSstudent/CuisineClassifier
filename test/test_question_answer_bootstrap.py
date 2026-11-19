from __future__ import annotations

from pathlib import Path

import pytest

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.knowledge_base import JsonKnowledgeBaseManifestLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_settings_keep_bounded_rag_defaults() -> None:
    """Verifies default RAG limits remain small and deterministic."""
    settings = ApplicationSettings()

    assert settings.maximum_rag_documents == 3
    assert settings.maximum_rag_graph_facts == 20
    assert settings.maximum_rag_excerpt_characters == 700
    assert settings.maximum_answer_graph_facts == 4


def test_settings_reject_invalid_rag_limits() -> None:
    """Verifies invalid context bounds fail at bootstrap configuration."""
    with pytest.raises(ValueError, match="maximum_rag_documents"):
        ApplicationSettings(maximum_rag_documents=0)
    with pytest.raises(ValueError, match="maximum_rag_excerpt_characters"):
        ApplicationSettings(maximum_rag_excerpt_characters=49)


def test_curated_bundle_bootstrap_exposes_question_answering() -> None:
    """Verifies the selected bundle builds the complete grounded QA pipeline."""
    manifest = (
        PROJECT_ROOT
        / "knowledge_bases"
        / "curated"
        / "knowledge_base.json"
    )
    if not manifest.is_file():
        pytest.skip("Curated bundle is not available")

    bundle = JsonKnowledgeBaseManifestLoader().load(manifest)
    container = build_application_container(bundle, ApplicationSettings())
    result = container.ask_cuisine_question.execute(
        "What Hungarian dish uses chicken and paprika?"
    )

    assert container.ask_cuisine_question is not None
    assert result.question == (
        "What Hungarian dish uses chicken and paprika?"
    )
    assert result.context.question == result.question
    assert result.answer.text
    assert (
        result.has_grounded_answer
        or result.answer.insufficient_evidence
    )
