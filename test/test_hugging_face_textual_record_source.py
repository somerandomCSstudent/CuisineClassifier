from __future__ import annotations

import json
from email.message import Message
from urllib.parse import parse_qs, urlparse

import pytest

from cuisine_assistant.importing import (
    HuggingFaceDatasetServerClient,
    HuggingFaceTextualRecordSource,
    ImportSourceType,
    TextualImportOptions,
    TextualImportSource,
)


class FakeResponse:
    """Provides a context-managed JSON HTTP response for adapter tests."""

    def __init__(self, document: dict) -> None:
        self._payload = json.dumps(document).encode("utf-8")
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def test_paginates_rows_and_honors_maximum_record_limit(monkeypatch) -> None:
    """Verifies bounded Dataset Viewer pagination without asset retrieval."""
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        query = parse_qs(urlparse(request.full_url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        rows = [
            {"row_idx": index, "row": {"id": str(index), "name": f"Dish {index}"}}
            for index in range(offset, offset + length)
        ]
        return FakeResponse(
            {
                "rows": rows,
                "num_rows_total": 250,
                "num_rows_per_page": length,
                "partial": False,
            }
        )

    monkeypatch.setattr(
        "cuisine_assistant.importing.hugging_face_source.urlopen",
        fake_urlopen,
    )
    client = HuggingFaceDatasetServerClient(page_size=100)

    rows = tuple(
        client.iter_rows(
            dataset="worldcuisines/food-kb",
            config="food-kb",
            split="train",
            revision="main",
            maximum_rows=150,
        )
    )

    assert len(rows) == 150
    assert len(requested_urls) == 2
    assert "length=100" in requested_urls[0]
    assert "offset=100" in requested_urls[1]
    assert "length=50" in requested_urls[1]


def test_returns_only_nested_row_payload(monkeypatch) -> None:
    """Verifies Dataset Viewer metadata is not exposed as an imported record."""

    def fake_urlopen(request, timeout):
        return FakeResponse(
            {
                "features": [{"feature_idx": 0, "name": "name"}],
                "rows": [
                    {
                        "row_idx": 7,
                        "truncated_cells": [],
                        "row": {"id": "rawon", "name": "Rawon"},
                    }
                ],
                "num_rows_total": 1,
                "num_rows_per_page": 1,
                "partial": False,
            }
        )

    monkeypatch.setattr(
        "cuisine_assistant.importing.hugging_face_source.urlopen",
        fake_urlopen,
    )
    client = HuggingFaceDatasetServerClient(page_size=10)

    rows = tuple(
        client.iter_rows(
            dataset="worldcuisines/food-kb",
            config="food-kb",
            split="train",
            revision=None,
            maximum_rows=10,
        )
    )

    assert rows == ({"id": "rawon", "name": "Rawon"},)


def test_adds_optional_bearer_token(monkeypatch) -> None:
    """Verifies gated datasets can use an explicit Hugging Face token."""
    authorization_headers: list[str | None] = []

    def fake_urlopen(request, timeout):
        authorization_headers.append(request.get_header("Authorization"))
        return FakeResponse(
            {
                "rows": [],
                "num_rows_total": 0,
                "num_rows_per_page": 0,
                "partial": False,
            }
        )

    monkeypatch.setattr(
        "cuisine_assistant.importing.hugging_face_source.urlopen",
        fake_urlopen,
    )
    client = HuggingFaceDatasetServerClient(token="hf_test", page_size=10)
    tuple(
        client.iter_rows(
            dataset="owner/dataset",
            config="default",
            split="train",
            revision=None,
            maximum_rows=1,
        )
    )

    assert authorization_headers == ["Bearer hf_test"]


def test_source_rejects_non_hugging_face_source() -> None:
    """Verifies the adapter cannot accidentally handle local source types."""
    adapter = HuggingFaceTextualRecordSource(
        HuggingFaceDatasetServerClient(page_size=10)
    )
    source = TextualImportSource(
        ImportSourceType.LOCAL_JSON,
        "local.json",
        "local",
    )

    with pytest.raises(ValueError, match="requires a Hugging Face source"):
        tuple(adapter.read(source, TextualImportOptions(maximum_records=1)))


def test_client_validates_timeout_and_page_size() -> None:
    """Verifies Dataset Viewer limits fail before any network request."""
    with pytest.raises(ValueError, match="timeout_seconds"):
        HuggingFaceDatasetServerClient(timeout_seconds=0)
    with pytest.raises(ValueError, match="page_size"):
        HuggingFaceDatasetServerClient(page_size=101)
