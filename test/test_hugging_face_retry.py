from __future__ import annotations

import io
import json
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from cuisine_assistant.importing.hugging_face_source import (
    HuggingFaceDatasetServerClient,
)


class FakeResponse:
    """Provides a successful JSON response after transient test failures."""

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


def successful_document() -> dict:
    """Returns one valid Dataset Viewer page."""
    return {
        "rows": [{"row": {"id": "1", "name": "Dish"}}],
        "num_rows_total": 1,
        "num_rows_per_page": 1,
        "partial": False,
    }


def test_retries_http_502_then_returns_rows(monkeypatch) -> None:
    """Verifies an intermittent gateway failure does not abort the build."""
    calls = 0
    delays: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise HTTPError(request.full_url, 502, "Bad Gateway", None, io.BytesIO())
        return FakeResponse(successful_document())

    monkeypatch.setattr(
        "cuisine_assistant.importing.hugging_face_source.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        "cuisine_assistant.importing.hugging_face_source.time.sleep",
        delays.append,
    )
    client = HuggingFaceDatasetServerClient(
        maximum_retries=3,
        retry_base_delay_seconds=1.0,
        page_size=10,
    )

    rows = tuple(
        client.iter_rows(
            dataset="owner/data",
            config="default",
            split="main",
            revision=None,
            maximum_rows=1,
        )
    )

    assert rows == ({"id": "1", "name": "Dish"},)
    assert calls == 3
    assert delays == [1.0, 2.0]


def test_does_not_retry_non_transient_http_error(monkeypatch) -> None:
    """Verifies authorization and request errors fail immediately."""
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(request.full_url, 404, "Not Found", None, io.BytesIO())

    monkeypatch.setattr(
        "cuisine_assistant.importing.hugging_face_source.urlopen",
        fake_urlopen,
    )
    client = HuggingFaceDatasetServerClient(
        maximum_retries=3,
        retry_base_delay_seconds=0,
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        tuple(
            client.iter_rows(
                dataset="owner/data",
                config="default",
                split="main",
                revision=None,
                maximum_rows=1,
            )
        )
    assert calls == 1


def test_retries_url_error_until_limit(monkeypatch) -> None:
    """Verifies transient connection errors respect the configured retry limit."""
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise URLError("temporary DNS failure")

    monkeypatch.setattr(
        "cuisine_assistant.importing.hugging_face_source.urlopen",
        fake_urlopen,
    )
    client = HuggingFaceDatasetServerClient(
        maximum_retries=2,
        retry_base_delay_seconds=0,
    )

    with pytest.raises(RuntimeError, match="temporary DNS failure"):
        tuple(
            client.iter_rows(
                dataset="owner/data",
                config="default",
                split="main",
                revision=None,
                maximum_rows=1,
            )
        )
    assert calls == 3


def test_validates_retry_configuration() -> None:
    """Verifies invalid retry settings fail before network access."""
    with pytest.raises(ValueError, match="maximum_retries"):
        HuggingFaceDatasetServerClient(maximum_retries=-1)
    with pytest.raises(ValueError, match="retry_base_delay_seconds"):
        HuggingFaceDatasetServerClient(retry_base_delay_seconds=-1)
