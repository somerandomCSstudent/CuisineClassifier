from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from cuisine_assistant.importing.models import (
    ImportSourceType,
    TextualImportOptions,
    TextualImportSource,
)
from cuisine_assistant.importing.ports import TextualRecordSource


class HuggingFaceDatasetServerClient:
    """Reads bounded JSON rows from the Hugging Face Dataset Viewer API."""

    BASE_URL = "https://datasets-server.huggingface.co/rows"
    MAXIMUM_PAGE_SIZE = 100

    def __init__(
        self,
        *,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        page_size: int = 100,
        maximum_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if not 1 <= page_size <= self.MAXIMUM_PAGE_SIZE:
            raise ValueError("page_size must be between 1 and 100")
        if maximum_retries < 0:
            raise ValueError("maximum_retries must not be negative")
        if retry_base_delay_seconds < 0:
            raise ValueError("retry_base_delay_seconds must not be negative")
        self._token = token or os.environ.get("HF_TOKEN")
        self._timeout_seconds = timeout_seconds
        self._page_size = page_size
        self._maximum_retries = maximum_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds

    def iter_rows(
        self,
        *,
        dataset: str,
        config: str,
        split: str,
        revision: str | None,
        maximum_rows: int,
    ) -> Iterable[Mapping[str, Any]]:
        """Yields row dictionaries without following or downloading image URLs."""
        offset = 0
        yielded = 0
        while yielded < maximum_rows:
            length = min(self._page_size, maximum_rows - yielded)
            document = self._request_page(
                dataset=dataset,
                config=config,
                split=split,
                revision=revision,
                offset=offset,
                length=length,
            )
            rows = document.get("rows")
            if not isinstance(rows, list):
                raise ValueError("Hugging Face rows response is missing a rows list")
            if not rows:
                return

            for item in rows:
                if yielded >= maximum_rows:
                    return
                if not isinstance(item, Mapping):
                    continue
                row = item.get("row")
                if isinstance(row, Mapping):
                    yield dict(row)
                    yielded += 1

            returned_count = len(rows)
            offset += returned_count
            total = document.get("num_rows_total")
            if isinstance(total, int) and offset >= total:
                return
            if returned_count < length:
                return

    def _request_page(
        self,
        *,
        dataset: str,
        config: str,
        split: str,
        revision: str | None,
        offset: int,
        length: int,
    ) -> dict[str, Any]:
        parameters = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
        if revision:
            parameters["revision"] = revision
        url = f"{self.BASE_URL}?{urlencode(parameters)}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "CuisineAssistant/0.1 text-only-import",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(url, headers=headers, method="GET")

        payload = self._open_with_retry(request)

        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Hugging Face Dataset Viewer returned invalid JSON"
            ) from error
        if not isinstance(document, dict):
            raise ValueError("Hugging Face rows response must be a JSON object")
        return document

    def _open_with_retry(self, request: Request) -> bytes:
        """Retries transient Dataset Viewer failures with exponential backoff."""
        retryable_status_codes = {429, 502, 503, 504}
        for attempt in range(self._maximum_retries + 1):
            try:
                with urlopen(
                    request,
                    timeout=self._timeout_seconds,
                ) as response:
                    content_type = response.headers.get_content_type()
                    if content_type != "application/json":
                        raise ValueError(
                            "Hugging Face Dataset Viewer returned non-JSON content"
                        )
                    return response.read()
            except HTTPError as error:
                should_retry = (
                    error.code in retryable_status_codes
                    and attempt < self._maximum_retries
                )
                if not should_retry:
                    raise RuntimeError(
                        "Hugging Face Dataset Viewer request failed with HTTP "
                        f"{error.code}"
                    ) from error
            except URLError as error:
                if attempt >= self._maximum_retries:
                    raise RuntimeError(
                        "Hugging Face Dataset Viewer request failed: "
                        f"{error.reason}"
                    ) from error

            delay = self._retry_base_delay_seconds * (2 ** attempt)
            if delay > 0:
                time.sleep(delay)

        raise RuntimeError(
            "Hugging Face Dataset Viewer request failed after retries"
        )


class HuggingFaceTextualRecordSource(TextualRecordSource):
    """Streams bounded Hugging Face rows for explicit offline text-only builds."""

    def __init__(self, client: HuggingFaceDatasetServerClient) -> None:
        self._client = client

    def read(
        self,
        source: TextualImportSource,
        options: TextualImportOptions,
    ) -> Iterable[Mapping[str, Any]]:
        """Returns Dataset Viewer rows without downloading referenced assets."""
        if source.source_type is not ImportSourceType.HUGGING_FACE:
            raise ValueError(
                "HuggingFaceTextualRecordSource requires a Hugging Face source"
            )
        return self._client.iter_rows(
            dataset=source.source_id,
            config=source.dataset_name,
            split=source.split,
            revision=source.revision,
            maximum_rows=options.maximum_records,
        )
