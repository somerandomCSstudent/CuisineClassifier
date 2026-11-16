from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cuisine_assistant.knowledge_base.models import (
    KnowledgeBaseBundle,
    KnowledgeBaseMetadata,
    KnowledgeBasePaths,
)


class JsonKnowledgeBaseManifestLoader:
    """Loads and validates one knowledge base bundle from its JSON manifest."""

    _REQUIRED_FIELDS = {
        "knowledge_base_id",
        "name",
        "version",
        "source",
        "language",
        "ontology_base_iri",
    }
    _COUNT_FIELDS = {
        "dish_count",
        "ingredient_count",
        "cuisine_count",
        "triple_count",
    }
    _FORBIDDEN_IMAGE_FIELDS = {
        "image",
        "images",
        "image_url",
        "image_urls",
        "image_file",
        "image_files",
        "image_embedding",
        "image_embeddings",
    }

    def load(self, manifest_path: str | Path) -> KnowledgeBaseBundle:
        """Creates a bundle whose root is the manifest parent directory."""
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Knowledge base manifest does not exist: {path}"
            )

        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid knowledge base JSON manifest: {path}"
            ) from error

        if not isinstance(document, dict):
            raise ValueError("Knowledge base manifest root must be a JSON object")

        self._reject_image_fields(document)
        self._validate_required_fields(document)
        metadata = self._map_metadata(document)
        paths = self._map_paths(path.parent, document.get("files"))

        if paths.manifest.resolve() != path.resolve():
            raise ValueError(
                "Manifest file mapping must reference the loaded manifest itself"
            )
        return KnowledgeBaseBundle(metadata=metadata, paths=paths)

    def _map_metadata(self, document: dict[str, Any]) -> KnowledgeBaseMetadata:
        counts = document.get("counts", {})
        if counts is None:
            counts = {}
        if not isinstance(counts, dict):
            raise ValueError("counts must be a JSON object")

        return KnowledgeBaseMetadata(
            knowledge_base_id=self._required_text(
                document,
                "knowledge_base_id",
            ),
            name=self._required_text(document, "name"),
            version=self._required_text(document, "version"),
            source=self._required_text(document, "source"),
            language=self._required_text(document, "language"),
            ontology_base_iri=self._required_text(
                document,
                "ontology_base_iri",
            ),
            dish_count=self._non_negative_integer(counts, "dish_count"),
            ingredient_count=self._non_negative_integer(
                counts,
                "ingredient_count",
            ),
            cuisine_count=self._non_negative_integer(
                counts,
                "cuisine_count",
            ),
            triple_count=self._non_negative_integer(
                counts,
                "triple_count",
            ),
        )

    def _map_paths(
        self,
        root: Path,
        file_mapping: Any,
    ) -> KnowledgeBasePaths:
        defaults = KnowledgeBasePaths.from_root(root)
        if file_mapping is None:
            return defaults
        if not isinstance(file_mapping, dict):
            raise ValueError("files must be a JSON object")

        allowed = {
            "ontology",
            "ingredients",
            "ingredient_aliases",
            "cuisines",
            "cuisine_aliases",
            "cuisine_ingredients",
            "dishes",
            "dish_ingredients",
            "recipes",
            "search_index",
            "manifest",
        }
        unknown = set(file_mapping) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported knowledge base file keys: {names}")

        values = {
            key: self._safe_bundle_path(
                root,
                file_mapping.get(key),
                getattr(defaults, key),
            )
            for key in allowed
        }
        return KnowledgeBasePaths(root=root, **values)

    def _safe_bundle_path(
        self,
        root: Path,
        value: Any,
        default: Path,
    ) -> Path:
        if value is None:
            return default
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Knowledge base file paths must be non-empty strings")

        relative = Path(value)
        if relative.is_absolute():
            raise ValueError("Knowledge base file paths must be relative")
        resolved = (root / relative).resolve()
        bundle_root = root.resolve()
        if resolved != bundle_root and bundle_root not in resolved.parents:
            raise ValueError("Knowledge base file path escapes the bundle root")
        return root / relative

    def _validate_required_fields(self, document: dict[str, Any]) -> None:
        missing = self._REQUIRED_FIELDS - set(document)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"Knowledge base manifest is missing required fields: {names}"
            )

    def _reject_image_fields(self, value: Any, path: str = "root") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = key.lower().replace("-", "_")
                if normalized_key in self._FORBIDDEN_IMAGE_FIELDS:
                    raise ValueError(
                        f"Image-related manifest field is not allowed: {path}.{key}"
                    )
                self._reject_image_fields(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._reject_image_fields(child, f"{path}[{index}]")

    @staticmethod
    def _required_text(document: dict[str, Any], field_name: str) -> str:
        value = document.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()

    @classmethod
    def _non_negative_integer(
        cls,
        counts: dict[str, Any],
        field_name: str,
    ) -> int:
        value = counts.get(field_name, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
        if value < 0:
            raise ValueError(f"{field_name} must not be negative")
        return value

    
