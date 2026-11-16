from __future__ import annotations

from pathlib import Path

from cuisine_assistant.knowledge_base.manifest_loader import (
    JsonKnowledgeBaseManifestLoader,
)
from cuisine_assistant.knowledge_base.models import KnowledgeBaseBundle
from cuisine_assistant.knowledge_base.ports import (
    KnowledgeBaseRegistry,
    KnowledgeBaseValidator,
)


class FileKnowledgeBaseRegistry(KnowledgeBaseRegistry):
    """Discovers, validates, and resolves local knowledge base bundles."""

    def __init__(
        self,
        knowledge_bases_root: str | Path,
        manifest_loader: JsonKnowledgeBaseManifestLoader,
        validator: KnowledgeBaseValidator,
        *,
        strict: bool = False,
    ) -> None:
        self._root = Path(knowledge_bases_root)
        self._manifest_loader = manifest_loader
        self._validator = validator
        self._strict = strict
        self._bundles = self._discover_bundles()

    def list_available(self) -> tuple[KnowledgeBaseBundle, ...]:
        """Returns valid bundles sorted by display name and stable identifier."""
        return self._bundles

    def get_by_id(self, knowledge_base_id: str) -> KnowledgeBaseBundle | None:
        """Returns one valid bundle by its stable identifier."""
        normalized_id = knowledge_base_id.strip()
        return next(
            (
                bundle
                for bundle in self._bundles
                if bundle.metadata.knowledge_base_id == normalized_id
            ),
            None,
        )

    def refresh(self) -> tuple[KnowledgeBaseBundle, ...]:
        """Rescans the registry root and returns the updated valid bundles."""
        self._bundles = self._discover_bundles()
        return self._bundles

    def _discover_bundles(self) -> tuple[KnowledgeBaseBundle, ...]:
        if not self._root.exists():
            return ()
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"Knowledge base registry root is not a directory: {self._root}"
            )

        bundles: list[KnowledgeBaseBundle] = []
        identifiers: dict[str, Path] = {}
        errors: list[str] = []

        for manifest_path in sorted(
            self._root.glob("*/knowledge_base.json"),
            key=lambda path: path.parent.name.lower(),
        ):
            try:
                bundle = self._manifest_loader.load(manifest_path)
                validation = self._validator.validate(bundle)
                if not validation.is_valid:
                    missing = ", ".join(
                        str(path.relative_to(bundle.paths.root))
                        for path in validation.missing_files
                    )
                    raise ValueError(
                        f"Incomplete knowledge base bundle: {missing}"
                    )

                knowledge_base_id = bundle.metadata.knowledge_base_id
                previous_manifest = identifiers.get(knowledge_base_id)
                if previous_manifest is not None:
                    raise ValueError(
                        "Duplicate knowledge_base_id "
                        f"'{knowledge_base_id}' in {previous_manifest} and "
                        f"{manifest_path}"
                    )
                identifiers[knowledge_base_id] = manifest_path
                bundles.append(bundle)
            except (OSError, ValueError) as error:
                message = f"{manifest_path}: {error}"
                if self._strict:
                    raise ValueError(
                        f"Failed to register knowledge base: {message}"
                    ) from error
                errors.append(message)

        bundles.sort(
            key=lambda bundle: (
                bundle.metadata.name.lower(),
                bundle.metadata.knowledge_base_id,
            )
        )
        return tuple(bundles)
