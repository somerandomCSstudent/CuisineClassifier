from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from rdflib import Graph


@dataclass(frozen=True)
class BundleSource:
    """Defines the existing project files used to build the curated bundle."""

    project_root: Path

    @property
    def ontology(self) -> Path:
        """Returns the existing curated RDF/XML ontology path."""
        return self.project_root / "ontology" / "cuisine.rdf"

    @property
    def processed_data(self) -> Path:
        """Returns the existing processed CSV directory."""
        return self.project_root / "data" / "processed"

    def search_index(self) -> Path:
        """Returns the first supported existing lexical search index path."""
        candidates = (
            self.project_root / "data" / "processed" / "search_index.csv",
            self.project_root / "data" / "indexes" / "search_index.csv",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])


class CuratedKnowledgeBasePackager:
    """Builds and atomically installs a complete curated knowledge base bundle."""

    REQUIRED_CSV_FILES = (
        "ingredients.csv",
        "ingredient_aliases.csv",
        "cuisines.csv",
        "cuisine_aliases.csv",
        "cuisine_ingredients.csv",
        "dishes.csv",
        "dish_ingredients.csv",
        "recipes.csv",
    )

    def __init__(
        self,
        source: BundleSource,
        destination: Path,
        ontology_base_iri: str,
    ) -> None:
        self._source = source
        self._destination = destination
        self._ontology_base_iri = ontology_base_iri.strip()
        if not self._ontology_base_iri:
            raise ValueError("ontology_base_iri must not be empty")

    def package(self) -> Path:
        """Copies all required files, writes metadata, validates, and installs the bundle."""
        source_files = self._resolve_source_files()
        self._validate_sources(source_files)
        parent = self._destination.parent
        parent.mkdir(parents=True, exist_ok=True)

        temporary_root = Path(
            tempfile.mkdtemp(prefix="curated-build-", dir=parent)
        )
        try:
            self._copy_bundle_files(source_files, temporary_root)
            manifest = self._create_manifest(temporary_root)
            (temporary_root / "knowledge_base.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._validate_built_bundle(temporary_root)
            self._install_atomically(temporary_root)
        except Exception:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise
        return self._destination

    def _resolve_source_files(self) -> dict[str, Path]:
        files = {
            "cuisine.rdf": self._source.ontology,
            "search_index.csv": self._source.search_index(),
        }
        files.update(
            {
                name: self._source.processed_data / name
                for name in self.REQUIRED_CSV_FILES
            }
        )
        return files

    @staticmethod
    def _validate_sources(source_files: dict[str, Path]) -> None:
        missing = tuple(path for path in source_files.values() if not path.is_file())
        if missing:
            details = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Cannot build the curated bundle because required files are missing:\n"
                f"{details}"
            )

    @staticmethod
    def _copy_bundle_files(source_files: dict[str, Path], target: Path) -> None:
        for target_name, source_path in source_files.items():
            shutil.copy2(source_path, target / target_name)

    def _create_manifest(self, bundle_root: Path) -> dict:
        graph = Graph()
        graph.parse(bundle_root / "cuisine.rdf", format="xml")
        return {
            "knowledge_base_id": "curated",
            "name": "Curated Demo Knowledge Base",
            "version": "1.0",
            "source": "manual",
            "language": "en",
            "ontology_base_iri": self._ontology_base_iri,
            "counts": {
                "dish_count": self._csv_record_count(bundle_root / "dishes.csv"),
                "ingredient_count": self._csv_record_count(
                    bundle_root / "ingredients.csv"
                ),
                "cuisine_count": self._csv_record_count(bundle_root / "cuisines.csv"),
                "triple_count": len(graph),
            },
            "files": {
                "ontology": "cuisine.rdf",
                "ingredients": "ingredients.csv",
                "ingredient_aliases": "ingredient_aliases.csv",
                "cuisines": "cuisines.csv",
                "cuisine_aliases": "cuisine_aliases.csv",
                "cuisine_ingredients": "cuisine_ingredients.csv",
                "dishes": "dishes.csv",
                "dish_ingredients": "dish_ingredients.csv",
                "recipes": "recipes.csv",
                "search_index": "search_index.csv",
                "manifest": "knowledge_base.json",
            },
        }

    @staticmethod
    def _csv_record_count(path: Path) -> int:
        with path.open(encoding="utf-8-sig", newline="") as source:
            return sum(1 for _ in csv.DictReader(source))

    @classmethod
    def _validate_built_bundle(cls, bundle_root: Path) -> None:
        required = (
            "knowledge_base.json",
            "cuisine.rdf",
            "search_index.csv",
            *cls.REQUIRED_CSV_FILES,
        )
        missing = tuple(name for name in required if not (bundle_root / name).is_file())
        if missing:
            raise ValueError(
                "Built curated bundle is incomplete: " + ", ".join(missing)
            )

        forbidden_extensions = {
            ".bmp",
            ".gif",
            ".jpeg",
            ".jpg",
            ".png",
            ".svg",
            ".webp",
        }
        image_files = tuple(
            path
            for path in bundle_root.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden_extensions
        )
        if image_files:
            raise ValueError("Image files are not allowed in a knowledge base bundle")

    def _install_atomically(self, temporary_root: Path) -> None:
        """Installs the bundle atomically and treats backup cleanup as best effort."""
        backup = self._destination.with_name(
            f"{self._destination.name}.backup-{uuid4().hex}"
        )

        if self._destination.exists():
            self._destination.replace(backup)

        try:
            temporary_root.replace(self._destination)
        except Exception:
            if backup.exists() and not self._destination.exists():
                backup.replace(self._destination)
            raise

        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line options for curated bundle packaging."""
    parser = argparse.ArgumentParser(
        description="Build the complete curated knowledge base bundle."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--ontology-base-iri",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    """Builds the curated bundle from the current project data."""
    arguments = parse_arguments()
    project_root = arguments.project_root.resolve()
    destination = project_root / "knowledge_bases" / "curated"
    packaged_path = CuratedKnowledgeBasePackager(
        source=BundleSource(project_root),
        destination=destination,
        ontology_base_iri=arguments.ontology_base_iri,
    ).package()
    print(f"Curated knowledge base bundle created: {packaged_path}")


if __name__ == "__main__":
    main()
