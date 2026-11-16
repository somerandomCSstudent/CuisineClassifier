from __future__ import annotations

import argparse
from pathlib import Path

from cuisine_assistant.application import SelectKnowledgeBaseUseCase
from cuisine_assistant.bootstrap import ApplicationSettings, build_application_container
from cuisine_assistant.importing import (
    DefaultTextualKnowledgeImporter,
    HuggingFaceDatasetServerClient,
    HuggingFaceTextualRecordSource,
    ImportSourceType,
    ImportedKnowledgeBaseBundleBuilder,
    TextualImportOptions,
    TextualImportSource,
    WorldCuisinesBundleBuilder,
    WorldCuisinesRecordMapper,
)
from cuisine_assistant.knowledge_base import (
    FileKnowledgeBaseRegistry,
    FileKnowledgeBaseValidator,
    JsonKnowledgeBaseManifestLoader,
)


class WorldCuisinesBuildCommand:
    """Builds a local image-free World Cuisines knowledge base bundle."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def run(
        self,
        *,
        maximum_records: int,
        ontology_base_iri: str,
        curated_knowledge_base_id: str,
        version: str,
        timeout_seconds: float,
    ) -> Path:
        """Imports bounded text rows and atomically installs the generated bundle."""
        curated_container = self._build_curated_container(
            curated_knowledge_base_id
        )
        record_source = HuggingFaceTextualRecordSource(
            HuggingFaceDatasetServerClient(
                timeout_seconds=timeout_seconds,
                page_size=100,
            )
        )
        mapper = WorldCuisinesRecordMapper(
            ingredient_extractor=curated_container.link_ingredients._extractor
        )
        importer = DefaultTextualKnowledgeImporter(
            record_source=record_source,
            record_mapper=mapper,
        )
        source = TextualImportSource(
            source_type=ImportSourceType.HUGGING_FACE,
            source_id="worldcuisines/food-kb",
            dataset_name="default",
            split="main",
            revision=None,
            language="en",
            license_name="CC BY-SA 4.0",
        )
        options = TextualImportOptions(
            maximum_records=maximum_records,
            extract_ingredients_from_description=True,
            include_descriptions=True,
            include_aliases=True,
            deduplicate_records=True,
        )
        batch = importer.import_records(source, options)
        if not batch.records:
            raise RuntimeError("The World Cuisines import returned no usable records")

        destination = (
            self._project_root / "knowledge_bases" / "world_cuisines"
        )
        built_path = WorldCuisinesBundleBuilder(
            ImportedKnowledgeBaseBundleBuilder()
        ).build(
            batch=batch,
            destination=destination,
            ontology_base_iri=ontology_base_iri,
            version=version,
        )
        self._validate_generated_bundle(built_path)
        self._print_summary(batch, built_path)
        return built_path

    def _build_curated_container(self, knowledge_base_id: str):
        validator = FileKnowledgeBaseValidator()
        registry = FileKnowledgeBaseRegistry(
            self._project_root / "knowledge_bases",
            JsonKnowledgeBaseManifestLoader(),
            validator,
            strict=False,
        )
        selected = SelectKnowledgeBaseUseCase(
            registry,
            validator,
        ).execute(knowledge_base_id)
        return build_application_container(
            bundle=selected.bundle,
            settings=ApplicationSettings(),
        )

    @staticmethod
    def _validate_generated_bundle(bundle_root: Path) -> None:
        loader = JsonKnowledgeBaseManifestLoader()
        validator = FileKnowledgeBaseValidator()
        bundle = loader.load(bundle_root / "knowledge_base.json")
        result = validator.validate(bundle)
        if not result.is_valid:
            missing = ", ".join(str(path) for path in result.missing_files)
            raise RuntimeError(f"Generated bundle is incomplete: {missing}")

    @staticmethod
    def _print_summary(batch, built_path: Path) -> None:
        print(f"World Cuisines bundle created: {built_path}")
        print(f"Imported records: {len(batch.records)}")
        print(f"Skipped records: {batch.skipped_record_count}")
        print(f"Duplicate records: {batch.duplicate_record_count}")
        print("Images downloaded: 0")
        print("Images processed: 0")


def parse_arguments() -> argparse.Namespace:
    """Parses options for an explicit offline World Cuisines build."""
    parser = argparse.ArgumentParser(
        description=(
            "Build an image-free World Cuisines knowledge base bundle from "
            "Hugging Face text and structured fields."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--maximum-records",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--ontology-base-iri",
        default="http://example.org/world-cuisines#",
    )
    parser.add_argument(
        "--curated-knowledge-base-id",
        default="curated",
    )
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    """Runs the bounded Hugging Face import and complete bundle build."""
    arguments = parse_arguments()
    if arguments.maximum_records < 1:
        raise ValueError("maximum_records must be at least 1")
    command = WorldCuisinesBuildCommand(arguments.project_root.resolve())
    command.run(
        maximum_records=arguments.maximum_records,
        ontology_base_iri=arguments.ontology_base_iri,
        curated_knowledge_base_id=arguments.curated_knowledge_base_id,
        version=arguments.version,
        timeout_seconds=arguments.timeout_seconds,
    )


if __name__ == "__main__":
    main()
