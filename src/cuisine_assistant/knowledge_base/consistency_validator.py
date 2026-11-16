from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, URIRef

from cuisine_assistant.knowledge_base.models import KnowledgeBaseBundle


@dataclass(frozen=True)
class ConsistencyIssue:
    """Describes one cross-file knowledge base consistency problem."""

    code: str
    message: str
    file_name: str
    row_number: int | None = None


@dataclass(frozen=True)
class KnowledgeBaseConsistencyReport:
    """Contains cross-file validation issues for one complete knowledge base bundle."""

    knowledge_base_id: str
    issues: tuple[ConsistencyIssue, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        """Returns whether no consistency issue was detected."""
        return not self.issues


class KnowledgeBaseConsistencyValidator:
    """Validates CSV references, manifest counts, and RDF entity consistency."""

    def validate(
        self,
        bundle: KnowledgeBaseBundle,
    ) -> KnowledgeBaseConsistencyReport:
        """Runs deterministic cross-file checks without modifying the bundle."""
        issues: list[ConsistencyIssue] = []
        paths = bundle.paths

        ingredients = self._read_index(
            paths.ingredients,
            "ingredient_id",
            issues,
        )
        cuisines = self._read_index(
            paths.cuisines,
            "cuisine_id",
            issues,
        )
        dishes = self._read_index(paths.dishes, "dish_id", issues)

        self._validate_foreign_keys(
            paths.ingredient_aliases,
            "ingredient_id",
            set(ingredients),
            issues,
        )
        self._validate_foreign_keys(
            paths.cuisine_aliases,
            "cuisine_id",
            set(cuisines),
            issues,
        )
        self._validate_foreign_keys(
            paths.cuisine_ingredients,
            "cuisine_id",
            set(cuisines),
            issues,
        )
        self._validate_foreign_keys(
            paths.cuisine_ingredients,
            "ingredient_id",
            set(ingredients),
            issues,
        )
        self._validate_foreign_keys(
            paths.dish_ingredients,
            "dish_id",
            set(dishes),
            issues,
        )
        self._validate_foreign_keys(
            paths.dish_ingredients,
            "ingredient_id",
            set(ingredients),
            issues,
        )
        self._validate_foreign_keys(
            paths.recipes,
            "dish_id",
            set(dishes),
            issues,
        )
        self._validate_foreign_keys(
            paths.recipes,
            "cuisine_id",
            set(cuisines),
            issues,
            allow_empty=True,
        )

        for row_number, row in self._read_rows(paths.dishes, issues):
            cuisine_id = (row.get("cuisine_id") or "").strip()
            if cuisine_id and cuisine_id not in cuisines:
                issues.append(
                    ConsistencyIssue(
                        "UNKNOWN_DISH_CUISINE",
                        f"Unknown cuisine_id '{cuisine_id}'",
                        paths.dishes.name,
                        row_number,
                    )
                )

        graph = self._load_graph(paths.ontology, issues)
        if graph is not None:
            self._validate_ontology_entities(
                graph,
                bundle.metadata.ontology_base_iri,
                ingredients,
                cuisines,
                dishes,
                issues,
            )
            self._validate_manifest_counts(
                bundle,
                len(ingredients),
                len(cuisines),
                len(dishes),
                len(graph),
                issues,
            )

        return KnowledgeBaseConsistencyReport(
            bundle.metadata.knowledge_base_id,
            tuple(issues),
        )

    @staticmethod
    def _read_index(
        path: Path,
        key_column: str,
        issues: list[ConsistencyIssue],
    ) -> dict[str, dict[str, str]]:
        index: dict[str, dict[str, str]] = {}
        for row_number, row in KnowledgeBaseConsistencyValidator._read_rows(
            path,
            issues,
        ):
            key = (row.get(key_column) or "").strip()
            if not key:
                issues.append(
                    ConsistencyIssue(
                        "EMPTY_PRIMARY_KEY",
                        f"Empty {key_column}",
                        path.name,
                        row_number,
                    )
                )
                continue
            if key in index:
                issues.append(
                    ConsistencyIssue(
                        "DUPLICATE_PRIMARY_KEY",
                        f"Duplicate {key_column} '{key}'",
                        path.name,
                        row_number,
                    )
                )
                continue
            index[key] = row
        return index

    @staticmethod
    def _read_rows(
        path: Path,
        issues: list[ConsistencyIssue],
    ) -> tuple[tuple[int, dict[str, str]], ...]:
        try:
            with path.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None:
                    issues.append(
                        ConsistencyIssue(
                            "MISSING_CSV_HEADER",
                            "CSV file has no header",
                            path.name,
                        )
                    )
                    return ()
                return tuple(
                    (row_number, dict(row))
                    for row_number, row in enumerate(reader, start=2)
                )
        except (OSError, csv.Error) as error:
            issues.append(
                ConsistencyIssue(
                    "CSV_READ_ERROR",
                    str(error),
                    path.name,
                )
            )
            return ()

    @staticmethod
    def _validate_foreign_keys(
        path: Path,
        column: str,
        valid_values: set[str],
        issues: list[ConsistencyIssue],
        *,
        allow_empty: bool = False,
    ) -> None:
        for row_number, row in KnowledgeBaseConsistencyValidator._read_rows(
            path,
            issues,
        ):
            value = (row.get(column) or "").strip()
            if not value and allow_empty:
                continue
            if value not in valid_values:
                issues.append(
                    ConsistencyIssue(
                        "UNKNOWN_FOREIGN_KEY",
                        f"Unknown {column} '{value}'",
                        path.name,
                        row_number,
                    )
                )

    @staticmethod
    def _load_graph(
        path: Path,
        issues: list[ConsistencyIssue],
    ) -> Graph | None:
        graph = Graph()
        try:
            graph.parse(path, format="xml")
        except Exception as error:
            issues.append(
                ConsistencyIssue(
                    "RDF_PARSE_ERROR",
                    str(error),
                    path.name,
                )
            )
            return None
        return graph

    @staticmethod
    def _validate_ontology_entities(
        graph: Graph,
        base_iri: str,
        ingredients: dict[str, dict[str, str]],
        cuisines: dict[str, dict[str, str]],
        dishes: dict[str, dict[str, str]],
        issues: list[ConsistencyIssue],
    ) -> None:
        for file_name, records in (
            ("ingredients.csv", ingredients),
            ("cuisines.csv", cuisines),
            ("dishes.csv", dishes),
        ):
            for identifier, row in records.items():
                ontology_id = (row.get("ontology_id") or "").strip()
                if not ontology_id:
                    issues.append(
                        ConsistencyIssue(
                            "MISSING_ONTOLOGY_ID",
                            f"Missing ontology_id for '{identifier}'",
                            file_name,
                        )
                    )
                    continue
                iri = URIRef(base_iri + ontology_id)
                if not any(graph.triples((iri, None, None))):
                    issues.append(
                        ConsistencyIssue(
                            "MISSING_RDF_ENTITY",
                            f"RDF entity does not exist: {iri}",
                            file_name,
                        )
                    )

    @staticmethod
    def _validate_manifest_counts(
        bundle: KnowledgeBaseBundle,
        ingredient_count: int,
        cuisine_count: int,
        dish_count: int,
        triple_count: int,
        issues: list[ConsistencyIssue],
    ) -> None:
        expected = {
            "ingredient_count": ingredient_count,
            "cuisine_count": cuisine_count,
            "dish_count": dish_count,
            "triple_count": triple_count,
        }
        metadata = bundle.metadata
        for field_name, actual_count in expected.items():
            manifest_count = getattr(metadata, field_name)
            if manifest_count != actual_count:
                issues.append(
                    ConsistencyIssue(
                        "MANIFEST_COUNT_MISMATCH",
                        (
                            f"{field_name} is {manifest_count} in the manifest "
                            f"but {actual_count} in bundle files"
                        ),
                        "knowledge_base.json",
                    )
                )
