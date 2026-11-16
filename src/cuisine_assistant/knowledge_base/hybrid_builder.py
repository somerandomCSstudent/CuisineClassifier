from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rdflib import Graph, Namespace, OWL, RDF, RDFS, URIRef

from cuisine_assistant.knowledge_base.models import KnowledgeBaseBundle


class HybridKnowledgeBaseBuilder:
    """Merges curated and imported bundles with curated records taking precedence."""

    _CATALOGS = {
        "ingredients.csv": "ingredient_id",
        "cuisines.csv": "cuisine_id",
        "dishes.csv": "dish_id",
        "recipes.csv": "recipe_id",
    }
    _RELATIONS = (
        "ingredient_aliases.csv",
        "cuisine_aliases.csv",
        "cuisine_ingredients.csv",
        "dish_ingredients.csv",
        "search_index.csv",
    )

    def build(
        self,
        curated: KnowledgeBaseBundle,
        imported: KnowledgeBaseBundle,
        destination: Path,
        *,
        ontology_base_iri: str,
        version: str = "1.0",
    ) -> Path:
        """Builds, validates structurally, and atomically installs a hybrid bundle."""
        if not ontology_base_iri.endswith(("#", "/")):
            raise ValueError("ontology_base_iri must end with '#' or '/'")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="hybrid-build-", dir=destination.parent))
        try:
            counts = self._merge_csv_files(curated, imported, temporary)
            triple_count = self._generate_rdf(temporary, ontology_base_iri)
            self._write_manifest(
                temporary,
                curated,
                imported,
                ontology_base_iri,
                version,
                counts,
                triple_count,
            )
            self._validate_files(temporary)
            self._install_atomically(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def _merge_csv_files(self, curated, imported, target: Path) -> dict[str, int]:
        counts: dict[str, int] = {}
        for file_name, key in self._CATALOGS.items():
            curated_rows, curated_fields = self._read_csv(curated.paths.root / file_name)
            imported_rows, imported_fields = self._read_csv(imported.paths.root / file_name)
            merged = {row[key]: row for row in imported_rows if row.get(key)}
            merged.update({row[key]: row for row in curated_rows if row.get(key)})
            fields = self._union_fields(curated_fields, imported_fields)
            self._write_csv(target / file_name, fields, merged.values())
            counts[file_name] = len(merged)

        for file_name in self._RELATIONS:
            curated_rows, curated_fields = self._read_csv(curated.paths.root / file_name)
            imported_rows, imported_fields = self._read_csv(imported.paths.root / file_name)
            fields = self._union_fields(curated_fields, imported_fields)
            merged: list[dict[str, str]] = []
            seen: set[tuple[str, ...]] = set()
            for row in (*curated_rows, *imported_rows):
                signature = tuple((row.get(field) or "").strip() for field in fields)
                if signature not in seen:
                    seen.add(signature)
                    merged.append(row)
            self._write_csv(target / file_name, fields, merged)
        return counts

    def _generate_rdf(self, root: Path, base_iri: str) -> int:
        graph = Graph()
        namespace = Namespace(base_iri)
        graph.bind("cuisine", namespace)
        for class_name in ("Dish", "Ingredient", "Cuisine"):
            graph.add((namespace[class_name], RDF.type, OWL.Class))
        graph.add((namespace.hasIngredient, RDF.type, OWL.ObjectProperty))
        graph.add((namespace.belongsToCuisine, RDF.type, OWL.ObjectProperty))

        ingredients, _ = self._read_csv(root / "ingredients.csv")
        cuisines, _ = self._read_csv(root / "cuisines.csv")
        dishes, _ = self._read_csv(root / "dishes.csv")
        relations, _ = self._read_csv(root / "dish_ingredients.csv")
        ingredient_by_id = {row["ingredient_id"]: row for row in ingredients}
        cuisine_by_id = {row["cuisine_id"]: row for row in cuisines}
        dish_by_id = {row["dish_id"]: row for row in dishes}

        for row in ingredients:
            subject = namespace[row["ontology_id"]]
            graph.add((subject, RDF.type, namespace.Ingredient))
            graph.add((subject, RDFS.label, self._literal(row.get("canonical_name"))))
        for row in cuisines:
            subject = namespace[row["ontology_id"]]
            graph.add((subject, RDF.type, namespace.Cuisine))
            graph.add((subject, RDFS.label, self._literal(row.get("name"))))
        for row in dishes:
            subject = namespace[row["ontology_id"]]
            graph.add((subject, RDF.type, namespace.Dish))
            graph.add((subject, RDFS.label, self._literal(row.get("name"))))
            cuisine = cuisine_by_id.get(row.get("cuisine_id", ""))
            if cuisine:
                graph.add(
                    (subject, namespace.belongsToCuisine, namespace[cuisine["ontology_id"]])
                )
        for relation in relations:
            dish = dish_by_id.get(relation.get("dish_id", ""))
            ingredient = ingredient_by_id.get(relation.get("ingredient_id", ""))
            if dish and ingredient:
                graph.add(
                    (
                        namespace[dish["ontology_id"]],
                        namespace.hasIngredient,
                        namespace[ingredient["ontology_id"]],
                    )
                )
        graph.serialize(root / "cuisine.rdf", format="xml")
        return len(graph)

    def _write_manifest(
        self, root, curated, imported, base_iri, version, counts, triple_count
    ) -> None:
        document = {
            "knowledge_base_id": "hybrid",
            "name": "Hybrid Cuisine Knowledge Base",
            "version": version,
            "source": f"{curated.metadata.knowledge_base_id}+{imported.metadata.knowledge_base_id}",
            "language": "en",
            "ontology_base_iri": base_iri,
            "created_at": datetime.now(UTC).isoformat(),
            "merge": {
                "curated_bundle": curated.metadata.knowledge_base_id,
                "imported_bundle": imported.metadata.knowledge_base_id,
                "conflict_policy": "curated_wins",
                "images_included": False,
            },
            "counts": {
                "dish_count": counts["dishes.csv"],
                "ingredient_count": counts["ingredients.csv"],
                "cuisine_count": counts["cuisines.csv"],
                "triple_count": triple_count,
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
        (root / "knowledge_base.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            return list(reader), list(reader.fieldnames or ())

    @staticmethod
    def _union_fields(first: list[str], second: list[str]) -> list[str]:
        return list(dict.fromkeys((*first, *second)))

    @staticmethod
    def _write_csv(path: Path, fields: list[str], rows) -> None:
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _literal(value: str | None):
        from rdflib import Literal
        return Literal(value or "")

    @staticmethod
    def _validate_files(root: Path) -> None:
        required = {
            "knowledge_base.json", "cuisine.rdf", "ingredients.csv",
            "ingredient_aliases.csv", "cuisines.csv", "cuisine_aliases.csv",
            "cuisine_ingredients.csv", "dishes.csv", "dish_ingredients.csv",
            "recipes.csv", "search_index.csv",
        }
        missing = required - {path.name for path in root.iterdir() if path.is_file()}
        if missing:
            raise ValueError("Hybrid bundle is incomplete: " + ", ".join(sorted(missing)))
        image_suffixes = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
        if any(path.suffix.lower() in image_suffixes for path in root.rglob("*")):
            raise ValueError("Images are not allowed in the hybrid bundle")

    @staticmethod
    def _install_atomically(temporary: Path, destination: Path) -> None:
        backup = destination.with_name(f"{destination.name}.backup-{uuid4().hex}")
        if destination.exists():
            destination.replace(backup)
        try:
            temporary.replace(destination)
        except Exception:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
