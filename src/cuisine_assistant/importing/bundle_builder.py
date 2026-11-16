from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

from cuisine_assistant.importing.models import TextualImportBatch


@dataclass(frozen=True)
class BundleBuildConfiguration:
    """Defines identity and ontology settings for one generated bundle."""

    knowledge_base_id: str
    name: str
    version: str
    ontology_base_iri: str
    destination: Path
    source_name: str
    language: str = "en"

    def __post_init__(self) -> None:
        for field_name in (
            "knowledge_base_id",
            "name",
            "version",
            "ontology_base_iri",
            "source_name",
            "language",
        ):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not self.ontology_base_iri.endswith(("#", "/")):
            raise ValueError("ontology_base_iri must end with '#' or '/'")


class ImportedKnowledgeBaseBundleBuilder:
    """Builds and atomically installs a complete bundle from normalized text records."""

    def build(
        self,
        batch: TextualImportBatch,
        configuration: BundleBuildConfiguration,
    ) -> Path:
        """Generates every CSV, RDF/XML, index, and manifest file as one unit."""
        destination = configuration.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix="bundle-build-", dir=destination.parent)
        )
        try:
            normalized = self._collect(batch)
            self._write_csv_files(temporary, normalized, batch)
            triple_count = self._write_rdf(temporary, normalized, configuration)
            self._write_manifest(
                temporary,
                normalized,
                batch,
                configuration,
                triple_count,
            )
            self._validate_complete_bundle(temporary)
            self._install_atomically(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def _collect(self, batch: TextualImportBatch) -> dict:
        dishes: dict[str, dict] = {}
        cuisines: dict[str, object] = {}
        ingredients: dict[str, dict] = {}
        dish_ingredients: set[tuple[str, str]] = set()
        cuisine_ingredient_counts: Counter[tuple[str, str]] = Counter()
        cuisine_dish_counts: Counter[str] = Counter()
        ingredient_aliases: set[tuple[str, str]] = set()
        cuisine_aliases: set[tuple[str, str]] = set()

        for record in batch.records:
            dish_id = record.external_id
            primary_cuisine_id = (
                record.cuisines[0].cuisine_id
                if record.cuisines
                else "uncategorized"
            )
            ontology_id = self._ontology_id(record.name, "Dish")
            category = record.categories[0] if record.categories else "Dish"
            dishes[dish_id] = {
                "dish_id": dish_id,
                "name": record.name,
                "dish_category": self._ontology_id(category),
                "cuisine_id": primary_cuisine_id,
                "description": record.description or "",
                "ontology_id": ontology_id,
                "source": batch.source.source_id,
                "source_url": record.source_url or "",
            }

            for cuisine in record.cuisines:
                cuisines.setdefault(cuisine.cuisine_id, cuisine)
                cuisine_dish_counts[cuisine.cuisine_id] += 1
                for alias in cuisine.aliases:
                    cuisine_aliases.add((cuisine.cuisine_id, alias))

            unique_ingredient_ids: set[str] = set()
            for mention in record.ingredients:
                ingredient_id = self._slugify(mention.canonical_text)
                if not ingredient_id:
                    continue
                ingredients.setdefault(
                    ingredient_id,
                    {
                        "ingredient_id": ingredient_id,
                        "canonical_name": mention.canonical_text.replace("_", " "),
                        "ingredient_category": "Ingredient",
                        "description": "",
                        "ontology_id": self._ontology_id(mention.canonical_text),
                    },
                )
                ingredient_aliases.add((ingredient_id, mention.surface_text))
                dish_ingredients.add((dish_id, ingredient_id))
                unique_ingredient_ids.add(ingredient_id)

            for cuisine in record.cuisines:
                for ingredient_id in unique_ingredient_ids:
                    cuisine_ingredient_counts[
                        (cuisine.cuisine_id, ingredient_id)
                    ] += 1

        return {
            "dishes": dishes,
            "cuisines": cuisines,
            "ingredients": ingredients,
            "dish_ingredients": dish_ingredients,
            "cuisine_ingredient_counts": cuisine_ingredient_counts,
            "cuisine_dish_counts": cuisine_dish_counts,
            "ingredient_aliases": ingredient_aliases,
            "cuisine_aliases": cuisine_aliases,
        }

    def _write_csv_files(self, root: Path, data: dict, batch: TextualImportBatch) -> None:
        self._write_csv(
            root / "dishes.csv",
            (
                "dish_id", "name", "dish_category", "cuisine_id",
                "description", "ontology_id", "source", "source_url",
            ),
            data["dishes"].values(),
        )
        self._write_csv(
            root / "ingredients.csv",
            (
                "ingredient_id", "canonical_name", "ingredient_category",
                "description", "ontology_id",
            ),
            data["ingredients"].values(),
        )

        cuisine_rows = []
        for cuisine_id, cuisine in data["cuisines"].items():
            cuisine_rows.append(
                {
                    "cuisine_id": cuisine_id,
                    "name": cuisine.name,
                    "cuisine_group": "Cuisine",
                    "origin_place_id": cuisine.origin_place_id or "",
                    "description": "",
                    "ontology_id": cuisine.ontology_id,
                }
            )
        self._write_csv(
            root / "cuisines.csv",
            (
                "cuisine_id", "name", "cuisine_group", "origin_place_id",
                "description", "ontology_id",
            ),
            cuisine_rows,
        )

        ingredient_alias_rows = []
        for index, (ingredient_id, alias) in enumerate(
            sorted(data["ingredient_aliases"]), start=1
        ):
            ingredient_alias_rows.append(
                {
                    "alias_id": f"ingredient_alias_{index}",
                    "ingredient_id": ingredient_id,
                    "alias": alias,
                    "alias_type": "imported",
                    "language": batch.source.language,
                    "match_priority": 80,
                    "verified": "false",
                }
            )
        self._write_csv(
            root / "ingredient_aliases.csv",
            (
                "alias_id", "ingredient_id", "alias", "alias_type",
                "language", "match_priority", "verified",
            ),
            ingredient_alias_rows,
        )

        cuisine_alias_rows = []
        for index, (cuisine_id, alias) in enumerate(
            sorted(data["cuisine_aliases"]), start=1
        ):
            cuisine_alias_rows.append(
                {
                    "alias_id": f"cuisine_alias_{index}",
                    "cuisine_id": cuisine_id,
                    "alias": alias,
                    "alias_type": "imported",
                    "language": batch.source.language,
                    "match_priority": 80,
                    "verified": "false",
                }
            )
        self._write_csv(
            root / "cuisine_aliases.csv",
            (
                "alias_id", "cuisine_id", "alias", "alias_type",
                "language", "match_priority", "verified",
            ),
            cuisine_alias_rows,
        )

        dish_ingredient_rows = [
            {
                "dish_id": dish_id,
                "ingredient_id": ingredient_id,
                "importance": 1.0,
                "is_core": "false",
                "is_optional": "false",
                "preparation": "",
                "quantity_text": "",
                "source": batch.source.source_id,
            }
            for dish_id, ingredient_id in sorted(data["dish_ingredients"])
        ]
        self._write_csv(
            root / "dish_ingredients.csv",
            (
                "dish_id", "ingredient_id", "importance", "is_core",
                "is_optional", "preparation", "quantity_text", "source",
            ),
            dish_ingredient_rows,
        )

        typicality_rows = []
        for (cuisine_id, ingredient_id), count in sorted(
            data["cuisine_ingredient_counts"].items()
        ):
            denominator = data["cuisine_dish_counts"][cuisine_id]
            score = count / denominator if denominator else 0.0
            typicality_rows.append(
                {
                    "cuisine_id": cuisine_id,
                    "ingredient_id": ingredient_id,
                    "typicality_score": f"{score:.6f}",
                    "evidence_count": count,
                    "source": batch.source.source_id,
                }
            )
        self._write_csv(
            root / "cuisine_ingredients.csv",
            (
                "cuisine_id", "ingredient_id", "typicality_score",
                "evidence_count", "source",
            ),
            typicality_rows,
        )

        recipe_rows = []
        for index, record in enumerate(batch.records, start=1):
            dish = data["dishes"][record.external_id]
            recipe_rows.append(
                {
                    "recipe_id": f"imported_recipe_{index}",
                    "dish_id": record.external_id,
                    "title": record.name,
                    "raw_text": record.description or record.name,
                    "ingredient_text": "; ".join(
                        mention.surface_text for mention in record.ingredients
                    ),
                    "instruction_text": "",
                    "cuisine_id": dish["cuisine_id"],
                    "data_split": "imported",
                    "source": batch.source.source_id,
                    "source_url": record.source_url or "",
                    "license": record.source_license or "",
                }
            )
        self._write_csv(
            root / "recipes.csv",
            (
                "recipe_id", "dish_id", "title", "raw_text",
                "ingredient_text", "instruction_text", "cuisine_id",
                "data_split", "source", "source_url", "license",
            ),
            recipe_rows,
        )

        index_rows = []
        for ingredient_id, ingredient in sorted(data["ingredients"].items()):
            index_rows.append(
                {
                    "normalized_term": ingredient["canonical_name"],
                    "entity_type": "ingredient",
                    "entity_id": ingredient_id,
                    "source_text": ingredient["canonical_name"],
                    "priority": 100,
                }
            )
        for ingredient_id, alias in sorted(data["ingredient_aliases"]):
            index_rows.append(
                {
                    "normalized_term": alias.lower(),
                    "entity_type": "ingredient",
                    "entity_id": ingredient_id,
                    "source_text": alias,
                    "priority": 80,
                }
            )
        for dish_id, dish in sorted(data["dishes"].items()):
            index_rows.append(
                {
                    "normalized_term": dish["name"].lower(),
                    "entity_type": "dish",
                    "entity_id": dish_id,
                    "source_text": dish["name"],
                    "priority": 100,
                }
            )
        self._write_csv(
            root / "search_index.csv",
            (
                "normalized_term", "entity_type", "entity_id",
                "source_text", "priority",
            ),
            index_rows,
        )

    def _write_rdf(
        self,
        root: Path,
        data: dict,
        configuration: BundleBuildConfiguration,
    ) -> int:
        rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        owl = "http://www.w3.org/2002/07/owl#"
        rdfs = "http://www.w3.org/2000/01/rdf-schema#"
        ET.register_namespace("rdf", rdf)
        ET.register_namespace("owl", owl)
        ET.register_namespace("rdfs", rdfs)
        root_element = ET.Element(f"{{{rdf}}}RDF")

        triple_count = 0
        for class_id in ("Dish", "Ingredient", "Cuisine"):
            element = ET.SubElement(root_element, f"{{{owl}}}Class")
            element.set(f"{{{rdf}}}about", configuration.ontology_base_iri + class_id)
            triple_count += 1

        for property_id in ("hasIngredient", "belongsToCuisine"):
            element = ET.SubElement(root_element, f"{{{owl}}}ObjectProperty")
            element.set(f"{{{rdf}}}about", configuration.ontology_base_iri + property_id)
            triple_count += 1

        for cuisine in data["cuisines"].values():
            individual = ET.SubElement(root_element, f"{{{owl}}}NamedIndividual")
            individual.set(
                f"{{{rdf}}}about",
                configuration.ontology_base_iri + cuisine.ontology_id,
            )
            type_element = ET.SubElement(individual, f"{{{rdf}}}type")
            type_element.set(
                f"{{{rdf}}}resource",
                configuration.ontology_base_iri + "Cuisine",
            )
            label = ET.SubElement(individual, f"{{{rdfs}}}label")
            label.text = cuisine.name
            triple_count += 3

        for ingredient in data["ingredients"].values():
            individual = ET.SubElement(root_element, f"{{{owl}}}NamedIndividual")
            individual.set(
                f"{{{rdf}}}about",
                configuration.ontology_base_iri + ingredient["ontology_id"],
            )
            type_element = ET.SubElement(individual, f"{{{rdf}}}type")
            type_element.set(
                f"{{{rdf}}}resource",
                configuration.ontology_base_iri + "Ingredient",
            )
            label = ET.SubElement(individual, f"{{{rdfs}}}label")
            label.text = ingredient["canonical_name"]
            triple_count += 3

        dish_links: dict[str, list[str]] = defaultdict(list)
        for dish_id, ingredient_id in data["dish_ingredients"]:
            dish_links[dish_id].append(ingredient_id)
        for dish in data["dishes"].values():
            individual = ET.SubElement(root_element, f"{{{owl}}}NamedIndividual")
            individual.set(
                f"{{{rdf}}}about",
                configuration.ontology_base_iri + dish["ontology_id"],
            )
            type_element = ET.SubElement(individual, f"{{{rdf}}}type")
            type_element.set(
                f"{{{rdf}}}resource",
                configuration.ontology_base_iri + "Dish",
            )
            label = ET.SubElement(individual, f"{{{rdfs}}}label")
            label.text = dish["name"]
            triple_count += 3
            cuisine = data["cuisines"].get(dish["cuisine_id"])
            if cuisine is not None:
                relation = ET.SubElement(
                    individual,
                    f"{{{configuration.ontology_base_iri}}}belongsToCuisine",
                )
                relation.set(
                    f"{{{rdf}}}resource",
                    configuration.ontology_base_iri + cuisine.ontology_id,
                )
                triple_count += 1
            for ingredient_id in sorted(dish_links[dish["dish_id"]]):
                ingredient = data["ingredients"][ingredient_id]
                relation = ET.SubElement(
                    individual,
                    f"{{{configuration.ontology_base_iri}}}hasIngredient",
                )
                relation.set(
                    f"{{{rdf}}}resource",
                    configuration.ontology_base_iri + ingredient["ontology_id"],
                )
                triple_count += 1

        ET.ElementTree(root_element).write(
            root / "cuisine.rdf",
            encoding="utf-8",
            xml_declaration=True,
        )
        return triple_count

    def _write_manifest(
        self,
        root: Path,
        data: dict,
        batch: TextualImportBatch,
        configuration: BundleBuildConfiguration,
        triple_count: int,
    ) -> None:
        document = {
            "knowledge_base_id": configuration.knowledge_base_id,
            "name": configuration.name,
            "version": configuration.version,
            "source": configuration.source_name,
            "language": configuration.language,
            "ontology_base_iri": configuration.ontology_base_iri,
            "created_at": datetime.now(UTC).isoformat(),
            "import": {
                "source_id": batch.source.source_id,
                "dataset_name": batch.source.dataset_name,
                "split": batch.source.split,
                "revision": batch.source.revision,
                "maximum_records": batch.options.maximum_records,
                "imported_records": len(batch.records),
                "skipped_records": batch.skipped_record_count,
                "duplicate_records": batch.duplicate_record_count,
                "images_included": False,
            },
            "counts": {
                "dish_count": len(data["dishes"]),
                "ingredient_count": len(data["ingredients"]),
                "cuisine_count": len(data["cuisines"]),
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
    def _write_csv(path: Path, fieldnames: tuple[str, ...], rows) -> None:
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _validate_complete_bundle(root: Path) -> None:
        required = {
            "knowledge_base.json", "cuisine.rdf", "ingredients.csv",
            "ingredient_aliases.csv", "cuisines.csv", "cuisine_aliases.csv",
            "cuisine_ingredients.csv", "dishes.csv", "dish_ingredients.csv",
            "recipes.csv", "search_index.csv",
        }
        actual = {path.name for path in root.iterdir() if path.is_file()}
        missing = required - actual
        if missing:
            raise ValueError(
                "Generated bundle is incomplete: " + ", ".join(sorted(missing))
            )
        image_suffixes = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
        if any(path.suffix.lower() in image_suffixes for path in root.rglob("*")):
            raise ValueError("Generated bundle must not contain image files")

    @staticmethod
    def _install_atomically(temporary: Path, destination: Path) -> None:
        backup = destination.with_name(
            f"{destination.name}.backup-{uuid4().hex}"
        )
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

    @staticmethod
    def _slugify(value: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    @staticmethod
    def _ontology_id(value: str, suffix: str = "") -> str:
        import re
        words = re.findall(r"[a-zA-Z0-9]+", value)
        identifier = "".join(word[:1].upper() + word[1:] for word in words)
        if suffix and not identifier.lower().endswith(suffix.lower()):
            identifier += suffix
        return identifier or suffix


class WorldCuisinesBundleBuilder:
    """Builds the World Cuisines bundle from an already normalized import batch."""

    def __init__(self, builder: ImportedKnowledgeBaseBundleBuilder) -> None:
        self._builder = builder

    def build(
        self,
        batch: TextualImportBatch,
        destination: Path,
        ontology_base_iri: str,
        *,
        version: str = "1.0",
    ) -> Path:
        """Builds the standard world_cuisines bundle with fixed source metadata."""
        return self._builder.build(
            batch,
            BundleBuildConfiguration(
                knowledge_base_id="world_cuisines",
                name="World Cuisines Knowledge Base",
                version=version,
                ontology_base_iri=ontology_base_iri,
                destination=destination,
                source_name="worldcuisines/food-kb",
                language=batch.source.language,
            ),
        )
