from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeBaseMetadata:
    """Describes the identity, source, version, and size of one knowledge base."""

    knowledge_base_id: str
    name: str
    version: str
    source: str
    language: str
    ontology_base_iri: str
    dish_count: int = 0
    ingredient_count: int = 0
    cuisine_count: int = 0
    triple_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "knowledge_base_id",
            "name",
            "version",
            "source",
            "language",
            "ontology_base_iri",
        ):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        for field_name in (
            "dish_count",
            "ingredient_count",
            "cuisine_count",
            "triple_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True)
class KnowledgeBasePaths:
    """Contains every file path belonging to one consistent knowledge base bundle."""

    root: Path
    ontology: Path
    ingredients: Path
    ingredient_aliases: Path
    cuisines: Path
    cuisine_aliases: Path
    cuisine_ingredients: Path
    dishes: Path
    dish_ingredients: Path
    recipes: Path
    search_index: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "KnowledgeBasePaths":
        """Builds the standard bundle layout from its root directory."""
        bundle_root = Path(root)
        return cls(
            root=bundle_root,
            ontology=bundle_root / "cuisine.rdf",
            ingredients=bundle_root / "ingredients.csv",
            ingredient_aliases=bundle_root / "ingredient_aliases.csv",
            cuisines=bundle_root / "cuisines.csv",
            cuisine_aliases=bundle_root / "cuisine_aliases.csv",
            cuisine_ingredients=bundle_root / "cuisine_ingredients.csv",
            dishes=bundle_root / "dishes.csv",
            dish_ingredients=bundle_root / "dish_ingredients.csv",
            recipes=bundle_root / "recipes.csv",
            search_index=bundle_root / "search_index.csv",
            manifest=bundle_root / "knowledge_base.json",
        )

    def required_files(self) -> tuple[Path, ...]:
        """Returns every mandatory file that must switch with the active bundle."""
        return (
            self.ontology,
            self.ingredients,
            self.ingredient_aliases,
            self.cuisines,
            self.cuisine_aliases,
            self.cuisine_ingredients,
            self.dishes,
            self.dish_ingredients,
            self.recipes,
            self.search_index,
            self.manifest,
        )


@dataclass(frozen=True)
class KnowledgeBaseBundle:
    """Represents one selectable and internally consistent knowledge base bundle."""

    metadata: KnowledgeBaseMetadata
    paths: KnowledgeBasePaths


@dataclass(frozen=True)
class KnowledgeBaseValidationResult:
    """Reports whether a bundle is complete and lists missing mandatory files."""

    is_valid: bool
    missing_files: tuple[Path, ...] = ()
