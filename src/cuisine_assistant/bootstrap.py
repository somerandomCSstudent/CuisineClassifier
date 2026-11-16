from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuisine_assistant.application import (
    ClassifyRecipeUseCase,
    LinkIngredientsUseCase,
)
from cuisine_assistant.catalog import (
    CsvCuisineCatalog,
    CsvCuisineTypicalityRepository,
    CsvIngredientCatalog,
)
from cuisine_assistant.classification import TypicalityCuisineClassifier
from cuisine_assistant.knowledge_graph import RdfLibKnowledgeGraphRepository
from cuisine_assistant.linking import IngredientEntityLinker
from cuisine_assistant.nlp.normalizer import EnglishTextNormalizer
from cuisine_assistant.nlp.spacy_extractor import SpacyPhraseIngredientExtractor
from cuisine_assistant.search.lexical_search import CsvLexicalEntitySearch


@dataclass(frozen=True)
class ApplicationSettings:
    """Defines project paths and classification settings for dependency wiring."""

    project_root: Path
    ontology_base_iri: str
    ontology_relative_path: Path = Path("ontology/cuisine.rdf")
    ingredients_relative_path: Path = Path("data/processed/ingredients.csv")
    aliases_relative_path: Path = Path("data/processed/ingredient_aliases.csv")
    cuisines_relative_path: Path = Path("data/processed/cuisines.csv")
    cuisine_ingredients_relative_path: Path = Path(
        "data/processed/cuisine_ingredients.csv"
    )
    maximum_predictions: int = 3
    minimum_classification_score: float = 0.0

    @property
    def ontology_path(self) -> Path:
        return self.project_root / self.ontology_relative_path

    @property
    def ingredients_path(self) -> Path:
        return self.project_root / self.ingredients_relative_path

    @property
    def aliases_path(self) -> Path:
        return self.project_root / self.aliases_relative_path

    @property
    def cuisines_path(self) -> Path:
        return self.project_root / self.cuisines_relative_path

    @property
    def cuisine_ingredients_path(self) -> Path:
        return self.project_root / self.cuisine_ingredients_relative_path


@dataclass(frozen=True)
class ApplicationContainer:
    """Exposes fully constructed application use cases to external adapters."""

    link_ingredients: LinkIngredientsUseCase
    classify_recipe: ClassifyRecipeUseCase


def build_application_container(
    settings: ApplicationSettings,
) -> ApplicationContainer:
    """Constructs concrete adapters and injects them into application services."""
    normalizer = EnglishTextNormalizer()
    entity_search = CsvLexicalEntitySearch(
        aliases_csv=settings.aliases_path,
        normalizer=normalizer,
    )
    extractor = SpacyPhraseIngredientExtractor(
        normalizer=normalizer,
        search=entity_search,
    )
    ingredient_catalog = CsvIngredientCatalog(settings.ingredients_path)
    knowledge_graph = RdfLibKnowledgeGraphRepository(
        ontology_path=settings.ontology_path,
        base_iri=settings.ontology_base_iri,
    )
    entity_linker = IngredientEntityLinker(
        ingredient_catalog=ingredient_catalog,
        knowledge_graph=knowledge_graph,
    )
    link_ingredients = LinkIngredientsUseCase(
        extractor=extractor,
        linker=entity_linker,
    )

    cuisine_catalog = CsvCuisineCatalog(settings.cuisines_path)
    typicality_repository = CsvCuisineTypicalityRepository(
        settings.cuisine_ingredients_path
    )
    classifier = TypicalityCuisineClassifier(
        cuisine_catalog=cuisine_catalog,
        typicality_repository=typicality_repository,
        maximum_predictions=settings.maximum_predictions,
        minimum_score=settings.minimum_classification_score,
    )
    classify_recipe = ClassifyRecipeUseCase(
        link_ingredients=link_ingredients,
        classifier=classifier,
    )

    return ApplicationContainer(
        link_ingredients=link_ingredients,
        classify_recipe=classify_recipe,
    )
