from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuisine_assistant.application import LinkIngredientsUseCase
from cuisine_assistant.catalog import CsvIngredientCatalog
from cuisine_assistant.knowledge_graph import RdfLibKnowledgeGraphRepository
from cuisine_assistant.linking import IngredientEntityLinker
from cuisine_assistant.nlp.normalizer import EnglishTextNormalizer
from cuisine_assistant.nlp.spacy_extractor import SpacyPhraseIngredientExtractor
from cuisine_assistant.search.lexical_search import CsvLexicalEntitySearch


@dataclass(frozen=True)
class ApplicationSettings:
    project_root: Path
    ontology_base_iri: str
    ontology_relative_path: Path = Path("ontology/cuisine.owl")
    ingredients_relative_path: Path = Path("data/processed/ingredients.csv")
    aliases_relative_path: Path = Path("data/processed/ingredient_aliases.csv")

    @property
    def ontology_path(self) -> Path:
        return self.project_root / self.ontology_relative_path

    @property
    def ingredients_path(self) -> Path:
        return self.project_root / self.ingredients_relative_path

    @property
    def aliases_path(self) -> Path:
        return self.project_root / self.aliases_relative_path


@dataclass(frozen=True)
class ApplicationContainer:
    link_ingredients: LinkIngredientsUseCase


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
    return ApplicationContainer(link_ingredients=link_ingredients)
