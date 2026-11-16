from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuisine_assistant.application import (
    ClassifyRecipeUseCase,
    LinkIngredientsUseCase,
    RecommendDishesUseCase,
)
from cuisine_assistant.catalog import (
    CsvCuisineCatalog,
    CsvCuisinePreferenceCatalog,
    CsvCuisineTypicalityRepository,
    CsvDishCatalog,
    CsvIngredientCatalog,
)
from cuisine_assistant.classification import TypicalityCuisineClassifier
from cuisine_assistant.knowledge_graph import RdfLibKnowledgeGraphRepository
from cuisine_assistant.linking import IngredientEntityLinker
from cuisine_assistant.nlp import (
    EnglishTextNormalizer,
    LexicalCuisinePreferenceExtractor,
    SpacyPhraseIngredientExtractor,
)
from cuisine_assistant.recommendation import KnowledgeBasedDishRecommender
from cuisine_assistant.search.lexical_search import CsvLexicalEntitySearch


@dataclass(frozen=True)
class ApplicationSettings:
    """Defines project paths and scoring settings for dependency wiring."""

    project_root: Path
    ontology_base_iri: str
    ontology_relative_path: Path = Path("ontology/cuisine.rdf")
    ingredients_relative_path: Path = Path("data/processed/ingredients.csv")
    aliases_relative_path: Path = Path("data/processed/ingredient_aliases.csv")
    cuisines_relative_path: Path = Path("data/processed/cuisines.csv")
    cuisine_ingredients_relative_path: Path = Path(
        "data/processed/cuisine_ingredients.csv"
    )
    dishes_relative_path: Path = Path("data/processed/dishes.csv")
    cuisine_aliases_relative_path: Path = Path(
        "data/processed/cuisine_aliases.csv"
    )
    maximum_predictions: int = 3
    minimum_classification_score: float = 0.0
    maximum_recommendations: int = 3
    minimum_recommendation_score: float = 0.0

    @property
    def ontology_path(self) -> Path:
        """Returns the resolved RDF/XML ontology path."""
        return self.project_root / self.ontology_relative_path

    @property
    def ingredients_path(self) -> Path:
        """Returns the resolved canonical ingredient catalog path."""
        return self.project_root / self.ingredients_relative_path

    @property
    def aliases_path(self) -> Path:
        """Returns the resolved ingredient alias catalog path."""
        return self.project_root / self.aliases_relative_path

    @property
    def cuisines_path(self) -> Path:
        """Returns the resolved canonical cuisine catalog path."""
        return self.project_root / self.cuisines_relative_path

    @property
    def cuisine_ingredients_path(self) -> Path:
        """Returns the resolved cuisine typicality table path."""
        return self.project_root / self.cuisine_ingredients_relative_path

    @property
    def dishes_path(self) -> Path:
        """Returns the resolved canonical dish catalog path."""
        return self.project_root / self.dishes_relative_path

    @property
    def cuisine_aliases_path(self) -> Path:
        """Returns the resolved cuisine preference alias path."""
        return self.project_root / self.cuisine_aliases_relative_path


@dataclass(frozen=True)
class ApplicationContainer:
    """Exposes fully constructed application use cases to external adapters."""

    link_ingredients: LinkIngredientsUseCase
    classify_recipe: ClassifyRecipeUseCase
    recommend_dishes: RecommendDishesUseCase


def build_application_container(
    settings: ApplicationSettings,
) -> ApplicationContainer:
    """Constructs adapters and injects them into application use cases."""
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

    cuisine_preference_catalog = CsvCuisinePreferenceCatalog(
        csv_path=settings.cuisine_aliases_path,
        cuisine_catalog=cuisine_catalog,
    )
    cuisine_preference_extractor = LexicalCuisinePreferenceExtractor(
        cuisine_preference_catalog
    )
    dish_catalog = CsvDishCatalog(settings.dishes_path)
    dish_recommender = KnowledgeBasedDishRecommender(
        dish_catalog=dish_catalog,
        ingredient_catalog=ingredient_catalog,
        knowledge_graph=knowledge_graph,
        maximum_recommendations=settings.maximum_recommendations,
        minimum_score=settings.minimum_recommendation_score,
    )
    recommend_dishes = RecommendDishesUseCase(
        classify_recipe=classify_recipe,
        cuisine_preference_extractor=cuisine_preference_extractor,
        dish_recommender=dish_recommender,
    )

    return ApplicationContainer(
        link_ingredients=link_ingredients,
        classify_recipe=classify_recipe,
        recommend_dishes=recommend_dishes,
    )
