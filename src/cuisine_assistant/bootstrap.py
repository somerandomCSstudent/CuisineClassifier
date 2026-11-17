from __future__ import annotations

from dataclasses import dataclass

from cuisine_assistant.application import (
    ClassifyRecipeUseCase,
    LinkIngredientsUseCase,
    RecommendDishesUseCase,
    SearchRecipesUseCase,
)
from cuisine_assistant.catalog import (
    CsvCuisineCatalog,
    CsvCuisinePreferenceCatalog,
    CsvCuisineTypicalityRepository,
    CsvDishCatalog,
    CsvIngredientCatalog,
)
from cuisine_assistant.classification import TypicalityCuisineClassifier
from cuisine_assistant.knowledge_base.models import KnowledgeBaseBundle
from cuisine_assistant.knowledge_graph import RdfLibKnowledgeGraphRepository
from cuisine_assistant.linking import IngredientEntityLinker
from cuisine_assistant.nlp import (
    EnglishTextNormalizer,
    LexicalCuisinePreferenceExtractor,
    SpacyPhraseIngredientExtractor,
)
from cuisine_assistant.recommendation import KnowledgeBasedDishRecommender
from cuisine_assistant.retrieval import (
    Bm25RecipeRetriever,
    CsvRecipeDocumentRepository,
)
from cuisine_assistant.search.lexical_search import CsvLexicalEntitySearch


@dataclass(frozen=True)
class ApplicationSettings:
    """Defines scoring and result limits while paths come from a selected bundle."""

    maximum_predictions: int = 3
    minimum_classification_score: float = 0.0
    maximum_recommendations: int = 3
    minimum_recommendation_score: float = 0.0
    maximum_recipe_search_results: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    recipe_title_weight: float = 0.20
    recipe_cuisine_weight: float = 0.15
    recipe_ingredient_weight: float = 0.25

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_predictions",
            "maximum_recommendations",
            "maximum_recipe_search_results",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1")
        for field_name in (
            "minimum_classification_score",
            "minimum_recommendation_score",
            "bm25_b",
            "recipe_title_weight",
            "recipe_cuisine_weight",
            "recipe_ingredient_weight",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be greater than zero")


@dataclass(frozen=True)
class ApplicationContainer:
    """Exposes use cases built from one active knowledge base bundle."""

    knowledge_base: KnowledgeBaseBundle
    link_ingredients: LinkIngredientsUseCase
    classify_recipe: ClassifyRecipeUseCase
    recommend_dishes: RecommendDishesUseCase
    search_recipes: SearchRecipesUseCase


def build_application_container(
    bundle: KnowledgeBaseBundle,
    settings: ApplicationSettings | None = None,
) -> ApplicationContainer:
    """Builds every repository, retriever, and use case from one selected bundle."""
    effective_settings = settings or ApplicationSettings()
    paths = bundle.paths

    normalizer = EnglishTextNormalizer()
    entity_search = CsvLexicalEntitySearch(
        aliases_csv=paths.ingredient_aliases,
        normalizer=normalizer,
    )
    extractor = SpacyPhraseIngredientExtractor(
        normalizer=normalizer,
        search=entity_search,
    )

    ingredient_catalog = CsvIngredientCatalog(paths.ingredients)
    knowledge_graph = RdfLibKnowledgeGraphRepository(
        ontology_path=paths.ontology,
        base_iri=bundle.metadata.ontology_base_iri,
    )
    entity_linker = IngredientEntityLinker(
        ingredient_catalog=ingredient_catalog,
        knowledge_graph=knowledge_graph,
    )
    link_ingredients = LinkIngredientsUseCase(
        extractor=extractor,
        linker=entity_linker,
    )

    cuisine_catalog = CsvCuisineCatalog(paths.cuisines)
    typicality_repository = CsvCuisineTypicalityRepository(
        paths.cuisine_ingredients
    )
    classifier = TypicalityCuisineClassifier(
        cuisine_catalog=cuisine_catalog,
        typicality_repository=typicality_repository,
        maximum_predictions=effective_settings.maximum_predictions,
        minimum_score=effective_settings.minimum_classification_score,
    )
    classify_recipe = ClassifyRecipeUseCase(
        link_ingredients=link_ingredients,
        classifier=classifier,
    )

    cuisine_preference_catalog = CsvCuisinePreferenceCatalog(
        csv_path=paths.cuisine_aliases,
        cuisine_catalog=cuisine_catalog,
    )
    cuisine_preference_extractor = LexicalCuisinePreferenceExtractor(
        cuisine_preference_catalog
    )
    dish_catalog = CsvDishCatalog(paths.dishes)
    dish_recommender = KnowledgeBasedDishRecommender(
        dish_catalog=dish_catalog,
        ingredient_catalog=ingredient_catalog,
        knowledge_graph=knowledge_graph,
        maximum_recommendations=effective_settings.maximum_recommendations,
        minimum_score=effective_settings.minimum_recommendation_score,
    )
    recommend_dishes = RecommendDishesUseCase(
        classify_recipe=classify_recipe,
        cuisine_preference_extractor=cuisine_preference_extractor,
        dish_recommender=dish_recommender,
    )

    recipe_repository = CsvRecipeDocumentRepository(paths.recipes)
    recipe_retriever = Bm25RecipeRetriever(
        recipe_repository,
        k1=effective_settings.bm25_k1,
        b=effective_settings.bm25_b,
        title_weight=effective_settings.recipe_title_weight,
        cuisine_weight=effective_settings.recipe_cuisine_weight,
        ingredient_weight=effective_settings.recipe_ingredient_weight,
    )
    search_recipes = SearchRecipesUseCase(
        link_ingredients=link_ingredients,
        cuisine_preference_extractor=cuisine_preference_extractor,
        recipe_retriever=recipe_retriever,
        maximum_results=effective_settings.maximum_recipe_search_results,
    )

    return ApplicationContainer(
        knowledge_base=bundle,
        link_ingredients=link_ingredients,
        classify_recipe=classify_recipe,
        recommend_dishes=recommend_dishes,
        search_recipes=search_recipes,
    )
