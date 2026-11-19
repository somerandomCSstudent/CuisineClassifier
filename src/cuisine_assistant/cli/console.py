from __future__ import annotations

from cuisine_assistant.application import (
    AskCuisineQuestionUseCase,
    ClassifyRecipeUseCase,
    LinkIngredientsUseCase,
    RecommendDishesUseCase,
    SearchRecipesUseCase,
)
from cuisine_assistant.cli.classification_result_formatter import (
    ClassificationResultFormatter,
)
from cuisine_assistant.cli.link_result_formatter import (
    LinkIngredientsResultFormatter,
)
from cuisine_assistant.cli.recipe_search_result_formatter import (
    RecipeSearchResultFormatter,
)
from cuisine_assistant.cli.recommendation_result_formatter import (
    RecommendationResultFormatter,
)
from cuisine_assistant.cli.grounded_answer_formatter import (
    GroundedAnswerFormatter,
)
from cuisine_assistant.knowledge_base.models import KnowledgeBaseMetadata


class ConsoleApplication:
    """Runs CLI workflows backed by one selected knowledge base bundle."""

    def __init__(
        self,
        knowledge_base: KnowledgeBaseMetadata,
        link_ingredients: LinkIngredientsUseCase,
        classify_recipe: ClassifyRecipeUseCase,
        recommend_dishes: RecommendDishesUseCase,
        search_recipes: SearchRecipesUseCase,
        link_formatter: LinkIngredientsResultFormatter,
        classification_formatter: ClassificationResultFormatter,
        recommendation_formatter: RecommendationResultFormatter,
        recipe_search_formatter: RecipeSearchResultFormatter,
        ask_cuisine_question: AskCuisineQuestionUseCase,
        grounded_answer_formatter: GroundedAnswerFormatter,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._link_ingredients = link_ingredients
        self._classify_recipe = classify_recipe
        self._recommend_dishes = recommend_dishes
        self._search_recipes = search_recipes
        self._link_formatter = link_formatter
        self._classification_formatter = classification_formatter
        self._recommendation_formatter = recommendation_formatter
        self._recipe_search_formatter = recipe_search_formatter
        self._ask_cuisine_question = ask_cuisine_question
        self._grounded_answer_formatter = (
            grounded_answer_formatter
        )

    def run(self) -> None:
        print("\nCuisine Assistant")
        print(
            f"Active knowledge base: {self._knowledge_base.name} "
            f"({self._knowledge_base.version})"
        )
        print("1 - Link recipe ingredients to the knowledge graph")
        print("2 - Classify a recipe by cultural cuisine")
        print("3 - Recommend dishes")
        print("4 - Search recipe documents")
        print("5 - Ask a grounded cuisine question")
        print("6 - Exit\n")

        while True:
            choice = input("Select an option: ").strip()
            if choice == "1":
                self._run_linking()
            elif choice == "2":
                self._run_classification()
            elif choice == "3":
                self._run_recommendation()
            elif choice == "4":
                self._run_recipe_search()
            elif choice == "5":
                self._run_grounded_question_answering()
            elif choice in {"6", "exit", "quit"}:
                return
            else:
                print(
                    "Unknown option. "
                    "Please choose 1, 2, 3, 4, 5, or 6.\n"
                )

    def _run_linking(self) -> None:
        recipe_text = self._read_recipe_text()
        result = self._link_ingredients.execute(recipe_text)
        print(self._link_formatter.format(result))
        print()

    def _run_classification(self) -> None:
        recipe_text = self._read_recipe_text()
        result = self._classify_recipe.execute(recipe_text)
        print(self._link_formatter.format(result.linking))
        print()
        print(self._classification_formatter.format(result))
        print()

    def _run_recommendation(self) -> None:
        user_text = self._read_recipe_text()
        result = self._recommend_dishes.execute(user_text)
        print(self._link_formatter.format(result.classification.linking))
        print()
        print(self._classification_formatter.format(result.classification))
        print()
        print(self._recommendation_formatter.format(result))
        print()

    def _run_recipe_search(self) -> None:
        query_text = input("Enter an English recipe search query:\n> ").strip()
        result = self._search_recipes.execute(query_text)
        print(self._link_formatter.format(result.linking))
        print()
        print(self._recipe_search_formatter.format(result))
        print()
        
    def _run_grounded_question_answering(self) -> None:
        """Answers one English cuisine question from bundle evidence."""
        question = input(
            "Enter an English cuisine question:\n> "
        ).strip()

        result = self._ask_cuisine_question.execute(
            question
        )

        print(
            self._grounded_answer_formatter.format(
                result
            )
        )
        print()

    @staticmethod
    def _read_recipe_text() -> str:
        return input("Enter an English recipe or ingredient list:\n> ").strip()
