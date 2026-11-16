from __future__ import annotations

from cuisine_assistant.application import (
    ClassifyRecipeUseCase,
    LinkIngredientsUseCase,
    RecommendDishesUseCase,
)
from cuisine_assistant.cli.classification_result_formatter import (
    ClassificationResultFormatter,
)
from cuisine_assistant.cli.link_result_formatter import (
    LinkIngredientsResultFormatter,
)
from cuisine_assistant.cli.recommendation_result_formatter import (
    RecommendationResultFormatter,
)


class ConsoleApplication:
    """Runs English linking, classification, and recommendation CLI workflows."""

    def __init__(
        self,
        link_ingredients: LinkIngredientsUseCase,
        classify_recipe: ClassifyRecipeUseCase,
        recommend_dishes: RecommendDishesUseCase,
        link_formatter: LinkIngredientsResultFormatter,
        classification_formatter: ClassificationResultFormatter,
        recommendation_formatter: RecommendationResultFormatter,
    ) -> None:
        self._link_ingredients = link_ingredients
        self._classify_recipe = classify_recipe
        self._recommend_dishes = recommend_dishes
        self._link_formatter = link_formatter
        self._classification_formatter = classification_formatter
        self._recommendation_formatter = recommendation_formatter

    def run(self) -> None:
        print("Cuisine Assistant")
        print("1 - Link recipe ingredients to the knowledge graph")
        print("2 - Classify a recipe by cultural cuisine")
        print("3 - Recommend dishes")
        print("4 - Exit\n")

        while True:
            choice = input("Select an option: ").strip()
            if choice == "1":
                self._run_linking()
            elif choice == "2":
                self._run_classification()
            elif choice == "3":
                self._run_recommendation()
            elif choice in {"4", "exit", "quit"}:
                return
            else:
                print("Unknown option. Please choose 1, 2, 3, or 4.\n")

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

    @staticmethod
    def _read_recipe_text() -> str:
        return input("Enter an English recipe or ingredient list:\n> ").strip()
