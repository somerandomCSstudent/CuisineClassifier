from __future__ import annotations

from cuisine_assistant.application import (
    ClassifyRecipeUseCase,
    LinkIngredientsUseCase,
)
from cuisine_assistant.cli.classification_result_formatter import (
    ClassificationResultFormatter,
)
from cuisine_assistant.cli.link_result_formatter import (
    LinkIngredientsResultFormatter,
)


class ConsoleApplication:
    """Runs English ingredient-linking and cuisine-classification CLI workflows."""

    def __init__(
        self,
        link_ingredients: LinkIngredientsUseCase,
        classify_recipe: ClassifyRecipeUseCase,
        link_formatter: LinkIngredientsResultFormatter,
        classification_formatter: ClassificationResultFormatter,
    ) -> None:
        self._link_ingredients = link_ingredients
        self._classify_recipe = classify_recipe
        self._link_formatter = link_formatter
        self._classification_formatter = classification_formatter

    def run(self) -> None:
        print("Cuisine Assistant")
        print("1 - Link recipe ingredients to the knowledge graph")
        print("2 - Classify a recipe by cultural cuisine")
        print("3 - Exit\n")

        while True:
            choice = input("Select an option: ").strip()
            if choice == "1":
                self._run_linking()
            elif choice == "2":
                self._run_classification()
            elif choice in {"3", "exit", "quit"}:
                return
            else:
                print("Unknown option. Please choose 1, 2, or 3.\n")

    def _run_linking(self) -> None:
        recipe_text = input("Enter an English recipe or ingredient list:\n> ").strip()
        result = self._link_ingredients.execute(recipe_text)
        print(self._link_formatter.format(result))
        print()

    def _run_classification(self) -> None:
        recipe_text = input("Enter an English recipe or ingredient list:\n> ").strip()
        result = self._classify_recipe.execute(recipe_text)
        print(self._link_formatter.format(result.linking))
        print()
        print(self._classification_formatter.format(result))
        print()
