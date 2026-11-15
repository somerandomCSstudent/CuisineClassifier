from __future__ import annotations

from cuisine_assistant.application import LinkIngredientsUseCase
from cuisine_assistant.cli.link_result_formatter import (
    LinkIngredientsResultFormatter,
)


class ConsoleApplication:
    """Runs the English interactive command-line ingredient-linking workflow."""

    def __init__(
        self,
        link_ingredients: LinkIngredientsUseCase,
        formatter: LinkIngredientsResultFormatter,
    ) -> None:
        self._link_ingredients = link_ingredients
        self._formatter = formatter

    def run(self) -> None:
        print("Cuisine Assistant - ingredient knowledge graph linking")
        print("Enter an English recipe or ingredient list.")
        print("Type 'exit' to quit.\n")

        while True:
            recipe_text = input("> ").strip()
            if recipe_text.lower() in {"exit", "quit"}:
                return
            result = self._link_ingredients.execute(recipe_text)
            print(self._formatter.format(result))
            print()
