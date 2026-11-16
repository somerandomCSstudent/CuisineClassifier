from __future__ import annotations

from cuisine_assistant.application import (
    KnowledgeBaseSummary,
    SelectKnowledgeBaseResult,
    SelectKnowledgeBaseUseCase,
)


class KnowledgeBaseSelector:
    """Prompts the user to choose one complete local knowledge base bundle."""

    def __init__(self, use_case: SelectKnowledgeBaseUseCase) -> None:
        self._use_case = use_case

    def select(self) -> SelectKnowledgeBaseResult:
        """Displays available bundles and returns the validated user selection."""
        summaries = self._use_case.list_available()
        if not summaries:
            raise RuntimeError(
                "No valid knowledge base bundle is available in knowledge_bases/."
            )

        print("Available knowledge bases:")
        for index, summary in enumerate(summaries, start=1):
            print(self._format_summary(index, summary))

        while True:
            choice = input("Select a knowledge base: ").strip()
            selected_id = self._resolve_choice(choice, summaries)
            if selected_id is None:
                print(
                    f"Unknown selection. Enter a number from 1 to "
                    f"{len(summaries)} or a knowledge base ID.\n"
                )
                continue
            return self._use_case.execute(selected_id)

    @staticmethod
    def _resolve_choice(
        choice: str,
        summaries: tuple[KnowledgeBaseSummary, ...],
    ) -> str | None:
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(summaries):
                return summaries[index - 1].knowledge_base_id
            return None

        normalized = choice.strip()
        return next(
            (
                item.knowledge_base_id
                for item in summaries
                if item.knowledge_base_id == normalized
            ),
            None,
        )

    @staticmethod
    def _format_summary(
        index: int,
        summary: KnowledgeBaseSummary,
    ) -> str:
        return (
            f"{index} - {summary.name} [{summary.knowledge_base_id}]\n"
            f"    Version: {summary.version} | Source: {summary.source} | "
            f"Language: {summary.language}\n"
            f"    Dishes: {summary.dish_count} | "
            f"Ingredients: {summary.ingredient_count} | "
            f"Cuisines: {summary.cuisine_count} | "
            f"RDF triples: {summary.triple_count}"
        )
