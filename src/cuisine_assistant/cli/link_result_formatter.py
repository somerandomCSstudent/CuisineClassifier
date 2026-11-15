from __future__ import annotations

from cuisine_assistant.application import LinkIngredientsResult


class LinkIngredientsResultFormatter:
    """Formats linked ingredient results for the English command-line interface."""

    def format(self, result: LinkIngredientsResult) -> str:
        if not result.linked_ingredients:
            return "No ingredients were detected."

        lines = ["Detected ingredients:"]
        for item in result.linked_ingredients:
            lines.extend(self._format_item(item))

        lines.extend(
            [
                "",
                "Summary:",
                f"- Linked to knowledge graph: {result.linked_count}",
                f"- Catalog only: {result.catalog_only_count}",
                f"- Unmatched: {result.unmatched_count}",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _format_item(item) -> list[str]:
        if item.ingredient is None:
            return [
                f"- Phrase: {item.extracted.surface_form}",
                "  Status: unmatched",
            ]

        graph_status = (
            "linked"
            if item.is_linked_to_knowledge_graph
            else "missing from knowledge graph"
        )
        ontology_id = item.ingredient.ontology_id
        rdf_types = (
            ", ".join(item.ontology_entity.rdf_types)
            if item.ontology_entity is not None
            and item.ontology_entity.rdf_types
            else "not available"
        )
        return [
            f"- Phrase: {item.extracted.surface_form}",
            f"  Canonical ingredient: {item.ingredient.canonical_name}",
            f"  Ingredient ID: {item.ingredient.ingredient_id}",
            f"  OWL entity: {ontology_id}",
            f"  RDF types: {rdf_types}",
            f"  Knowledge graph status: {graph_status}",
        ]
