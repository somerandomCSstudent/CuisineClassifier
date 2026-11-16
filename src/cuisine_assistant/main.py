from __future__ import annotations

import os
from pathlib import Path

from cuisine_assistant.bootstrap import (
    ApplicationSettings,
    build_application_container,
)
from cuisine_assistant.cli import (
    ClassificationResultFormatter,
    ConsoleApplication,
    LinkIngredientsResultFormatter,
)


def main() -> None:
    """Builds the application and starts the English command-line interface."""
    project_root = Path(__file__).resolve().parents[2]
    ontology_base_iri = os.environ.get(
        "CUISINE_ONTOLOGY_BASE_IRI",
        "http://example.org/cuisine#",
    )
    container = build_application_container(
        ApplicationSettings(
            project_root=project_root,
            ontology_base_iri=ontology_base_iri,
        )
    )
    ConsoleApplication(
        link_ingredients=container.link_ingredients,
        classify_recipe=container.classify_recipe,
        link_formatter=LinkIngredientsResultFormatter(),
        classification_formatter=ClassificationResultFormatter(),
    ).run()


if __name__ == "__main__":
    main()
