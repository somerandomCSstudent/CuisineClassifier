from __future__ import annotations

import argparse
from pathlib import Path

from cuisine_assistant.knowledge_base import (
    HybridKnowledgeBaseBuilder,
    JsonKnowledgeBaseManifestLoader,
)


def main() -> None:
    """Builds the hybrid bundle from curated and imported local bundles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ontology-base-iri", default="urn:cuisine:hybrid#")
    parser.add_argument("--version", default="1.0")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    loader = JsonKnowledgeBaseManifestLoader()
    curated = loader.load(root / "knowledge_bases/curated/knowledge_base.json")
    imported = loader.load(root / "knowledge_bases/world_cuisines/knowledge_base.json")
    destination = root / "knowledge_bases/hybrid"
    HybridKnowledgeBaseBuilder().build(
        curated,
        imported,
        destination,
        ontology_base_iri=arguments.ontology_base_iri,
        version=arguments.version,
    )
    print(f"Hybrid knowledge base created: {destination}")


if __name__ == "__main__":
    main()
