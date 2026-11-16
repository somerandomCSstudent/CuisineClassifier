from __future__ import annotations

import argparse
from pathlib import Path

from cuisine_assistant.knowledge_base import (
    FileKnowledgeBaseValidator,
    JsonKnowledgeBaseManifestLoader,
    KnowledgeBaseConsistencyValidator,
)


def main() -> None:
    """Validates bundle completeness and CSV-RDF cross-file consistency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    arguments = parser.parse_args()
    bundle = JsonKnowledgeBaseManifestLoader().load(
        arguments.bundle_root / "knowledge_base.json"
    )
    completeness = FileKnowledgeBaseValidator().validate(bundle)
    if not completeness.is_valid:
        for path in completeness.missing_files:
            print(f"MISSING FILE: {path}")
        raise SystemExit(1)
    report = KnowledgeBaseConsistencyValidator().validate(bundle)
    if not report.is_valid:
        for issue in report.issues:
            location = issue.file_name
            if issue.row_number is not None:
                location += f":{issue.row_number}"
            print(f"{issue.code} | {location} | {issue.message}")
        raise SystemExit(1)
    print(f"Knowledge base is consistent: {report.knowledge_base_id}")


if __name__ == "__main__":
    main()
