from __future__ import annotations

import json
from dataclasses import asdict

from cuisine_assistant.application.extract_ingredients import ExtractIngredientsUseCase


class ConsoleApplication:
    def __init__(self, use_case: ExtractIngredientsUseCase) -> None:
        self._use_case = use_case

    def run(self) -> None:
        print("Cuisine Assistant - ingredient extraction baseline")
        print("Enter an English recipe or ingredient list. Type 'exit' to quit.\n")
        while True:
            text = input("> ").strip()
            if text.lower() in {"exit", "quit"}:
                return
            result = self._use_case.execute(text)
            print(json.dumps([asdict(item) for item in result], indent=2))
            print()
