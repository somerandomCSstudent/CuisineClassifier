# CuisineClassifier

Project for VIMIAC22 course

## Run

```bash
python -m venv .venv
# activate the environment
pip install -e ".[dev]"
pytest
python -m cuisine_assistant.main
```

Example input:

```text
Fresh tomatoes, mozzarella and basil with extra virgin olive oil
```

The application can later receive `SpacyIngredientExtractor`, `HybridEntitySearch`, or an embedding search implementation without changing the use case or CLI
