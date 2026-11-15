from __future__ import annotations

import re

from cuisine_assistant.domain.ports import TextNormalizer


class EnglishTextNormalizer(TextNormalizer):
    def normalize(self, text: str) -> str:
        text = text.lower().replace("’", "'")
        text = re.sub(r"[^a-z0-9\s'-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
