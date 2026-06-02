"""Provider OpenAI.

Differenze rispetto alla base:
- usa `max_completion_tokens` (richiesto dai reasoning model, accettato dagli altri);
- rileva i reasoning model dal nome — sia la serie o (o1/o3/o4...) sia la
  famiglia GPT-5 (gpt-5, gpt-5-mini, gpt-5.1, ...) — per omettere `temperature`
  (che quei modelli rifiutano) e impostare `reasoning_effort`;
- supporta `reasoning_effort` (default 'medium' con --think).
La cache del prompt OpenAI (`prompt_tokens_details.cached_tokens`) è già
normalizzata dalla base.

Nota: per i modelli gpt-5.4 e successivi la combinazione reasoning + tool è
disponibile solo sulla Responses API, non sulla Chat Completions usata qui;
come think model conviene quindi restare su gpt-5 / gpt-5-mini / gpt-5.1 o o3.
"""

from __future__ import annotations

import re

from .base import OpenAICompatProvider


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    token_param = "max_completion_tokens"
    supports_reasoning_effort = True
    # Serie o (o1/o3/o4-mini, o3-pro…) e famiglia GPT-5 (gpt-5, gpt-5-mini, gpt-5.x).
    # Non fa match con gpt-4o (la 'o' è dopo '4') né con gpt-4.1.
    reasoning_regex = re.compile(r"^(?:o\d|gpt-5)", re.IGNORECASE)
