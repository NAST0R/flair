"""Provider DeepSeek (compatibile OpenAI).

Differenze rispetto alla base:
- usa `max_tokens` (default della base);
- la modalità thinking dei modelli V4 si attiva con un PARAMETRO, non col nome:
  `extra_body={"thinking": {"type": "enabled"}}` quando si usa --think, più un
  eventuale `reasoning_effort` (high|max) da DEEPSEEK_REASONING_EFFORT.
  (Gli alias legacy deepseek-chat/deepseek-reasoner portano invece la modalità
  nel nome e verranno ritirati il 2026-07-24.)
- NB: sull'API V4 il thinking è ABILITATO DI DEFAULT anche senza parametro: il
  fast mode di flair si affida deliberatamente a quel default (tracce brevi,
  qualità buona sul campo). L'effort esplicito è una proprietà del --think.
- DeepSeek accetta `temperature` anche in thinking mode (in thinking la ignora),
  quindi viene sempre inviata — a differenza dei reasoning model OpenAI.

La cache del prefisso e il `reasoning_content` sono gestiti dalla base.
"""

from __future__ import annotations

import re

from .base import OpenAICompatProvider


class DeepSeekProvider(OpenAICompatProvider):
    name = "deepseek"
    token_param = "max_tokens"
    # Utile solo per gli alias legacy (deepseek-reasoner). Coi nomi V4 la modalità
    # thinking è controllata dal parametro, non dal nome.
    reasoning_regex = re.compile(r"reasoner", re.IGNORECASE)

    def _apply_reasoning(self, params, model: str, think: bool) -> None:
        params["temperature"] = self.cfg.active.temperature
        # V4 (deepseek-v4-flash / deepseek-v4-pro): thinking via parametro. NB: lato
        # API il thinking è attivo di default anche senza parametro; qui lo rendiamo
        # esplicito con --think e, se configurato, regoliamo l'effort (high|max —
        # l'API mappa low/medium→high e xhigh→max per compatibilità).
        if think and model.startswith("deepseek-v4"):
            params["extra_body"] = {"thinking": {"type": "enabled"}}
            effort = self.cfg.active.reasoning_effort
            if effort:
                params["reasoning_effort"] = effort
        # Alias legacy: la modalità è già nel nome del modello, niente da aggiungere.
