"""Provider DeepSeek: API ufficiale E host terzi che servono gli stessi pesi.

Il profilo si deduce dall'ENDPOINT — nessun flag da ricordare (override
esplicito: FLAIR_DEEPSEEK_FIRST_PARTY=true|false, default auto):

- FIRST-PARTY (api.deepseek.com): protocollo completo. Il reasoning dei turni
  con tool call torna all'API (obbligatorio dal protocollo V4); la modalità
  thinking dei modelli V4 si attiva con un PARAMETRO, non col nome:
  `extra_body={"thinking": {"type": "enabled"}}` con --think, più un eventuale
  `reasoning_effort` (high|max) da DEEPSEEK_REASONING_EFFORT. NB: sull'API V4
  il thinking è ABILITATO DI DEFAULT anche senza parametro: il fast mode di
  flair si affida deliberatamente a quel default. Listino a fasce orarie.
- TERZI (OpenRouter, DeepInfra, ...): stessi pesi, API OpenAI-compatibile
  STANDARD — le estensioni proprietarie non partono: reasoning spogliato a
  request-time (dalla base, su copie), niente extra_body (il thinking segue i
  default dell'host; --think resta lo switch di modello), listino flat (gli
  override FLAIR_PRICE_* portano il listino dell'host). Gli slug vendor
  ('deepseek-ai/DeepSeek-V4-Pro-0813') sono normalizzati per thinking e prezzi.

`temperature` è un parametro STANDARD e viene sempre inviata in entrambi i
profili (DeepSeek la accetta anche in thinking mode, dove la ignora). Cintura
di sicurezza per endpoint sconosciuti: se il passback del reasoning viene
rigettato (400), si passa al profilo compat e si ritenta UNA volta.

(Gli alias legacy deepseek-chat/deepseek-reasoner portano la modalità nel nome
e verranno ritirati il 2026-07-24.) Cache del prefisso e strip del
`reasoning_content` sono gestiti dalla base.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from openai import BadRequestError

from ..config import _model_key, deepseek_first_party
from .base import LLMResponse, OpenAICompatProvider

log = logging.getLogger("flair.llm.deepseek")


class DeepSeekProvider(OpenAICompatProvider):
    name = "deepseek"
    token_param = "max_tokens"
    # Utile solo per gli alias legacy (deepseek-reasoner). Coi nomi V4 la modalità
    # thinking è controllata dal parametro, non dal nome.
    reasoning_regex = re.compile(r"reasoner", re.IGNORECASE)

    # Protocollo thinking V4 (solo first-party): il reasoning dei turni con tool
    # call torna all'API. Ridefinito per-istanza nel costruttore.
    keeps_reasoning_history = True

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Profilo dedotto dall'endpoint: first-party = protocollo completo;
        # qualsiasi altro host = standard-compat. Attributi d'ISTANZA: lo stesso
        # processo può alternare endpoint diversi con /provider senza residui.
        self.first_party = deepseek_first_party(cfg.active.base_url)
        self.keeps_reasoning_history = self.first_party
        self.banded_pricing = self.first_party

    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 think: bool = False, max_tokens: int | None = None, stream: bool = False,
                 on_delta: Callable[[str], None] | None = None,
                 on_reasoning: Callable[[str], None] | None = None,
                 on_reasoning_delta: Callable[[str], None] | None = None,
                 tool_choice: str | None = None) -> LLMResponse:
        try:
            return super().complete(messages, tools=tools, think=think, max_tokens=max_tokens,
                                    stream=stream, on_delta=on_delta, on_reasoning=on_reasoning,
                                    on_reasoning_delta=on_reasoning_delta, tool_choice=tool_choice)
        except BadRequestError as exc:
            # Cintura per endpoint sconosciuti (es. first-party FORZATO via env su
            # un host che poi rigetta il passback): si degrada al profilo compat e
            # si ritenta una volta — lo strip avviene in _build_params, su copie.
            # Da lì in poi il profilo resta compat: niente loop di tentativi.
            if self.keeps_reasoning_history and "reasoning" in str(exc).lower():
                log.warning("Endpoint rejected the reasoning passback: retrying with the compat profile.")
                self.keeps_reasoning_history = False
                return super().complete(messages, tools=tools, think=think, max_tokens=max_tokens,
                                        stream=stream, on_delta=on_delta, on_reasoning=on_reasoning,
                                        on_reasoning_delta=on_reasoning_delta, tool_choice=tool_choice)
            raise

    def _apply_reasoning(self, params, model: str, think: bool) -> None:
        # Parametro standard: in ENTRAMBI i profili (DeepSeek la accetta sempre).
        params["temperature"] = self.cfg.active.temperature
        if not self.first_party:
            # Host terzi: API standard, niente estensioni proprietarie. Il thinking
            # segue i default dell'host; --think resta lo switch di modello.
            return
        # V4 (deepseek-v4-flash / deepseek-v4-pro): thinking via parametro. Il nome
        # è normalizzato (slug vendor con FLAIR_DEEPSEEK_FIRST_PARTY=true). NB: lato
        # API il thinking è attivo di default anche senza parametro; qui lo rendiamo
        # esplicito con --think e, se configurato, regoliamo l'effort (high|max —
        # l'API mappa low/medium→high e xhigh→max per compatibilità).
        if _model_key(model).startswith("deepseek-v4"):
            if think:
                params["extra_body"] = {"thinking": {"type": "enabled"}}
                if self.cfg.active.reasoning_effort:
                    params["reasoning_effort"] = self.cfg.active.reasoning_effort
            elif self.cfg.active.fast_reasoning_effort:
                # Opt-in "via di mezzo": il flash pensa già di default (effort high
                # lato server); qui alziamo la profondità del loop veloce senza
                # cambiare modello. Variabile non impostata = richiesta identica
                # a prima (nessun parametro: default server intatto).
                params["extra_body"] = {"thinking": {"type": "enabled"}}
                params["reasoning_effort"] = self.cfg.active.fast_reasoning_effort
        # Alias legacy: la modalità è già nel nome del modello, niente da aggiungere.
