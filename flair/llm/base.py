"""Base del layer LLM.

Entrambi i provider (DeepSeek e OpenAI) parlano il protocollo OpenAI Chat
Completions, quindi la logica vera vive qui in `OpenAICompatProvider`. Le
sottoclassi configurano solo le differenze (parametro per i token, reasoning
model, endpoint).

Principi:
- I messaggi sono dict in formato OpenAI e NON vengono mai mutati dal provider:
  è l'agente a garantire la crescita append-only (cache del prefisso).
- Il `reasoning_content` (DeepSeek) è restituito a parte e NON va re-inviato.
- Parsing robusto degli argomenti delle tool call (path Windows, doppia codifica).
- Retry con backoff SOLO sugli errori transitori; i 4xx (es. richiesta troppo
  lunga) vengono rilanciati subito perché ritentarli è inutile.
- Streaming opzionale: i delta vengono assemblati nella stessa `LLMResponse`.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

try:  # presente in openai>=1.x
    from openai import BadRequestError
except Exception:  # pragma: no cover
    BadRequestError = Exception  # type: ignore

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF = (1.0, 2.0, 4.0)
_TRANSIENT = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)

_OVERFLOW_RX = re.compile(
    r"context length|maximum context|context window|too many tokens|"
    r"reduce the (?:length|number)|context_length_exceeded|exceeds the maximum",
    re.IGNORECASE,
)


def is_context_overflow(exc: Exception) -> bool:
    """True se l'eccezione indica un contesto troppo lungo per il modello."""
    return isinstance(exc, BadRequestError) and bool(_OVERFLOW_RX.search(str(exc)))


def parse_tool_args(raw: Any) -> dict:
    """Parsa in modo robusto gli argomenti JSON di una tool call."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}

    # 1. JSON ben formato (il caso normale): non si tocca nulla.
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        if isinstance(result, str):  # doppia codifica
            inner = json.loads(result)
            if isinstance(inner, dict):
                return inner
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. JSON malformato: causa dominante = path Windows con backslash singoli.
    #    Convertire tutti i backslash in slash è il recupero più sicuro.
    try:
        result = json.loads(raw.replace("\\", "/"))
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Ultima risorsa: neutralizza solo i backslash che non formano escape validi.
    valid_after = set('"\\/bfnrtu')
    chars = list(raw)
    for i, c in enumerate(chars):
        if c == "\\":
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt not in valid_after:
                chars[i] = "/"
    try:
        result = json.loads("".join(chars))
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    log.warning("Argomenti tool call non parsabili: %.120s", raw)
    return {"_raw": raw}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
            self.cache_hit_tokens + other.cache_hit_tokens,
            self.cache_miss_tokens + other.cache_miss_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class LLMResponse:
    content: str = ""
    reasoning: str = ""          # CoT del reasoning model — NON re-inviare
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


OnDelta = Callable[[str], None]


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
        max_tokens: int | None = None,
        stream: bool = False,
        on_delta: OnDelta | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def estimate_cost(self, usage: Usage, cfg) -> float:
        miss = usage.cache_miss_tokens or usage.prompt_tokens
        return (
            usage.cache_hit_tokens / 1_000_000 * cfg.price_cache_hit
            + miss / 1_000_000 * cfg.price_cache_miss
            + usage.completion_tokens / 1_000_000 * cfg.price_output
        )


class OpenAICompatProvider(LLMProvider):
    """Provider per qualsiasi backend compatibile OpenAI Chat Completions."""

    token_param: str = "max_tokens"          # OpenAI usa "max_completion_tokens"
    reasoning_regex: re.Pattern | None = None  # match sul nome → reasoning model
    supports_reasoning_effort: bool = False    # solo OpenAI accetta reasoning_effort

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        pc = cfg.active
        self._client = OpenAI(api_key=pc.api_key, base_url=pc.base_url, timeout=cfg.request_timeout)

    def is_reasoning_model(self, model: str) -> bool:
        return bool(self.reasoning_regex and self.reasoning_regex.search(model))

    def _build_params(self, messages, tools, think, max_tokens) -> dict[str, Any]:
        pc = self.cfg.active
        model = pc.think_model if think else pc.model
        params: dict[str, Any] = {"model": model, "messages": messages}
        params[self.token_param] = max_tokens if max_tokens is not None else self.cfg.max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        self._apply_reasoning(params, model, think)
        return params

    def _apply_reasoning(self, params: dict, model: str, think: bool) -> None:
        """Imposta i parametri legati a ragionamento/temperatura. Comportamento di
        default (OpenAI-style); DeepSeek lo sovrascrive (thinking via parametro)."""
        pc = self.cfg.active
        if self.is_reasoning_model(model):
            # I reasoning model rifiutano `temperature`. `reasoning_effort` si invia
            # solo dove è supportato (OpenAI); con `--think` si imposta un default
            # 'medium' se non specificato, perché alcuni modelli (gpt-5.1+) di default
            # NON ragionano a meno di riceverlo esplicitamente.
            if self.supports_reasoning_effort:
                effort = pc.reasoning_effort or ("medium" if think else None)
                if effort:
                    params["reasoning_effort"] = effort
        else:
            params["temperature"] = pc.temperature

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
        max_tokens: int | None = None,
        stream: bool = False,
        on_delta: OnDelta | None = None,
    ) -> LLMResponse:
        params = self._build_params(messages, tools, think, max_tokens)

        if stream and on_delta is not None:
            try:
                return self._complete_stream(params, on_delta)
            except _TRANSIENT as exc:
                log.warning("Streaming fallito (%s) — fallback senza streaming", type(exc).__name__)
            # i non-transitori (es. BadRequest) escono e li gestisce il chiamante

        return self._normalize(self._call_with_retry(params))

    # ── interni ───────────────────────────────────────────────────────────

    def _call_with_retry(self, params: dict):
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.chat.completions.create(**params)
            except _TRANSIENT as exc:
                last = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF[attempt] + random.uniform(0, 0.5)
                    log.warning("Chiamata LLM transitoria fallita (%d/%d): %s — retry tra %.1fs",
                                attempt + 1, _MAX_RETRIES, type(exc).__name__, wait)
                    time.sleep(wait)
            except Exception:
                raise  # non-transitorio: inutile ritentare
        assert last is not None
        raise last

    def _complete_stream(self, params: dict, on_delta: OnDelta) -> LLMResponse:
        params = {**params, "stream": True, "stream_options": {"include_usage": True}}
        stream = self._client.chat.completions.create(**params)

        content: list[str] = []
        reasoning: list[str] = []
        acc: dict[int, dict] = {}
        usage_obj = None

        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                content.append(piece)
                on_delta(piece)
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning.append(rc)
            for tcd in (getattr(delta, "tool_calls", None) or []):
                slot = acc.setdefault(tcd.index, {"id": None, "name": None, "args": ""})
                if getattr(tcd, "id", None):
                    slot["id"] = tcd.id
                fn = getattr(tcd, "function", None)
                if fn:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments

        tool_calls = [
            ToolCall(id=s["id"] or f"call_{i}", name=s["name"], arguments=parse_tool_args(s["args"]))
            for i, s in sorted(acc.items()) if s["name"]
        ]
        return LLMResponse(content="".join(content), reasoning="".join(reasoning),
                           tool_calls=tool_calls, usage=self._usage(usage_obj))

    def _normalize(self, resp) -> LLMResponse:
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tool_calls.append(ToolCall(id=tc.id, name=fn.name, arguments=parse_tool_args(fn.arguments)))
        return LLMResponse(
            content=msg.content or "",
            reasoning=getattr(msg, "reasoning_content", "") or "",
            tool_calls=tool_calls,
            usage=self._usage(resp.usage),
        )

    @staticmethod
    def _usage(u) -> Usage:
        if not u:
            return Usage()
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )
        hit = getattr(u, "prompt_cache_hit_tokens", None)
        miss = getattr(u, "prompt_cache_miss_tokens", None)
        if hit is not None or miss is not None:
            usage.cache_hit_tokens = hit or 0
            usage.cache_miss_tokens = miss or 0
        else:
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            usage.cache_hit_tokens = cached or 0
            usage.cache_miss_tokens = (usage.prompt_tokens or 0) - (cached or 0)
        cdet = getattr(u, "completion_tokens_details", None)
        if cdet is not None:
            usage.reasoning_tokens = getattr(cdet, "reasoning_tokens", 0) or 0
        return usage
