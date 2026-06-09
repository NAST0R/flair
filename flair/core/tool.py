"""Astrazione dei tool.

Ogni tool tiene insieme, in un solo posto, la sua funzione e il suo schema
JSON. Questo elimina il drift tra "dispatch" e "schemi" che affliggeva il
vecchio progetto (due liste parallele da tenere sincronizzate a mano).

Un `Tool` è creato dal decoratore `@tool(...)`; resta richiamabile come una
normale funzione (`mytool(ctx, **args)`), ma espone anche `.schema()` e il
flag `.destructive`. Un `Toolset` raccoglie i tool e fornisce schemi e dispatch.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_TRUE_STRINGS = {"1", "true", "yes", "si", "sì", "y", "vero", "on"}
_INT_RX = re.compile(r"[+-]?\d+$")


def _coerce(value: Any, schema_type: str | None) -> Any:
    """Riallinea un argomento al tipo dichiarato dallo schema quando il modello invia
    un tipo diverso (tipicamente stringhe al posto di int/bool/array). Lascia intatto
    ciò che è già corretto o non coercibile in modo sicuro: non si rompe mai un valore
    valido, e i tipi 'string' non vengono toccati."""
    if schema_type == "integer":
        if isinstance(value, bool):       # in Python bool è int: non trasformarlo
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and _INT_RX.match(value.strip()):
            return int(value.strip())
        return value
    if schema_type == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value
        return value
    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in _TRUE_STRINGS
        return bool(value)
    if schema_type == "array":
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return parsed
                except ValueError:
                    pass
            return [value]                # singolo elemento → lista
        return value
    return value


class ToolError(Exception):
    """Errore atteso di un tool (input non valido, vincolo violato).

    A differenza di un'eccezione generica, viene riportato all'LLM come
    messaggio pulito senza traceback: è informazione, non un crash.
    """


@dataclass
class ToolContext:
    """Stato condiviso passato a ogni tool. Estendibile senza toccare le firme."""
    cfg: Any
    provider: Any = None    # per i tool che delegano a un sub-agente (es. explore)
    delegated_usage: Any = None  # usage riportato dai tool che delegano; l'agente lo somma a turno+sessione


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema dei parametri
    func: Callable[..., str]
    destructive: bool = False
    _accepts: frozenset = field(init=False, repr=False, compare=False)
    _var_kw: bool = field(init=False, repr=False, compare=False)
    _types: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Precalcola parametri accettati (per scartare quelli inventati) e i loro tipi
        # dichiarati (per riallineare gli argomenti che il modello manda nel tipo
        # sbagliato — es. offset="2", ignore_case="true", extensions="mp3").
        sig = inspect.signature(self.func)
        self._var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        self._accepts = frozenset(sig.parameters)  # include 'ctx', innocuo
        props = (self.parameters or {}).get("properties", {})
        self._types = {k: v.get("type") for k, v in props.items() if isinstance(v, dict)}

    def __call__(self, ctx: ToolContext, **kwargs) -> str:
        # 1) Coercizione dei tipi sugli argomenti previsti; 2) scarto tollerante di
        # quelli non previsti (un kwarg inventato non deve far fallire la chiamata: il
        # tool gira e si segnala cosa è stato ignorato). NB: un argomento OBBLIGATORIO
        # mancante dà comunque errore a valle (TypeError), quindi i refusi seri restano
        # visibili. Se il tool accetta **kwargs, ci fidiamo e passiamo tutto.
        if self._var_kw:
            return self.func(ctx, **kwargs)
        clean: dict[str, Any] = {}
        unknown: list[str] = []
        for k, v in kwargs.items():
            if k not in self._accepts:
                unknown.append(k)
                continue
            t = self._types.get(k)
            clean[k] = _coerce(v, t) if t else v
        out = self.func(ctx, **clean)
        if unknown:
            out = f"ℹ️ Argomenti ignorati (non previsti da {self.name}): {', '.join(unknown)}.\n" + out
        return out

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(name: str, description: str, parameters: dict, destructive: bool = False):
    """Decoratore che trasforma una funzione `(ctx, **args) -> str` in un Tool."""
    def deco(fn: Callable[..., str]) -> Tool:
        return Tool(name=name, description=description, parameters=parameters,
                    func=fn, destructive=destructive)
    return deco


class Toolset:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def catalog(self) -> list[tuple[str, str]]:
        """(nome, descrizione) di ogni tool, in ordine — per elencarli nella UI."""
        return [(t.name, t.description) for t in self._tools.values()]
