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

# Nomi che il modello usa a volte al posto del nome reale di un parametro. Servono SOLO
# a rendere il messaggio d'errore più azionabile (NON a rimappare in silenzio: l'argomento
# va comunque inviato col nome esatto dello schema — così il modello capisce subito perché).
_ARG_ALIASES = {
    "path": {"file", "filename", "filepath", "file_path", "fname", "pathname", "filenames", "files"},
}


def _missing_args_message(tool_name: str, missing: list[str], unknown: list[str]) -> str:
    """Messaggio azionabile per argomenti obbligatori mancanti: nomina cosa manca, le
    eventuali chiavi ignorate, e — se una di esse somiglia a un argomento mancante (es.
    `filename` per `path`) — lo suggerisce. NON inizia con ❌ (lo antepone il chiamante)."""
    parts = [f"A «{tool_name}» mancano argomenti obbligatori: {', '.join(missing)}."]
    if unknown:
        parts.append(f"Ho ignorato chiavi non previste: {', '.join(unknown)}.")
        hints: list[str] = []
        for miss in missing:
            aliases = _ARG_ALIASES.get(miss, set())
            for unk in unknown:
                ul = unk.lower()
                if (ul in aliases or miss in ul or ul in miss) and f"«{unk}»→«{miss}»" not in hints:
                    hints.append(f"«{unk}»→«{miss}»")
        if hints:
            parts.append("Forse: " + ", ".join(hints) + "?")
    parts.append("Gli strumenti sono stateless: a ogni chiamata includi sempre TUTTI gli "
                 "argomenti obbligatori, coi nomi esatti dello schema.")
    return " ".join(parts)


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
    memory: Any = None      # SessionMemory della sessione (tool `remember`); None = non disponibile


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
    _required: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Precalcola parametri accettati (per scartare quelli inventati) e i loro tipi
        # dichiarati (per riallineare gli argomenti che il modello manda nel tipo
        # sbagliato — es. offset="2", ignore_case="true", extensions="mp3").
        sig = inspect.signature(self.func)
        self._var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        self._accepts = frozenset(sig.parameters)  # include 'ctx', innocuo
        props = (self.parameters or {}).get("properties", {})
        self._types = {k: v.get("type") for k, v in props.items() if isinstance(v, dict)}
        # Argomenti che la FUNZIONE richiede davvero: dichiarati 'required' nello schema,
        # presenti nella firma e SENZA default. Solo la loro assenza impedisce la chiamata
        # → errore azionabile, senza falsi positivi su argomenti che hanno già un default.
        req = (self.parameters or {}).get("required", []) or []
        self._required = tuple(
            r for r in req
            if r in sig.parameters and sig.parameters[r].default is inspect.Parameter.empty
        )

    def __call__(self, ctx: ToolContext, **kwargs) -> str:
        # 1) Coercizione dei tipi sugli argomenti previsti; 2) scarto tollerante di quelli
        # non previsti (un kwarg inventato non deve far fallire la chiamata: il tool gira e
        # si segnala cosa è stato ignorato); 3) se manca un argomento OBBLIGATORIO si solleva
        # un ToolError azionabile PRIMA di chiamare la funzione (vedi sotto). Se il tool
        # accetta **kwargs, ci fidiamo e passiamo tutto.
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
        # Argomento obbligatorio mancante (spesso il modello manda `filename` invece di
        # `path`, o lo omette nei contesti lunghi): errore azionabile che nomina cosa manca
        # e le chiavi scartate, invece di un TypeError grezzo che non dice quale chiave usare.
        missing = [r for r in self._required if r not in clean]
        if missing:
            raise ToolError(_missing_args_message(self.name, missing, unknown))
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

    def catalog(self) -> list[tuple[str, str]]:
        """(nome, descrizione) di ogni tool, in ordine — per elencarli nella UI."""
        return [(t.name, t.description) for t in self._tools.values()]
