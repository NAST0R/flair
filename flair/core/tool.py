"""Astrazione dei tool.

Ogni tool tiene insieme, in un solo posto, la sua funzione e il suo schema
JSON. Questo elimina il drift tra "dispatch" e "schemi" che affliggeva il
vecchio progetto (due liste parallele da tenere sincronizzate a mano).

Un `Tool` è creato dal decoratore `@tool(...)`; resta richiamabile come una
normale funzione (`mytool(ctx, **args)`), ma espone anche `.schema()` e il
flag `.destructive`. Un `Toolset` raccoglie i tool e fornisce schemi e dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """Errore atteso di un tool (input non valido, vincolo violato).

    A differenza di un'eccezione generica, viene riportato all'LLM come
    messaggio pulito senza traceback: è informazione, non un crash.
    """


@dataclass
class ToolContext:
    """Stato condiviso passato a ogni tool. Estendibile senza toccare le firme."""
    cfg: Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema dei parametri
    func: Callable[..., str]
    destructive: bool = False

    def __call__(self, ctx: ToolContext, **kwargs) -> str:
        return self.func(ctx, **kwargs)

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
