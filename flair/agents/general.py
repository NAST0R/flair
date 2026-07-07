"""Costruzione dell'agente generico (automazione desktop + conversazione + web)."""

from __future__ import annotations

from .. import prompts
from ..core.agent import Agent
from ..core.tool import Toolset
from ..tools import memory as memory_tools
from ..tools import system as system_tools
from ..tools import web as web_tools


def build(cfg, provider, conversation=None, **callbacks) -> Agent:
    tools = system_tools.TOOLS + web_tools.TOOLS
    if getattr(cfg, "memory_enabled", True):
        # Stessa memoria di sessione dell'agente coding (condivisa via ToolContext).
        tools = [*tools, memory_tools.remember]
    if getattr(cfg, "read_only", False):
        # Esecuzione non presidiata: niente write/edit/comandi sull'intera macchina.
        tools = [t for t in tools if not t.destructive]
    return Agent(
        name="general",
        cfg=cfg,
        provider=provider,
        toolset=Toolset(tools),
        system_prompt=prompts.load("general"),
        conversation=conversation,
        **callbacks,
    )
