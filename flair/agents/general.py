"""Costruzione dell'agente generico (automazione desktop + conversazione + web)."""

from __future__ import annotations

from .. import prompts
from ..core.agent import Agent
from ..core.tool import Toolset
from ..tools import system as system_tools
from ..tools import web as web_tools


def build(cfg, provider, **callbacks) -> Agent:
    return Agent(
        name="general",
        cfg=cfg,
        provider=provider,
        toolset=Toolset(system_tools.TOOLS + web_tools.TOOLS),
        system_prompt=prompts.load("general"),
        **callbacks,
    )
