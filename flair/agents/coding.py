"""Costruzione dell'agente di coding."""

from __future__ import annotations

from .. import prompts
from ..core.agent import Agent
from ..core.tool import Toolset
from ..tools import coding as coding_tools
from ..tools import web as web_tools


def build(cfg, provider, conversation=None, **callbacks) -> Agent:
    system_prompt = prompts.load("coding") + prompts.project_instructions(cfg.root)
    return Agent(
        name="coding",
        cfg=cfg,
        provider=provider,
        # Tool del progetto + ricerca web (sola lettura): utile quando servono
        # informazioni reperibili online (doc di librerie, API, messaggi d'errore).
        toolset=Toolset(coding_tools.TOOLS + web_tools.TOOLS),
        system_prompt=system_prompt,
        conversation=conversation,
        **callbacks,
    )
