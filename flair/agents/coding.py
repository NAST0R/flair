"""Costruzione dell'agente di coding."""

from __future__ import annotations

from .. import prompts
from ..core.agent import Agent
from ..core.tool import Toolset
from ..tools import coding as coding_tools
from ..tools import plan as plan_tools
from ..tools import subagent as subagent_tools
from ..tools import web as web_tools


def build(cfg, provider, conversation=None, **callbacks) -> Agent:
    system_prompt = prompts.load("coding") + prompts.project_instructions(cfg.root)
    # Tool del progetto + ricerca web (sola lettura) + delega di ricerca a un
    # sub-agente in sola lettura (`explore`) + scaletta dei passi (`plan`) per
    # i task multi-step.
    tools = coding_tools.TOOLS + web_tools.TOOLS + [subagent_tools.explore, plan_tools.plan]
    if getattr(cfg, "read_only", False):
        # Esecuzione non presidiata: nessuna modifica al filesystem né comandi.
        tools = [t for t in tools if not t.destructive]
    return Agent(
        name="coding",
        cfg=cfg,
        provider=provider,
        toolset=Toolset(tools),
        system_prompt=system_prompt,
        conversation=conversation,
        **callbacks,
    )
