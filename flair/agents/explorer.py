"""Costruzione del sub-agente esploratore.

Sola lettura e contesto isolato: usa solo i tool non distruttivi del coding agent
(`repo_map`, `list_directory`, `glob`, `grep`, `read_file`) più la ricerca web. Non
può modificare file né eseguire comandi. Serve a delegare un'indagine che
richiederebbe molte letture, tenendo snello il contesto del genitore.
"""

from __future__ import annotations

from .. import prompts
from ..core.agent import Agent
from ..core.tool import Toolset
from ..tools import coding as coding_tools
from ..tools import web as web_tools


def build(cfg, provider, conversation=None, **callbacks) -> Agent:
    # Solo i tool di lettura del coding agent (niente edit/scrittura/comandi) + web.
    readonly = [t for t in coding_tools.TOOLS if not t.destructive]
    system_prompt = prompts.load("explorer") + prompts.project_instructions(cfg.root)
    return Agent(
        name="explorer",
        cfg=cfg,
        provider=provider,
        toolset=Toolset(readonly + web_tools.TOOLS),
        system_prompt=system_prompt,
        conversation=conversation,
        **callbacks,
    )
