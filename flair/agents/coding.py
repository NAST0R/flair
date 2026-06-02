"""Costruzione dell'agente di coding."""

from __future__ import annotations

from .. import prompts
from ..core.agent import Agent
from ..core.tool import Toolset
from ..tools import coding as coding_tools


def build(cfg, provider, **callbacks) -> Agent:
    system_prompt = prompts.load("coding") + prompts.project_instructions(cfg.root)
    return Agent(
        name="coding",
        cfg=cfg,
        provider=provider,
        toolset=Toolset(coding_tools.TOOLS),
        system_prompt=system_prompt,
        **callbacks,
    )
