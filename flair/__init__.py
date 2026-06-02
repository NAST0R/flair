"""flair 3.0 — assistente AI agentico (coding + generico) su DeepSeek e OpenAI.

Architettura: un solo motore agentico append-only (cache del prefisso),
provider intercambiabili, due agenti (coding sandboxato, generico desktop),
router minimale.
"""

from .config import Config, load_config
from .core import Agent, AgentResult, Tool, Toolset
from .llm import create_provider

__version__ = "3.0.0"
__all__ = ["Config", "load_config", "create_provider", "Agent", "AgentResult", "Toolset", "Tool", "__version__"]
