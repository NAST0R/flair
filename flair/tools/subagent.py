"""Tool `explore`: delega una ricerca a un sub-agente in sola lettura con contesto
ISOLATO.

L'esplorazione (molte `read_file`/`grep`/`repo_map`) resta nel contesto del
sub-agente e non gonfia quello del genitore: torna solo la sintesi finale. È la
leva principale di economia di token sui task grandi — più una capacità in più
(indagine specializzata) — restando sicura: il sub-agente non modifica nulla,
è confinato alla radice e ha un tetto di passi.
"""

from __future__ import annotations

from ..core.tool import ToolContext, tool


def _footer(steps: int) -> str:
    # Suffisso del risultato (model-facing: asserito dalla guardia inglese).
    return f"\n\n— 🔭 explored in {steps} step{'' if steps == 1 else 's'} (isolated context)"


@tool(
    "explore",
    ("Delegate an exploration/research question about the code to a read-only sub-agent "
     "with a separate context. Useful for investigations that would require many reads "
     "(e.g. \"where and how is X implemented?\", \"which files handle Y?\"): the sub-agent "
     "reads on your behalf and returns only a concise synthesis, without filling your "
     "context. It modifies nothing. Ask a precise, self-contained question; to read/edit "
     "a specific file you already know, use read_file/edit_file directly."),
    {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "The research question/goal, precise and self-contained."},
        },
        "required": ["task"],
    },
)
def explore(ctx: ToolContext, task: str) -> str:
    # Import pigro: spezza il ciclo tools → agents → tools a tempo di import.
    from ..agents import explorer
    from ..core.agent import Conversation

    if ctx.provider is None:
        return "❌ explore is not available in this context (provider missing)."

    # Sub-agente FRESCO con conversazione isolata: è il senso dell'isolamento di
    # contesto (le sue letture non entrano nel contesto del genitore).
    sub = explorer.build(ctx.cfg, ctx.provider, conversation=Conversation())
    steps_cap = getattr(ctx.cfg, "explorer_max_steps", 20)
    result = sub.run(task, max_steps=steps_cap)

    # Riporta i token del sub-agente al genitore tramite il contesto: sarà l'agente
    # a sommarli UNA volta sia al turno sia al totale di sessione (contabilità
    # centralizzata, niente doppi conteggi). Il costo delegato è reale e va contato.
    ctx.delegated_usage = result.usage if ctx.delegated_usage is None else ctx.delegated_usage + result.usage

    answer = result.content.strip() or "(no result from the exploration)"
    return answer + _footer(result.steps)
