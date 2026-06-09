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


@tool(
    "explore",
    ("Delega una domanda di esplorazione/ricerca sul codice a un sub-agente in sola "
     "lettura, con contesto separato. Utile per indagini che richiederebbero molte "
     "letture (es. \"dove e come è implementato X?\", \"quali file gestiscono Y?\"): il "
     "sub-agente legge per conto tuo e ti restituisce solo una sintesi concisa, senza "
     "riempire il tuo contesto. Non modifica nulla. Formula una domanda precisa e "
     "autosufficiente; per leggere/modificare un file specifico che già conosci usa "
     "direttamente read_file/edit_file."),
    {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "La domanda/obiettivo di ricerca, precisa e autosufficiente."},
        },
        "required": ["task"],
    },
)
def explore(ctx: ToolContext, task: str) -> str:
    # Import pigro: spezza il ciclo tools → agents → tools a tempo di import.
    from ..agents import explorer
    from ..core.agent import Conversation

    if ctx.provider is None:
        return "❌ explore non disponibile in questo contesto (provider mancante)."

    # Sub-agente FRESCO con conversazione isolata: è il senso dell'isolamento di
    # contesto (le sue letture non entrano nel contesto del genitore).
    sub = explorer.build(ctx.cfg, ctx.provider, conversation=Conversation())
    steps_cap = getattr(ctx.cfg, "explorer_max_steps", 20)
    result = sub.run(task, max_steps=steps_cap)

    # Riporta i token del sub-agente al genitore tramite il contesto: sarà l'agente
    # a sommarli UNA volta sia al turno sia al totale di sessione (contabilità
    # centralizzata, niente doppi conteggi). Il costo delegato è reale e va contato.
    ctx.delegated_usage = result.usage if ctx.delegated_usage is None else ctx.delegated_usage + result.usage

    answer = result.content.strip() or "(nessun risultato dall'esplorazione)"
    n = result.steps
    return f"{answer}\n\n— 🔭 esplorato in {n} pass{'o' if n == 1 else 'i'} (contesto isolato)"
