"""Tool `remember`: appunta un fatto DUREVOLE nella memoria di sessione.

Non distruttivo per costruzione: non tocca i file dell'utente — scrive solo
nello stato in RAM della sessione (e, al salvataggio, nel sidecar dedicato
dentro la cartella sessioni di flair, con tetto duro). Le guardie (dedup,
filtro segreti, limiti) vivono in flair.memory.SessionMemory e sono
deterministiche: zero chiamate LLM, esito sempre spiegato al modello.
"""

from __future__ import annotations

from ..core.tool import ToolContext, tool


@tool(
    "remember",
    ("Memorizza un fatto DUREVOLE e non ovvio, utile nelle sessioni future: comandi "
     "del progetto, convenzioni, vincoli, preferenze dell'utente. UNA riga concisa "
     "per nota. NON usarlo per lo stato del lavoro in corso (vive già nella "
     "conversazione) né per segreti/credenziali (verrebbero rifiutati)."),
    {
        "type": "object",
        "properties": {
            "note": {"type": "string", "description": "Il fatto da ricordare, in una riga concisa."},
        },
        "required": ["note"],
    },
    destructive=False,
)
def remember(ctx: ToolContext, note: str) -> str:
    mem = getattr(ctx, "memory", None)
    if mem is None:
        return "❌ Memoria di sessione non disponibile in questa modalità."
    ok, msg = mem.add(note)
    return ("✓ " if ok else "⚠️ ") + msg
