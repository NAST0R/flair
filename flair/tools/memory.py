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
    ("Store a DURABLE, non-obvious fact useful in future sessions: project commands, "
     "conventions, constraints, user preferences. ONE concise line per note. Do NOT "
     "use it for in-progress work state (it already lives in the conversation) nor "
     "for secrets/credentials (they would be rejected)."),
    {
        "type": "object",
        "properties": {
            "note": {"type": "string", "description": "The fact to remember, one concise line."},
        },
        "required": ["note"],
    },
    destructive=False,
)
def remember(ctx: ToolContext, note: str) -> str:
    mem = getattr(ctx, "memory", None)
    if mem is None:
        return "❌ Session memory is not available in this mode."
    ok, msg = mem.add(note)
    return ("✓ " if ok else "⚠️ ") + msg
