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


@tool(
    "forget",
    ("Remove ONE note from the session memory when it is clearly OBSOLETE or WRONG: "
     "superseded by a newer fact, contradicted by something you just verified, or the "
     "user asked for it. Identify the note by its text — exact, or a fragment that "
     "matches only that note. Do not prune notes just to tidy up. The note can still "
     "appear in your instructions until the session is reloaded: once forgotten, "
     "disregard it."),
    {
        "type": "object",
        "properties": {
            "note": {"type": "string",
                     "description": "The note to forget: its exact text, or a fragment matching only that note."},
        },
        "required": ["note"],
    },
    destructive=False,
)
def forget(ctx: ToolContext, note: str) -> str:
    """Speculare a `remember`, e con lo stesso contratto sulla cache: la nota esce
    subito dalla memoria (e dal sidecar al prossimo salvataggio), ma il system
    prompt si aggiorna solo al prossimo confine di sessione — riscriverlo adesso
    romperebbe il prefisso in cache. Il risultato del tool è ciò che rende il
    modello consapevole della rimozione nel frattempo."""
    mem = getattr(ctx, "memory", None)
    if mem is None:
        return "❌ Session memory is not available in this mode."
    ok, msg, removed = mem.remove(note, by_index=False)
    if not ok:
        return f"⚠️ {msg}"
    return (f"✓ Forgotten: «{removed}» — {msg} It may still appear in your instructions "
            "until the session is reloaded: disregard it from now on.")
