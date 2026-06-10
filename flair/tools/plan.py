"""Tool `plan`: scaletta/TODO esplicita per i task multi-step.

Il vero divoratore di token non è il prezzo per token ma il *flailing*: un task
che prende 25 passi invece di 12 costa il doppio. Una scaletta che il modello
scrive e aggiorna (spuntando i passi) lo tiene focalizzato: meno divagazioni,
più affidabilità sui task lunghi — la stessa contromisura dei coding agent di
riferimento.

Design volutamente semplice e robusto:
- stateless: ogni chiamata RISCRIVE l'intera scaletta (liste corte, costo nullo,
  zero stato da sincronizzare); la versione più recente vive nella conversazione;
- tollerante: accetta voci come oggetti {title, status} o semplici stringhe, e
  status anche in inglese (pending/in_progress/done...);
- limitato: tetto su numero di passi e lunghezza dei titoli (niente abusi di
  contesto);
- non distruttivo: è solo testo, nessun effetto sul filesystem.
"""

from __future__ import annotations

from ..core.tool import ToolContext, tool

_MAX_STEPS = 30
_MAX_TITLE = 200

# Stati canonici e sinonimi tollerati (il modello mescola le lingue).
_STATUS = {
    "da_fare": "da_fare", "todo": "da_fare", "pending": "da_fare", "aperto": "da_fare",
    "in_corso": "in_corso", "in_progress": "in_corso", "doing": "in_corso", "wip": "in_corso",
    "fatto": "fatto", "done": "fatto", "completed": "fatto", "completato": "fatto", "ok": "fatto",
}
_ICON = {"da_fare": "○", "in_corso": "▸", "fatto": "✔"}


def _normalize(step) -> tuple[str, str] | None:
    """(titolo, status) da una voce; None se la voce è inutilizzabile."""
    if isinstance(step, str):
        title, status = step, "da_fare"
    elif isinstance(step, dict):
        title = step.get("title") or step.get("titolo") or step.get("content") or ""
        status = str(step.get("status") or step.get("stato") or "da_fare")
    else:
        return None
    title = " ".join(str(title).split())
    if not title:
        return None
    if len(title) > _MAX_TITLE:
        title = title[: _MAX_TITLE - 1] + "…"
    return title, _STATUS.get(status.strip().lower(), "da_fare")


@tool(
    "plan",
    ("Scrive o aggiorna la scaletta dei passi per il task corrente. Usalo all'inizio "
     "dei task multi-step (3+ passi distinti) e RIscrivilo man mano che procedi, "
     "marcando ogni passo come da_fare, in_corso o fatto e aggiungendo/rimuovendo passi "
     "se il piano cambia. Ti tiene focalizzato e evita passi sprecati. Per i task "
     "semplici (1-2 passi) non serve."),
    {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "La scaletta COMPLETA e aggiornata, in ordine.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Il passo, breve e concreto."},
                        "status": {"type": "string", "enum": ["da_fare", "in_corso", "fatto"],
                                   "description": "Stato del passo (default: da_fare)."},
                    },
                    "required": ["title"],
                },
            },
        },
        "required": ["steps"],
    },
)
def plan(ctx: ToolContext, steps: list) -> str:
    if not isinstance(steps, list) or not steps:
        return "❌ Scaletta vuota: passa `steps` con almeno un passo ({title, status})."
    norm = [n for n in (_normalize(s) for s in steps) if n]
    if not norm:
        return "❌ Nessun passo valido: ogni voce deve avere un `title`."
    extra = ""
    if len(norm) > _MAX_STEPS:
        extra = f"\n…[{len(norm) - _MAX_STEPS} passi oltre il limite di {_MAX_STEPS}: accorpa la scaletta]"
        norm = norm[:_MAX_STEPS]
    done = sum(1 for _, st in norm if st == "fatto")
    lines = [f"📋 Piano ({done}/{len(norm)} fatti)"]
    for title, st in norm:
        suffix = " (in corso)" if st == "in_corso" else ""
        lines.append(f"  {_ICON[st]} {title}{suffix}")
    return "\n".join(lines) + extra
