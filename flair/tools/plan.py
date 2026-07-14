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
    "todo": "todo", "da_fare": "todo", "pending": "todo", "aperto": "todo",
    "in_progress": "in_progress", "in_corso": "in_progress", "doing": "in_progress", "wip": "in_progress",
    "done": "done", "fatto": "done", "completed": "done", "completato": "done", "ok": "done",
}
_ICON = {"todo": "○", "in_progress": "▸", "done": "✔"}


def _normalize(step) -> tuple[str, str] | None:
    """(titolo, status) da una voce; None se la voce è inutilizzabile."""
    if isinstance(step, str):
        title, status = step, "todo"
    elif isinstance(step, dict):
        title = step.get("title") or step.get("titolo") or step.get("content") or ""
        status = str(step.get("status") or step.get("stato") or "todo")
    else:
        return None
    title = " ".join(str(title).split())
    if not title:
        return None
    if len(title) > _MAX_TITLE:
        title = title[: _MAX_TITLE - 1] + "…"
    return title, _STATUS.get(status.strip().lower(), "todo")


@tool(
    "plan",
    ("Write or update the step-by-step plan for the current task. Use it at the "
     "start of multi-step tasks (3+ distinct steps) and REwrite it as you go, "
     "marking each step as todo, in_progress or done, adding/removing steps "
     "if the plan changes. It keeps you focused and avoids wasted steps. Skip "
     "it for simple tasks (1-2 steps)."),
    {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "The COMPLETE, up-to-date plan, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "The step, short and concrete."},
                        "status": {"type": "string", "enum": ["todo", "in_progress", "done"],
                                   "description": "Step status (default: todo)."},
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
        return "❌ Empty plan: pass `steps` with at least one step ({title, status})."
    norm = [n for n in (_normalize(s) for s in steps) if n]
    if not norm:
        return "❌ No valid steps: each entry must have a `title`."
    extra = ""
    if len(norm) > _MAX_STEPS:
        extra = f"\n…[{len(norm) - _MAX_STEPS} steps over the {_MAX_STEPS} limit: consolidate the plan]"
        norm = norm[:_MAX_STEPS]
    done = sum(1 for _, st in norm if st == "done")
    lines = [f"📋 Plan ({done}/{len(norm)} done)"]
    for title, st in norm:
        suffix = " (in progress)" if st == "in_progress" else ""
        lines.append(f"  {_ICON[st]} {title}{suffix}")
    return "\n".join(lines) + extra
