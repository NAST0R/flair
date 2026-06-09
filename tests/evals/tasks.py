"""Task di valutazione per flair (eval harness).

Ogni task è autosufficiente: prepara una cartella di lavoro, dà un prompt
all'agente e verifica il risultato in modo deterministico. Si eseguono dal vivo
con `run_evals.py` (servono le API key, come per flair); il runner riporta per
ciascuno successo, passi, token e cache-hit — così si **misura** se una modifica
migliora o peggiora flair, invece di andare a sensazione.

Il `check` riceve (cartella_di_lavoro, risposta_finale_dell'agente) e torna bool.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalTask:
    name: str
    description: str
    prompt: str
    setup: Callable[[Path], None]
    check: Callable[[Path, str], bool]
    agent: str = "coding"
    think: bool = False


def _load_module(workdir: Path, modname: str):
    """Importa un modulo .py dalla cartella di lavoro, fresco (senza cache)."""
    name = f"_eval_{modname}"
    spec = importlib.util.spec_from_file_location(name, workdir / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # serve durante l'exec (es. se il modulo definisce dataclass)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


# ── Task 1: correggere un bug che fa fallire un test ──────────────────────────
def _setup_fix(wd: Path) -> None:
    (wd / "mathops.py").write_text(
        "def add(a, b):\n    # bug: sottrae invece di sommare\n    return a - b\n", encoding="utf-8")
    (wd / "test_mathops.py").write_text(
        "from mathops import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")


def _check_fix(wd: Path, answer: str) -> bool:
    try:
        return _load_module(wd, "mathops").add(2, 3) == 5
    except Exception:
        return False


# ── Task 2: aggiungere una funzione secondo specifica ─────────────────────────
def _setup_add(wd: Path) -> None:
    (wd / "textutils.py").write_text("# aggiungi qui le utility di testo\n", encoding="utf-8")


def _check_add(wd: Path, answer: str) -> bool:
    try:
        slug = _load_module(wd, "textutils").slugify
        return slug("Ciao Mondo!") == "ciao-mondo" and slug("  A  B  ") == "a-b"
    except Exception:
        return False


# ── Task 3: trovare dove è definito un simbolo (esercita `explore`) ───────────
def _setup_find(wd: Path) -> None:
    (wd / "app.py").write_text("from billing import compute_total\n\nprint(compute_total([1, 2]))\n", encoding="utf-8")
    (wd / "billing.py").write_text(
        "TAX = 0.22\n\n\ndef compute_total(items):\n    return sum(items) * (1 + TAX)\n", encoding="utf-8")
    helpers = wd / "helpers"
    helpers.mkdir()
    (helpers / "io_utils.py").write_text("def read_csv(path):\n    return []\n", encoding="utf-8")


def _check_find(wd: Path, answer: str) -> bool:
    return "billing.py" in answer


TASKS: list[EvalTask] = [
    EvalTask(
        "fix-failing", "Corregge un bug che fa fallire un test",
        "Il test in test_mathops.py fallisce. Trova e correggi il bug nel codice "
        "sorgente (non modificare il test).",
        _setup_fix, _check_fix),
    EvalTask(
        "add-function", "Aggiunge una funzione secondo specifica",
        "In textutils.py aggiungi una funzione `slugify(s)` che restituisce la stringa "
        "in minuscolo, con gli spazi (anche multipli) sostituiti da un singolo trattino, "
        "gli spazi iniziali/finali rimossi e la punteggiatura eliminata. "
        "Esempio: 'Ciao Mondo!' deve dare 'ciao-mondo'.",
        _setup_add, _check_add),
    EvalTask(
        "find-definition", "Trova dove è definito un simbolo (usa explore)",
        "In quale file è definita la funzione `compute_total`? Indica il file.",
        _setup_find, _check_find),
]
