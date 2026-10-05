"""Controllo automatico dopo ogni modifica di un file (FLAIR_POST_EDIT_CMD).

Senza, un errore di sintassi introdotto da un edit lo si scopre passi dopo,
lanciando i test: altri giri di modello e altro contesto. Con un comando qui
(`ruff check {path}`, `python -m py_compile {path}`, `node --check {path}`…) il
modello vede il problema NELLO STESSO PASSO in cui l'ha creato.

Due regole tengono basso il costo:

* l'output entra nel risultato del tool **solo se il comando fallisce** (exit
  diverso da zero): un file pulito non aggiunge un solo token;
* il risultato del tool è un messaggio appena prodotto e non ancora inviato,
  quindi accodarvi l'esito è append-only — la cache del prefisso non si tocca.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
from pathlib import Path

_TIMEOUT = 30
_MAX_OUTPUT = 2000


def _quote(path: Path) -> str:
    """Il path come argomento di shell, su entrambi i sistemi."""
    return subprocess.list2cmdline([str(path)]) if os.name == "nt" else shlex.quote(str(path))


def run_check(cfg, path: Path) -> str:
    """Esegue il controllo configurato su `path`. Ritorna il testo da accodare al
    risultato del tool, oppure "" se non c'è nulla da dire (comando non
    configurato, file fuori dal filtro, oppure controllo superato)."""
    command = (getattr(cfg, "post_edit_cmd", "") or "").strip()
    if not command:
        return ""
    pattern = (getattr(cfg, "post_edit_glob", "") or "*").strip()
    if not any(fnmatch.fnmatch(path.name, p.strip()) for p in pattern.split(",") if p.strip()):
        return ""
    full = command.replace("{path}", _quote(path)) if "{path}" in command else f"{command} {_quote(path)}"
    try:
        proc = subprocess.run(full, shell=True, capture_output=True, text=True,
                              errors="replace", timeout=_TIMEOUT,
                              cwd=str(getattr(cfg, "root", None) or path.parent))
    except subprocess.TimeoutExpired:
        return f"\n[post-edit check timed out after {_TIMEOUT}s: {command}]"
    except OSError as exc:
        return f"\n[post-edit check could not run ({exc}): {command}]"
    if proc.returncode == 0:
        return ""
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if len(out) > _MAX_OUTPUT:
        out = out[:_MAX_OUTPUT] + "\n…[output truncated]"
    return f"\n[post-edit check failed (exit {proc.returncode}): {command}]\n{out}"
