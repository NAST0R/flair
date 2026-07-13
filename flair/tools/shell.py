"""Esecuzione robusta di comandi di shell, multipiattaforma.

Due problemi che questo modulo risolve:

1. Su Windows i comandi multi-riga passati a cmd.exe (shell=True) vengono spezzati
   sui ritorni a capo: un `powershell -Command "...multi-riga..."` non arriva intatto
   e fallisce silenziosamente.
2. Gli script PowerShell complessi (here-string, Add-Type, più istruzioni) sono fragili
   inline a causa dell'escaping delle virgolette.

Per entrambi scriviamo lo script in un `.ps1` dentro una cartella temporanea ed
eseguiamo PowerShell con `-File`. La cartella temporanea è gestita da un context
manager (`TemporaryDirectory`): viene SEMPRE rimossa all'uscita dal blocco — anche
in caso di eccezione o timeout — quindi non restano file orfani sul sistema.
I comandi a riga singola continuano a passare per la shell (pipe, redirezioni,
%VAR%, builtin di cmd/sh).
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import tempfile

from . import fs

_OS = platform.system()  # 'Windows' | 'Darwin' | 'Linux'
_PS_EXE = "powershell" if _OS == "Windows" else "pwsh"

# `powershell [-flag...] -Command "<script>"` → cattura <script> (anche multi-riga).
_PS_WRAPPER = re.compile(
    r'^(?:powershell|pwsh)(?:\.exe)?\b[^"]*?\s-(?:command|c)\s+"(.*)"$',
    re.IGNORECASE | re.DOTALL,
)


def run_powershell_script(script: str, timeout: int, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Esegue uno script PowerShell scrivendolo in un `.ps1` temporaneo.

    Il file vive dentro una `TemporaryDirectory`: alla fine del blocco `with` la
    cartella (e quindi il `.ps1`) viene rimossa SEMPRE — successo, errore o timeout.
    Questa è la garanzia di pulizia: nessun residuo sul filesystem.

    Solleva ``subprocess.TimeoutExpired`` / ``FileNotFoundError`` come
    ``subprocess.run``; la gestione resta al chiamante.
    """
    with tempfile.TemporaryDirectory(prefix="flair_ps_") as tmpdir:
        path = os.path.join(tmpdir, "script.ps1")
        with open(path, "w", encoding="utf-8-sig") as fh:  # BOM: PowerShell legge l'UTF-8
            fh.write(script)
        return subprocess.run(
            [_PS_EXE, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
            capture_output=True, text=True, errors="replace", timeout=timeout, cwd=cwd,
        )


def powershell_script_for(command: str, is_windows: bool) -> str | None:
    """Per un comando *multi-riga su Windows*, restituisce il corpo PowerShell da
    eseguire tramite `.ps1` (scartando un eventuale wrapper `powershell -Command
    "..."`, così il nesting di processi sparisce). Restituisce ``None`` quando il
    comando va eseguito normalmente con la shell (riga singola, o non-Windows)."""
    if not is_windows or "\n" not in command:
        return None
    m = _PS_WRAPPER.match(command.strip())
    return m.group(1) if m else command


def run_shell(command: str, timeout: int, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Esegue ``command`` e restituisce il CompletedProcess (stdout/stderr/returncode).

    Solleva ``subprocess.TimeoutExpired`` allo scadere del timeout, come
    ``subprocess.run``: la gestione resta a carico del chiamante.
    """
    script = powershell_script_for(command, _OS == "Windows")
    if script is not None:
        return run_powershell_script(script, timeout, cwd)
    return subprocess.run(command, shell=True, capture_output=True, text=True,
                          errors="replace", timeout=timeout, cwd=cwd)


def format_command_output(proc: subprocess.CompletedProcess, command: str | None,
                          max_chars: int, hint: str = "") -> str:
    """Compone l'output di un processo (stdout + eventuale stderr + exit code) e lo
    tronca. Se ``command`` è dato, antepone la riga ``$ command``. Unico punto per
    questa formattazione, condiviso da run_command (coding e generico) e PowerShell."""
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    header = (f"$ {command}\n" if command else "") + f"(exit code {proc.returncode})\n"
    return fs._trunc(header + out.strip(), max_chars, hint=hint)


def run_command_impl(command: str, timeout: int, cwd: str | None, max_chars: int) -> str:
    """Esegue un comando di shell e ne restituisce l'output formattato/troncato.
    Condivisa dai due agenti: cambia solo ``cwd`` (coding → radice del progetto;
    generico → directory di processo)."""
    try:
        proc = run_shell(command, timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return f"❌ Command timed out after {timeout}s: {command}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error running the command: {exc}"
    return format_command_output(proc, command, max_chars, hint="filter or redirect the output")
