"""Tool per l'agente di coding — confinati alla radice del progetto (cfg.root).

Lettura/scrittura/ricerca dirette: il modello apre i file e li legge, niente
indirezioni "outline → indovina nome → leggi" (la causa dei loop nel vecchio
progetto). Gli edit sono per match esatto di stringa: payload minimi, niente
riscritture di interi file.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path

from ..core.tool import ToolContext, ToolError, tool
from . import fs

# ── read_file ────────────────────────────────────────────────────────────────

@tool(
    "read_file",
    "Legge un file di testo del progetto con numeri di riga. Usa offset/limit per i file grandi.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path del file, relativo alla radice del progetto."},
            "offset": {"type": "integer", "description": "Prima riga (1-based). Default 1."},
            "limit": {"type": "integer", "description": "Numero massimo di righe. Omesso = fino alla fine."},
        },
        "required": ["path"],
    },
)
def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int | None = None) -> str:
    return fs.read_file_impl(ctx.cfg.root, path, offset, limit, ctx.cfg.read_file_max_chars)


# ── list_directory ───────────────────────────────────────────────────────────

@tool(
    "list_directory",
    "Elenca file e sottocartelle di una directory del progetto (un livello).",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory relativa alla radice. Default '.'."}},
    },
)
def list_directory(ctx: ToolContext, path: str = ".") -> str:
    return fs.list_dir_impl(ctx.cfg.root, path, ctx.cfg.list_dir_max_entries)


# ── glob ─────────────────────────────────────────────────────────────────────

@tool(
    "glob",
    "Trova file per pattern glob, es. '**/*.py' o 'src/**/*.ts'.",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Pattern glob."},
            "path": {"type": "string", "description": "Directory di partenza. Default '.'."},
        },
        "required": ["pattern"],
    },
)
def glob(ctx: ToolContext, pattern: str, path: str = ".") -> str:
    base = fs.resolve(ctx.cfg.root, path)
    matches: list[str] = []
    for root_dir, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in fs.NOISE_DIRS]
        for f in files:
            full = Path(root_dir) / f
            relp = fs.display(ctx.cfg.root, full)
            if fnmatch.fnmatch(relp, pattern) or fnmatch.fnmatch(f, pattern):
                matches.append(relp)
    matches.sort()
    if not matches:
        return f"Nessun file corrisponde a '{pattern}' sotto {fs.display(ctx.cfg.root, base)}"
    out = f"{len(matches)} file per '{pattern}':\n" + "\n".join(matches[:300])
    if len(matches) > 300:
        out += f"\n...[altri {len(matches) - 300}]"
    return out


# ── grep ─────────────────────────────────────────────────────────────────────

@tool(
    "grep",
    "Cerca una regex nei file di testo del progetto. Ritorna path:riga: contenuto. Ottimo per definizioni e usi di un simbolo.",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Espressione regolare."},
            "path": {"type": "string", "description": "Directory di partenza. Default '.'."},
            "glob_filter": {"type": "string", "description": "Filtro nomi file, es. '*.py'. Opzionale."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive. Default false."},
        },
        "required": ["pattern"],
    },
)
def grep(ctx: ToolContext, pattern: str, path: str = ".", glob_filter: str = "", ignore_case: bool = False) -> str:
    base = fs.resolve(ctx.cfg.root, path)
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"❌ Regex non valida: {exc}"

    results: list[str] = []
    for root_dir, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in fs.NOISE_DIRS]
        for f in files:
            if glob_filter and not fnmatch.fnmatch(f, glob_filter):
                continue
            full = Path(root_dir) / f
            if full.suffix.lower() in fs._BINARY_EXT:
                continue
            try:
                with full.open(encoding="utf-8", errors="replace") as fh:
                    for n, line in enumerate(fh, 1):
                        if rx.search(line):
                            results.append(f"{fs.display(ctx.cfg.root, full)}:{n}: {line.rstrip()}")
                            if len(results) >= 400:
                                break
            except OSError:
                continue
        if len(results) >= 400:
            break

    if not results:
        return f"Nessuna corrispondenza per /{pattern}/ sotto {fs.display(ctx.cfg.root, base)}"
    out = f"{len(results)} corrispondenze per /{pattern}/:\n" + "\n".join(results)
    return fs._trunc(out, ctx.cfg.grep_max_chars, hint="restringi pattern o path")


# ── edit_file ────────────────────────────────────────────────────────────────

@tool(
    "edit_file",
    ("Modifica chirurgica: sostituisce old_string con new_string. old_string deve essere "
     "univoco nel file (includi contesto) a meno di replace_all. Preferito a write_file "
     "per cambiare parti di un file esistente."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path del file da modificare."},
            "old_string": {"type": "string", "description": "Testo esatto da sostituire (spazi/indentazione inclusi)."},
            "new_string": {"type": "string", "description": "Testo nuovo."},
            "replace_all": {"type": "boolean", "description": "Sostituisce tutte le occorrenze. Default false."},
        },
        "required": ["path", "old_string", "new_string"],
    },
    destructive=True,
)
def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = fs.resolve(ctx.cfg.root, path)
    if not p.exists():
        return f"❌ Il file non esiste: {fs.display(ctx.cfg.root, p)} (usa write_file per crearlo)"
    if p.is_dir():
        return f"❌ È una directory: {fs.display(ctx.cfg.root, p)}"

    text = p.read_text(encoding="utf-8", errors="replace")
    # apply_edit solleva ToolError (gestita a monte come messaggio pulito) se il
    # match non è univoco; altrimenti applica anche con tolleranza sugli spazi.
    new_text, strategy = fs.apply_edit(text, old_string, new_string, replace_all)
    if new_text == text:
        return f"⚠️ Nessuna modifica: il nuovo testo è identico a quello presente in {fs.display(ctx.cfg.root, p)}."
    p.write_text(new_text, encoding="utf-8")
    note = "" if strategy == "esatto" else f" [match: {strategy}]"
    return f"✓ Modificato {fs.display(ctx.cfg.root, p)}{note}."


# ── write_file ───────────────────────────────────────────────────────────────

@tool(
    "write_file",
    "Crea o sovrascrive un intero file del progetto (crea le cartelle intermedie). Per modifiche puntuali usare edit_file.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path del file."},
            "content": {"type": "string", "description": "Contenuto completo del file."},
        },
        "required": ["path", "content"],
    },
    destructive=True,
)
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    p = fs.resolve(ctx.cfg.root, path)
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    verb = "Sovrascritto" if existed else "Creato"
    return f"✓ {verb} {fs.display(ctx.cfg.root, p)} ({len(content)} caratteri)."


# ── run_command ──────────────────────────────────────────────────────────────

@tool(
    "run_command",
    ("Esegue un comando nella shell di sistema, nella radice del progetto (test, build, "
     "git, linter...). Usa cmd su Windows e sh su Unix. Ritorna stdout+stderr ed exit code."),
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Comando da eseguire."},
            "timeout": {"type": "integer", "description": "Timeout in secondi. Default 120."},
        },
        "required": ["command"],
    },
    destructive=True,
)
def run_command(ctx: ToolContext, command: str, timeout: int = 120) -> str:
    try:
        proc = subprocess.run(command, shell=True, cwd=str(ctx.cfg.root),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"❌ Comando andato in timeout dopo {timeout}s: {command}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Errore eseguendo il comando: {exc}"
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    header = f"$ {command}\n(exit code {proc.returncode})\n"
    return fs._trunc(header + out.strip(), ctx.cfg.command_max_chars, hint="filtra o reindirizza l'output")


@tool(
    "multi_edit",
    ("Applica più sostituzioni a UN file in una sola chiamata, in ordine e in modo "
     "atomico (se una fallisce, il file non viene toccato). Più efficiente di tante "
     "edit_file separate. Ogni edit usa la stessa logica resiliente di edit_file."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path del file."},
            "edits": {
                "type": "array",
                "description": "Lista di modifiche, applicate in sequenza.",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string", "description": "Testo da sostituire."},
                        "new_string": {"type": "string", "description": "Testo nuovo."},
                        "replace_all": {"type": "boolean", "description": "Sostituire tutte le occorrenze (default false)."},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["path", "edits"],
    },
    destructive=True,
)
def multi_edit(ctx: ToolContext, path: str, edits: list) -> str:
    p = fs.resolve(ctx.cfg.root, path)
    if not p.exists():
        return f"❌ Il file non esiste: {fs.display(ctx.cfg.root, p)} (usa write_file per crearlo)"
    if p.is_dir():
        return f"❌ È una directory: {fs.display(ctx.cfg.root, p)}"
    if not isinstance(edits, list) or not edits:
        return "❌ 'edits' deve essere una lista non vuota di modifiche."

    text = p.read_text(encoding="utf-8", errors="replace")
    working = text
    notes = []
    for i, e in enumerate(edits, 1):
        if not isinstance(e, dict) or "old_string" not in e or "new_string" not in e:
            return f"❌ Modifica #{i} non valida: servono 'old_string' e 'new_string'."
        # apply_edit solleva ToolError su match non univoco → annulliamo tutto
        # (il file non è ancora stato scritto: l'operazione resta atomica).
        try:
            working, strategy = fs.apply_edit(working, e["old_string"], e["new_string"], e.get("replace_all", False))
        except ToolError as exc:
            return f"❌ Modifica #{i} non applicata ({exc}) — nessuna modifica scritta sul file."
        notes.append(strategy)

    if working == text:
        return f"⚠️ Nessuna modifica: il risultato è identico a {fs.display(ctx.cfg.root, p)}."
    p.write_text(working, encoding="utf-8")
    extra = [n for n in notes if n != "esatto"]
    suffix = f" [match: {', '.join(extra)}]" if extra else ""
    return f"✓ Applicate {len(edits)} modifiche a {fs.display(ctx.cfg.root, p)}{suffix}."


TOOLS = [read_file, list_directory, glob, grep, edit_file, multi_edit, write_file, run_command]
