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
from pathlib import Path

from ..core.tool import ToolContext, ToolError, tool
from . import fs, repomap, shell

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
    if not base.exists():
        return f"❌ Il path non esiste: {fs.display(ctx.cfg.root, base)}"
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
    ("Cerca una regex nei file di testo del progetto. Ritorna path:riga: contenuto. Ottimo per "
     "definizioni e usi di un simbolo. Con `context` mostra anche N righe attorno a ogni match "
     "(spesso evita una read_file successiva); con `files_only` elenca solo i file."),
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Espressione regolare."},
            "path": {"type": "string", "description": "Directory di partenza. Default '.'."},
            "glob_filter": {"type": "string", "description": "Filtro nomi file, es. '*.py'. Opzionale."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive. Default false."},
            "context": {"type": "integer", "description": "Righe di contesto prima/dopo ogni match (stile grep -C, max 10). Default 0."},
            "files_only": {"type": "boolean", "description": "Se true elenca solo i file col numero di match, senza le righe. Default false."},
        },
        "required": ["pattern"],
    },
)
def grep(ctx: ToolContext, pattern: str, path: str = ".", glob_filter: str = "",
         ignore_case: bool = False, context: int = 0, files_only: bool = False) -> str:
    base = fs.resolve(ctx.cfg.root, path)
    if not base.exists():
        return f"❌ Il path non esiste: {fs.display(ctx.cfg.root, base)}"
    ignore_case = fs.as_bool(ignore_case)   # il modello può inviare "true"/"false" come stringa
    files_only = fs.as_bool(files_only)
    try:
        context = max(0, min(int(context or 0), 10))  # tetto: il contesto non deve diventare una read_file
    except (TypeError, ValueError):
        context = 0
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"❌ Regex non valida: {exc}"

    def _files():
        # Se 'path' è già un file, cerca in QUEL file (intuitivo: "grep in questo
        # file"); altrimenti percorri la cartella saltando le dir di rumore.
        if base.is_file():
            yield base
            return
        for root_dir, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in fs.NOISE_DIRS]
            for f in files:
                if glob_filter and not fnmatch.fnmatch(f, glob_filter):
                    continue
                yield Path(root_dir) / f

    results: list[str] = []
    n_match = 0
    for full in _files():
        if full.suffix.lower() in fs._BINARY_EXT:
            continue
        relp = fs.display(ctx.cfg.root, full)
        try:
            if files_only:
                # Scoperta larga a costo minimo: solo "file (n. match)", niente righe.
                with full.open(encoding="utf-8", errors="replace") as fh:
                    count = sum(1 for line in fh if rx.search(line))
                if count:
                    results.append(f"{relp} ({count})")
                    n_match += count
            elif context == 0:
                # Percorso classico, streaming riga per riga (invariato).
                with full.open(encoding="utf-8", errors="replace") as fh:
                    for n, line in enumerate(fh, 1):
                        if rx.search(line):
                            results.append(f"{relp}:{n}: {line.rstrip()}")
                            n_match += 1
                            if len(results) >= 400:
                                break
            else:
                # Con contesto: righe del file in memoria (solo per i file che matchano),
                # intervalli [i-C, i+C] FUSI se si toccano — come `grep -C` — così due
                # match vicini non duplicano le righe. Match marcati con ':', contesto
                # con '-', blocchi separati da '--'.
                lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
                hits = [i for i, line in enumerate(lines) if rx.search(line)]
                if not hits:
                    continue
                n_match += len(hits)
                hitset = set(hits)
                spans: list[list[int]] = []
                for i in hits:
                    lo, hi = max(0, i - context), min(len(lines) - 1, i + context)
                    if spans and lo <= spans[-1][1] + 1:
                        spans[-1][1] = max(spans[-1][1], hi)
                    else:
                        spans.append([lo, hi])
                for lo, hi in spans:
                    if results:
                        results.append("--")
                    for i in range(lo, hi + 1):
                        sep = ":" if i in hitset else "-"
                        results.append(f"{relp}{sep}{i + 1}{sep} {lines[i].rstrip()}")
                    if len(results) >= 400:
                        break
        except OSError:
            continue
        if len(results) >= 400:
            break

    if not results:
        return f"Nessuna corrispondenza per /{pattern}/ sotto {fs.display(ctx.cfg.root, base)}"
    label = f"{len(results)} file con corrispondenze" if files_only else f"{n_match} corrispondenze"
    out = f"{label} per /{pattern}/:\n" + "\n".join(results)
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
    return fs.edit_file_impl(ctx.cfg.root, path, old_string, new_string, replace_all)


# ── write_file ───────────────────────────────────────────────────────────────

@tool(
    "write_file",
    ("Crea o sovrascrive un intero file del progetto (crea le cartelle intermedie). Per "
     "modifiche puntuali usare edit_file. Per file molto grandi, scrivi la prima parte e "
     "poi aggiungi il resto con append=true (eviti di superare il limite di token)."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path del file."},
            "content": {"type": "string", "description": "Contenuto completo del file."},
            "append": {"type": "boolean", "description": "Aggiunge in coda invece di sovrascrivere (per scrivere un file grande in più parti). Default false."},
        },
        "required": ["path", "content"],
    },
    destructive=True,
)
def write_file(ctx: ToolContext, path: str, content: str, append: bool = False) -> str:
    return fs.write_file_impl(ctx.cfg.root, path, content, append)


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
    return shell.run_command_impl(command, timeout, cwd=str(ctx.cfg.root), max_chars=ctx.cfg.command_max_chars)


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


@tool(
    "repo_map",
    ("Mappa compatta del progetto: per ogni file sorgente le definizioni di primo "
     "livello (funzioni, classi e relative firme). Usalo PRIMA di esplorare in "
     "profondità, per orientarti a basso costo invece di tante chiamate list/grep/read. "
     "È una panoramica: poi leggi col read_file i file che ti servono. Copre Python "
     "(preciso, via AST), JS/TS, Go, Rust, Java, C#, C/C++ e molti altri linguaggi."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Sottocartella da mappare (default: tutta la radice del progetto)."},
        },
    },
)
def repo_map(ctx: ToolContext, path: str = ".") -> str:
    base = fs.resolve(ctx.cfg.root, path)
    if not base.exists():
        return f"❌ Il path non esiste: {fs.display(ctx.cfg.root, base)}"
    return repomap.build_repo_map(ctx.cfg.root, path, ctx.cfg.repomap_max_chars)


@tool(
    "move_path",
    ("Sposta o rinomina un file o una cartella DENTRO la root del progetto (entrambi i capi "
     "confinati). La destinazione non deve esistere; le cartelle intermedie vengono create. "
     "Preferito a mv/move via run_command: cross-platform e senza sorprese."),
    {
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "Percorso di origine (file o cartella)."},
            "dst": {"type": "string", "description": "Percorso di destinazione (non deve esistere)."},
        },
        "required": ["src", "dst"],
    },
    destructive=True,
)
def move_path(ctx: ToolContext, src: str, dst: str) -> str:
    return fs.move_path_impl(ctx.cfg.root, src, dst)


TOOLS = [read_file, list_directory, glob, grep, repo_map, edit_file, multi_edit, write_file, move_path, run_command]
