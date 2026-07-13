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
    "Read a project text file with line numbers. Use offset/limit for large files.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the project root."},
            "offset": {"type": "integer", "description": "First line (1-based). Default 1."},
            "limit": {"type": "integer", "description": "Maximum number of lines. Omitted = to the end."},
        },
        "required": ["path"],
    },
)
def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int | None = None) -> str:
    return fs.read_file_impl(ctx.cfg.root, path, offset, limit, ctx.cfg.read_file_max_chars)


# ── list_directory ───────────────────────────────────────────────────────────

@tool(
    "list_directory",
    "List files and subfolders of a project directory (one level).",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory relative to the root. Default '.'."}},
    },
)
def list_directory(ctx: ToolContext, path: str = ".") -> str:
    return fs.list_dir_impl(ctx.cfg.root, path, ctx.cfg.list_dir_max_entries)


# ── glob ─────────────────────────────────────────────────────────────────────

@tool(
    "glob",
    "Find files by glob pattern, e.g. '**/*.py' or 'src/**/*.ts'.",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "path": {"type": "string", "description": "Starting directory. Default '.'."},
        },
        "required": ["pattern"],
    },
)
def glob(ctx: ToolContext, pattern: str, path: str = ".") -> str:
    base = fs.resolve(ctx.cfg.root, path)
    if not base.exists():
        return f"❌ Path does not exist: {fs.display(ctx.cfg.root, base)}"
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
        return f"No files match '{pattern}' under {fs.display(ctx.cfg.root, base)}"
    out = f"{len(matches)} files for '{pattern}':\n" + "\n".join(matches[:300])
    if len(matches) > 300:
        out += f"\n...[{len(matches) - 300} more]"
    return out


# ── grep ─────────────────────────────────────────────────────────────────────

@tool(
    "grep",
    ("Search a regex in the project's text files. Returns path:line: content. Great for "
     "definitions and usages of a symbol. With `context` it also shows N lines around each "
     "match (often saves a follow-up read_file); with `files_only` it lists only the files."),
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression."},
            "path": {"type": "string", "description": "Starting directory. Default '.'."},
            "glob_filter": {"type": "string", "description": "Filename filter, e.g. '*.py'. Optional."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive. Default false."},
            "context": {"type": "integer", "description": "Context lines before/after each match (grep -C style, max 10). Default 0."},
            "files_only": {"type": "boolean", "description": "If true, list only the files with their match counts, no lines. Default false."},
        },
        "required": ["pattern"],
    },
)
def grep(ctx: ToolContext, pattern: str, path: str = ".", glob_filter: str = "",
         ignore_case: bool = False, context: int = 0, files_only: bool = False) -> str:
    base = fs.resolve(ctx.cfg.root, path)
    if not base.exists():
        return f"❌ Path does not exist: {fs.display(ctx.cfg.root, base)}"
    ignore_case = fs.as_bool(ignore_case)   # il modello può inviare "true"/"false" come stringa
    files_only = fs.as_bool(files_only)
    try:
        context = max(0, min(int(context or 0), 10))  # tetto: il contesto non deve diventare una read_file
    except (TypeError, ValueError):
        context = 0
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"❌ Invalid regex: {exc}"

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
        return f"No matches for /{pattern}/ under {fs.display(ctx.cfg.root, base)}"
    label = f"{len(results)} files with matches" if files_only else f"{n_match} matches"
    out = f"{label} for /{pattern}/:\n" + "\n".join(results)
    return fs._trunc(out, ctx.cfg.grep_max_chars, hint="narrow the pattern or path")


# ── edit_file ────────────────────────────────────────────────────────────────

@tool(
    "edit_file",
    ("Surgical edit: replaces old_string with new_string. old_string must be unique in "
     "the file (include context) unless replace_all. Preferred over write_file for "
     "changing parts of an existing file."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the file to edit."},
            "old_string": {"type": "string", "description": "Exact text to replace (whitespace/indentation included)."},
            "new_string": {"type": "string", "description": "New text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences. Default false."},
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
    ("Create or overwrite a whole project file (intermediate folders are created). For "
     "targeted changes use edit_file. For very large files, write the first part and "
     "then add the rest with append=true (avoids exceeding the token limit)."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path."},
            "content": {"type": "string", "description": "Full file content."},
            "append": {"type": "boolean", "description": "Append instead of overwriting (to write a large file in parts). Default false."},
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
    ("Run a command in the system shell, in the project root (tests, builds, git, "
     "linters...). Uses cmd on Windows and sh on Unix. Returns stdout+stderr and the exit code."),
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to run."},
            "timeout": {"type": "integer", "description": "Timeout in seconds. Default 120."},
        },
        "required": ["command"],
    },
    destructive=True,
)
def run_command(ctx: ToolContext, command: str, timeout: int = 120) -> str:
    return shell.run_command_impl(command, timeout, cwd=str(ctx.cfg.root), max_chars=ctx.cfg.command_max_chars)


@tool(
    "multi_edit",
    ("Apply multiple replacements to ONE file in a single call, in order and atomically "
     "(if one fails, the file is left untouched). More efficient than many separate "
     "edit_file calls. Each edit uses the same resilient matching as edit_file."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path."},
            "edits": {
                "type": "array",
                "description": "List of edits, applied in sequence.",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string", "description": "Text to replace."},
                        "new_string": {"type": "string", "description": "New text."},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."},
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
        return f"❌ File does not exist: {fs.display(ctx.cfg.root, p)} (use write_file to create it)"
    if p.is_dir():
        return f"❌ It is a directory: {fs.display(ctx.cfg.root, p)}"
    if not isinstance(edits, list) or not edits:
        return "❌ 'edits' must be a non-empty list of edits."

    text = p.read_text(encoding="utf-8", errors="replace")
    working = text
    notes = []
    for i, e in enumerate(edits, 1):
        if not isinstance(e, dict) or "old_string" not in e or "new_string" not in e:
            return f"❌ Edit #{i} is invalid: 'old_string' and 'new_string' are required."
        # apply_edit solleva ToolError su match non univoco → annulliamo tutto
        # (il file non è ancora stato scritto: l'operazione resta atomica).
        try:
            working, strategy = fs.apply_edit(working, e["old_string"], e["new_string"], e.get("replace_all", False))
        except ToolError as exc:
            return f"❌ Edit #{i} not applied ({exc}) — no changes were written to the file."
        notes.append(strategy)

    if working == text:
        return f"⚠️ No change: the result is identical to {fs.display(ctx.cfg.root, p)}."
    p.write_text(working, encoding="utf-8")
    extra = [n for n in notes if n != "exact"]
    suffix = f" [match: {', '.join(extra)}]" if extra else ""
    return f"✓ Applied {len(edits)} edits to {fs.display(ctx.cfg.root, p)}{suffix}."


@tool(
    "repo_map",
    ("Compact map of the project: for each source file, its top-level definitions "
     "(functions, classes and their signatures). Use it BEFORE exploring in depth, to "
     "orient yourself cheaply instead of many list/grep/read calls. It is an overview: "
     "then read the files you need with read_file. Covers Python (precise, via AST), "
     "JS/TS, Go, Rust, Java, C#, C/C++ and many other languages."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Subfolder to map (default: the whole project root)."},
        },
    },
)
def repo_map(ctx: ToolContext, path: str = ".") -> str:
    base = fs.resolve(ctx.cfg.root, path)
    if not base.exists():
        return f"❌ Path does not exist: {fs.display(ctx.cfg.root, base)}"
    return repomap.build_repo_map(ctx.cfg.root, path, ctx.cfg.repomap_max_chars)


@tool(
    "move_path",
    ("Move or rename a file or a folder INSIDE the project root (both endpoints "
     "confined). The destination must not exist; intermediate folders are created. "
     "Preferred over mv/move via run_command: cross-platform and with no surprises."),
    {
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "Source path (file or folder)."},
            "dst": {"type": "string", "description": "Destination path (must not exist)."},
        },
        "required": ["src", "dst"],
    },
    destructive=True,
)
def move_path(ctx: ToolContext, src: str, dst: str) -> str:
    return fs.move_path_impl(ctx.cfg.root, src, dst)


TOOLS = [read_file, list_directory, glob, grep, repo_map, edit_file, multi_edit, write_file, move_path, run_command]
