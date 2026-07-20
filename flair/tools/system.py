"""Tool per l'agente generico — automazione desktop cross-platform.

Funzionano su Windows, macOS e Linux. Operano sull'intera macchina (non sono
confinati a una radice), perché il loro scopo è proprio quello: aprire il
browser, trovare un file in Musica, leggere un documento sul Desktop, dare info
di sistema. I tool potenzialmente rischiosi (run_command) restano dietro al
gate di approvazione.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from ..core.tool import ToolContext, tool
from . import fs, shell

_OS = platform.system()  # 'Windows' | 'Darwin' | 'Linux'


def _run_quiet(args: list[str], text_input: str | None = None, timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, errors="replace",
                          input=text_input, timeout=timeout)


def _user_dirs() -> list[Path]:
    """Cartelle utente comuni in cui cercare (curate, esistenti)."""
    home = Path.home()
    names = ["Music", "Downloads", "Desktop", "Documents", "Videos", "Pictures",
             "Musica", "Scaricati", "Scrivania", "Documenti", "Video", "Immagini"]
    dirs = [home / n for n in names]
    # Linux: rispetta le XDG user dirs (possono essere localizzate).
    if _OS == "Linux":
        for key in ("MUSIC", "DOWNLOAD", "DESKTOP", "DOCUMENTS", "VIDEOS", "PICTURES"):
            try:
                r = _run_quiet(["xdg-user-dir", key])
                if r.returncode == 0 and r.stdout.strip():
                    dirs.append(Path(r.stdout.strip()))
            except (OSError, subprocess.SubprocessError):
                break
    seen, out = set(), []
    for d in dirs:
        if d.exists() and d.is_dir() and str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out or [home]


# ── open_url ─────────────────────────────────────────────────────────────────

@tool(
    "open_url",
    "Open a URL in the default browser. Use this for 'open the browser' or to open a website.",
    {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "URL to open (https://...). To open just the browser use about:blank."}},
        "required": ["url"],
    },
)
def open_url(ctx: ToolContext, url: str) -> str:
    if not (url.startswith("http://") or url.startswith("https://") or url.startswith("about:")):
        url = "https://" + url
    try:
        ok = webbrowser.open(url)
        return f"✓ Opened in the browser: {url}" if ok else f"⚠️ Could not open the browser for {url}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error opening the URL: {exc}"


# ── open_path ────────────────────────────────────────────────────────────────

@tool(
    "open_path",
    "Open a file or folder with the system default application (e.g. a song in the player, a PDF in the reader, a folder in the file manager).",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path of the file or folder (absolute or ~)."}},
        "required": ["path"],
    },
)
def open_path(ctx: ToolContext, path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"❌ Path does not exist: {p}"
    try:
        if _OS == "Windows":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif _OS == "Darwin":
            _run_quiet(["open", str(p)])
        else:
            _run_quiet(["xdg-open", str(p)])
        return f"✓ Opened: {p}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error opening the path: {exc}"


# ── open_application ─────────────────────────────────────────────────────────

@tool(
    "open_application",
    "Launch an application by name (e.g. 'chrome', 'notepad', 'code', 'spotify'). Best-effort cross-platform.",
    {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Application name or executable."}},
        "required": ["name"],
    },
)
def open_application(ctx: ToolContext, name: str) -> str:
    try:
        if _OS == "Windows":
            # 'start' risolve gli eseguibili via PATH e App Paths del registro.
            subprocess.Popen(["cmd", "/c", "start", "", name], shell=False)
            return f"✓ Launch requested: {name}"
        if _OS == "Darwin":
            r = _run_quiet(["open", "-a", name])
            return f"✓ Launched: {name}" if r.returncode == 0 else f"❌ App not found: {name} ({r.stderr.strip()})"
        # Linux: prova l'eseguibile diretto, poi gtk-launch.
        try:
            subprocess.Popen([name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"✓ Launched: {name}"
        except FileNotFoundError:
            r = _run_quiet(["gtk-launch", name])
            if r.returncode == 0:
                return f"✓ Launched: {name}"
            return f"❌ Application not found: {name}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error launching the application: {exc}"


# ── search_files ─────────────────────────────────────────────────────────────

@tool(
    "search_files",
    ("Search files on the computer by name and optional extension. By default it searches "
     "the user folders (Music, Downloads, Desktop, Documents, Videos, Pictures). Use "
     "'locations' to search elsewhere. Great for 'find me a song', 'where is that PDF'."),
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text to search in the filename (case-insensitive). Empty = any name."},
            "extensions": {"type": "array", "items": {"type": "string"}, "description": "Extensions to filter, e.g. ['.mp3', '.flac']. Optional."},
            "locations": {"type": "array", "items": {"type": "string"}, "description": "Folders to search in. Optional (default: user folders)."},
        },
    },
)
def search_files(ctx: ToolContext, query: str = "", extensions: list[str] | None = None, locations: list[str] | None = None) -> str:
    # Il modello a volte passa una stringa al posto di una lista (es. extensions="mp3"):
    # senza questo, "mp3" verrebbe iterato carattere per carattere → zero risultati silenziosi.
    if isinstance(extensions, str):
        extensions = [extensions]
    if isinstance(locations, str):
        locations = [locations]
    q = (query or "").lower()
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in (extensions or [])}
    roots = [Path(p).expanduser() for p in locations] if locations else _user_dirs()
    roots = [r for r in roots if r.exists()]
    if not roots:
        return "❌ None of the given folders exists."

    results: list[str] = []
    scanned = 0
    truncated = False
    for base in roots:
        for root_dir, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in fs.NOISE_DIRS and not d.startswith(".")]
            for f in files:
                scanned += 1
                if scanned > ctx.cfg.search_max_scanned:
                    truncated = True
                    break
                if exts and Path(f).suffix.lower() not in exts:
                    continue
                if q and q not in f.lower():
                    continue
                results.append(str(Path(root_dir) / f))
                if len(results) >= ctx.cfg.search_max_results:
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break

    if not results:
        where = ", ".join(str(r) for r in roots)
        return f"No files found (query='{query}', extensions={sorted(exts) or 'any'}) in: {where}"
    out = f"{len(results)} files found:\n" + "\n".join(results)
    if truncated:
        out += "\n...[search stopped at the limit; narrow query/extensions or set 'locations']"
    return out


# ── list_directory (whole machine) ───────────────────────────────────────────

@tool(
    "list_directory",
    "List the contents of any folder on the computer (one level).",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Folder (absolute or ~). Default: home."}},
    },
)
def list_directory(ctx: ToolContext, path: str = "~") -> str:
    return fs.list_dir_impl(None, path, ctx.cfg.list_dir_max_entries)


# ── read_file (whole machine) ────────────────────────────────────────────────

@tool(
    "read_file",
    "Read any text file on the computer, with line numbers.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (absolute or ~)."},
            "offset": {"type": "integer", "description": "First line (1-based). Default 1."},
            "limit": {"type": "integer", "description": "Maximum number of lines."},
        },
        "required": ["path"],
    },
)
def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int | None = None) -> str:
    return fs.read_file_impl(None, path, offset, limit, ctx.cfg.read_file_max_chars)


# ── write_file / edit_file (qualsiasi percorso del computer) ──────────────────

@tool(
    "write_file",
    ("Create or overwrite any text file on the computer with the given content (missing "
     "folders are created too). USE THIS to create a file (e.g. a report), not "
     "run_command: it is direct and reliable, no shell/escaping issues. For very large "
     "files, write the first part and add the rest with append=true."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (absolute or ~)."},
            "content": {"type": "string", "description": "Full file content."},
            "append": {"type": "boolean", "description": "Append instead of overwriting (to write a large file in parts). Default false."},
        },
        "required": ["path", "content"],
    },
    destructive=True,
)
def write_file(ctx: ToolContext, path: str, content: str, append: bool = False) -> str:
    return fs.write_file_impl(None, path, content, append)


@tool(
    "edit_file",
    ("Replace an exact portion of text in any file on the computer (matching resilient "
     "to whitespace/indentation). For small targeted changes; to create or rewrite a "
     "whole file use write_file."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (absolute or ~)."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "New text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."},
        },
        "required": ["path", "old_string", "new_string"],
    },
    destructive=True,
)
def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    return fs.edit_file_impl(None, path, old_string, new_string, replace_all)


# ── run_command ──────────────────────────────────────────────────────────────

@tool(
    "run_command",
    "Run a command in the system shell (cmd on Windows, sh on Unix). For advanced system operations. Returns stdout+stderr and the exit code.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to run."},
            "timeout": {"type": "integer", "description": "Timeout in seconds. Default 60."},
        },
        "required": ["command"],
    },
    destructive=True,
)
def run_command(ctx: ToolContext, command: str, timeout: int = 60) -> str:
    return shell.run_command_impl(command, timeout, cwd=None, max_chars=ctx.cfg.command_max_chars)


# ── run_powershell (script su file temporaneo, pulizia garantita) ─────────────

@tool(
    "run_powershell",
    ("Run a PowerShell script, even MULTI-LINE (here-strings, Add-Type, multiple "
     "statements). flair writes it to a temporary file, runs it and ALWAYS deletes it "
     "(even on error or timeout). USE THIS for complex PowerShell instead of passing it "
     "inline to run_command: it avoids escaping and newline issues."),
    {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "The PowerShell script (plain PowerShell, no shell escaping)."},
            "timeout": {"type": "integer", "description": "Timeout in seconds. Default 60."},
        },
        "required": ["script"],
    },
    destructive=True,
)
def run_powershell(ctx: ToolContext, script: str, timeout: int = 60) -> str:
    try:
        proc = shell.run_powershell_script(script, timeout)
    except subprocess.TimeoutExpired:
        return f"❌ PowerShell script timed out after {timeout}s."
    except FileNotFoundError:
        return "❌ PowerShell not found on this system."
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error running the PowerShell script: {exc}"
    return shell.format_command_output(proc, None, ctx.cfg.command_max_chars, hint="filter the output")


# ── system_info ──────────────────────────────────────────────────────────────

@tool(
    "system_info",
    "Return system information: OS, version, architecture, CPU, RAM (if available), hostname.",
    {"type": "object", "properties": {}},
)
def system_info(ctx: ToolContext) -> str:
    lines = [
        f"OS: {platform.system()} {platform.release()} ({platform.version()})",
        f"Architecture: {platform.machine()}",
        f"Processor: {platform.processor() or 'n/a'}",
        f"Logical CPUs: {os.cpu_count()}",
        f"Hostname: {platform.node()}",
        f"Python: {platform.python_version()} ({sys.executable})",
    ]
    try:
        import psutil  # opzionale

        vm = psutil.virtual_memory()
        lines.append(f"RAM: {vm.used // (1024**2)} MB used / {vm.total // (1024**2)} MB ({vm.percent}%)")
        lines.append(f"CPU usage: {psutil.cpu_percent(interval=0.2)}%")
    except Exception:  # noqa: BLE001
        lines.append("RAM/CPU%: (install 'psutil' for details)")
    return "\n".join(lines)


# ── get_datetime ─────────────────────────────────────────────────────────────

@tool(
    "get_datetime",
    "Return the current system date and time. Use it instead of guessing the date.",
    {"type": "object", "properties": {}},
)
def get_datetime(ctx: ToolContext) -> str:
    now = datetime.now().astimezone()
    return now.strftime("%A %d %B %Y, %H:%M:%S %Z").strip()


# ── clipboard ────────────────────────────────────────────────────────────────

def _clipboard_get() -> tuple[bool, str]:
    try:
        import pyperclip  # opzionale, multipiattaforma

        return True, pyperclip.paste()
    except Exception:  # noqa: BLE001
        pass
    try:
        if _OS == "Windows":
            r = _run_quiet(["powershell", "-NoProfile", "-Command", "Get-Clipboard"])
        elif _OS == "Darwin":
            r = _run_quiet(["pbpaste"])
        else:
            for cmd in (["xclip", "-selection", "clipboard", "-o"], ["xsel", "-b"], ["wl-paste"]):
                try:
                    r = _run_quiet(cmd)
                    break
                except FileNotFoundError:
                    continue
            else:
                return False, "no clipboard tool (install xclip/xsel/wl-clipboard or pyperclip)"
        if r.returncode == 0:
            return True, r.stdout
        return False, r.stderr.strip() or "clipboard error"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _clipboard_set(text: str) -> tuple[bool, str]:
    try:
        import pyperclip

        pyperclip.copy(text)
        return True, ""
    except Exception:  # noqa: BLE001
        pass
    try:
        if _OS == "Windows":
            _run_quiet(["clip"], text_input=text)
        elif _OS == "Darwin":
            _run_quiet(["pbcopy"], text_input=text)
        else:
            for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"], ["wl-copy"]):
                try:
                    _run_quiet(cmd, text_input=text)
                    break
                except FileNotFoundError:
                    continue
            else:
                return False, "no clipboard tool available"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


@tool(
    "clipboard_get",
    "Read the text currently in the system clipboard.",
    {"type": "object", "properties": {}},
)
def clipboard_get(ctx: ToolContext) -> str:
    ok, val = _clipboard_get()
    if not ok:
        return f"❌ Could not read the clipboard: {val}"
    return f"Clipboard:\n{val}" if val else "(clipboard is empty)"


@tool(
    "clipboard_set",
    "Write text to the system clipboard.",
    {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to copy to the clipboard."}},
        "required": ["text"],
    },
)
def clipboard_set(ctx: ToolContext, text: str) -> str:
    ok, err = _clipboard_set(text)
    return "✓ Copied to the clipboard." if ok else f"❌ Could not write to the clipboard: {err}"


TOOLS = [
    open_url, open_path, open_application, search_files, list_directory,
    read_file, write_file, edit_file, run_command, run_powershell, system_info,
    get_datetime, clipboard_get, clipboard_set,
]
