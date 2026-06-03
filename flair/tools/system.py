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
    "Apre un URL nel browser predefinito. Usa questo per 'apri il browser' o per aprire un sito.",
    {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "URL da aprire (https://...). Per aprire solo il browser usa about:blank."}},
        "required": ["url"],
    },
)
def open_url(ctx: ToolContext, url: str) -> str:
    if not (url.startswith("http://") or url.startswith("https://") or url.startswith("about:")):
        url = "https://" + url
    try:
        ok = webbrowser.open(url)
        return f"✓ Aperto nel browser: {url}" if ok else f"⚠️ Non sono riuscito ad aprire il browser per {url}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Errore aprendo l'URL: {exc}"


# ── open_path ────────────────────────────────────────────────────────────────

@tool(
    "open_path",
    "Apre un file o una cartella con l'applicazione predefinita del sistema (es. una canzone nel player, un PDF nel lettore, una cartella nel file manager).",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path del file o della cartella (assoluto o ~)."}},
        "required": ["path"],
    },
)
def open_path(ctx: ToolContext, path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"❌ Il path non esiste: {p}"
    try:
        if _OS == "Windows":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif _OS == "Darwin":
            _run_quiet(["open", str(p)])
        else:
            _run_quiet(["xdg-open", str(p)])
        return f"✓ Aperto: {p}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Errore aprendo il path: {exc}"


# ── open_application ─────────────────────────────────────────────────────────

@tool(
    "open_application",
    "Avvia un'applicazione per nome (es. 'chrome', 'notepad', 'code', 'spotify'). Best-effort cross-platform.",
    {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Nome o eseguibile dell'applicazione."}},
        "required": ["name"],
    },
)
def open_application(ctx: ToolContext, name: str) -> str:
    try:
        if _OS == "Windows":
            # 'start' risolve gli eseguibili via PATH e App Paths del registro.
            subprocess.Popen(["cmd", "/c", "start", "", name], shell=False)
            return f"✓ Avvio richiesto: {name}"
        if _OS == "Darwin":
            r = _run_quiet(["open", "-a", name])
            return f"✓ Avviato: {name}" if r.returncode == 0 else f"❌ App non trovata: {name} ({r.stderr.strip()})"
        # Linux: prova l'eseguibile diretto, poi gtk-launch.
        try:
            subprocess.Popen([name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"✓ Avviato: {name}"
        except FileNotFoundError:
            r = _run_quiet(["gtk-launch", name])
            if r.returncode == 0:
                return f"✓ Avviato: {name}"
            return f"❌ Applicazione non trovata: {name}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Errore avviando l'applicazione: {exc}"


# ── search_files ─────────────────────────────────────────────────────────────

@tool(
    "search_files",
    ("Cerca file sul computer per nome ed eventuale estensione. Di default cerca nelle "
     "cartelle utente (Musica, Download, Desktop, Documenti, Video, Immagini). Usa "
     "'locations' per cercare altrove. Ottimo per 'trovami una canzone', 'dov'è quel PDF'."),
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Testo da cercare nel nome del file (case-insensitive). Vuoto = qualsiasi nome."},
            "extensions": {"type": "array", "items": {"type": "string"}, "description": "Estensioni da filtrare, es. ['.mp3', '.flac']. Opzionale."},
            "locations": {"type": "array", "items": {"type": "string"}, "description": "Cartelle in cui cercare. Opzionale (default: cartelle utente)."},
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
        return "❌ Nessuna delle cartelle indicate esiste."

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
        return f"Nessun file trovato (query='{query}', estensioni={sorted(exts) or 'tutte'}) in: {where}"
    out = f"{len(results)} file trovati:\n" + "\n".join(results)
    if truncated:
        out += "\n...[ricerca interrotta al limite; restringi query/estensioni o indica 'locations']"
    return out


# ── list_directory (whole machine) ───────────────────────────────────────────

@tool(
    "list_directory",
    "Elenca il contenuto di una cartella qualsiasi del computer (un livello).",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Cartella (assoluta o ~). Default: home."}},
    },
)
def list_directory(ctx: ToolContext, path: str = "~") -> str:
    return fs.list_dir_impl(None, path, ctx.cfg.list_dir_max_entries)


# ── read_file (whole machine) ────────────────────────────────────────────────

@tool(
    "read_file",
    "Legge un file di testo qualsiasi del computer, con numeri di riga.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path del file (assoluto o ~)."},
            "offset": {"type": "integer", "description": "Prima riga (1-based). Default 1."},
            "limit": {"type": "integer", "description": "Numero massimo di righe."},
        },
        "required": ["path"],
    },
)
def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int | None = None) -> str:
    return fs.read_file_impl(None, path, offset, limit, ctx.cfg.read_file_max_chars)


# ── write_file / edit_file (qualsiasi percorso del computer) ──────────────────

@tool(
    "write_file",
    ("Crea o sovrascrive un file di testo qualsiasi del computer col contenuto fornito "
     "(crea anche le cartelle mancanti). USA QUESTO per creare un file (es. un report), "
     "non run_command: è diretto e affidabile, niente problemi di shell/escaping."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path del file (assoluto o ~)."},
            "content": {"type": "string", "description": "Contenuto completo del file."},
        },
        "required": ["path", "content"],
    },
    destructive=True,
)
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    p = fs.resolve(None, path)
    if p.is_dir():
        return f"❌ È una directory: {p}"
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    verbo = "Sovrascritto" if existed else "Creato"
    return f"✓ {verbo} {p} ({len(content)} caratteri)."


@tool(
    "edit_file",
    ("Sostituisce una porzione esatta di testo in un file qualsiasi del computer "
     "(match resiliente a spazi/indentazione). Per piccole modifiche mirate; per "
     "creare o riscrivere un intero file usa write_file."),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path del file (assoluto o ~)."},
            "old_string": {"type": "string", "description": "Testo esatto da sostituire."},
            "new_string": {"type": "string", "description": "Testo nuovo."},
            "replace_all": {"type": "boolean", "description": "Sostituire tutte le occorrenze (default false)."},
        },
        "required": ["path", "old_string", "new_string"],
    },
    destructive=True,
)
def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = fs.resolve(None, path)
    if not p.exists():
        return f"❌ Il file non esiste: {p} (usa write_file per crearlo)"
    if p.is_dir():
        return f"❌ È una directory: {p}"
    text = p.read_text(encoding="utf-8", errors="replace")
    new_text, strategy = fs.apply_edit(text, old_string, new_string, replace_all)
    if new_text == text:
        return f"⚠️ Nessuna modifica: il risultato è identico a {p}."
    p.write_text(new_text, encoding="utf-8")
    note = "" if strategy == "esatto" else f" [match: {strategy}]"
    return f"✓ Modificato {p}{note}."


# ── run_command ──────────────────────────────────────────────────────────────

@tool(
    "run_command",
    "Esegue un comando nella shell di sistema (cmd su Windows, sh su Unix). Per operazioni di sistema avanzate. Ritorna stdout+stderr ed exit code.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Comando da eseguire."},
            "timeout": {"type": "integer", "description": "Timeout in secondi. Default 60."},
        },
        "required": ["command"],
    },
    destructive=True,
)
def run_command(ctx: ToolContext, command: str, timeout: int = 60) -> str:
    try:
        proc = shell.run_shell(command, timeout)
    except subprocess.TimeoutExpired:
        return f"❌ Comando andato in timeout dopo {timeout}s: {command}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Errore eseguendo il comando: {exc}"
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    return fs._trunc(f"$ {command}\n(exit code {proc.returncode})\n" + out.strip(),
                     ctx.cfg.command_max_chars, hint="filtra l'output")


# ── run_powershell (script su file temporaneo, pulizia garantita) ─────────────

@tool(
    "run_powershell",
    ("Esegue uno script PowerShell, anche su PIÙ RIGHE (here-string, Add-Type, più "
     "istruzioni). flair lo scrive in un file temporaneo, lo esegue e lo CANCELLA sempre "
     "(anche in caso di errore o timeout). USA QUESTO per PowerShell complesso invece di "
     "passarlo inline a run_command: così eviti i problemi di escaping e di a-capo."),
    {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "Lo script PowerShell (PowerShell normale, senza escaping da shell)."},
            "timeout": {"type": "integer", "description": "Timeout in secondi. Default 60."},
        },
        "required": ["script"],
    },
    destructive=True,
)
def run_powershell(ctx: ToolContext, script: str, timeout: int = 60) -> str:
    try:
        proc = shell.run_powershell_script(script, timeout)
    except subprocess.TimeoutExpired:
        return f"❌ Script PowerShell andato in timeout dopo {timeout}s."
    except FileNotFoundError:
        return "❌ PowerShell non trovato su questo sistema."
    except Exception as exc:  # noqa: BLE001
        return f"❌ Errore eseguendo lo script PowerShell: {exc}"
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    return fs._trunc(f"(exit code {proc.returncode})\n" + out.strip(),
                     ctx.cfg.command_max_chars, hint="filtra l'output")


# ── system_info ──────────────────────────────────────────────────────────────

@tool(
    "system_info",
    "Restituisce informazioni sul sistema: OS, versione, architettura, CPU, RAM (se disponibile), hostname.",
    {"type": "object", "properties": {}},
)
def system_info(ctx: ToolContext) -> str:
    lines = [
        f"OS: {platform.system()} {platform.release()} ({platform.version()})",
        f"Architettura: {platform.machine()}",
        f"Processore: {platform.processor() or 'n/d'}",
        f"CPU logiche: {os.cpu_count()}",
        f"Hostname: {platform.node()}",
        f"Python: {platform.python_version()} ({sys.executable})",
    ]
    try:
        import psutil  # opzionale

        vm = psutil.virtual_memory()
        lines.append(f"RAM: {vm.used // (1024**2)} MB usati / {vm.total // (1024**2)} MB ({vm.percent}%)")
        lines.append(f"Uso CPU: {psutil.cpu_percent(interval=0.2)}%")
    except Exception:  # noqa: BLE001
        lines.append("RAM/CPU%: (installa 'psutil' per i dettagli)")
    return "\n".join(lines)


# ── get_datetime ─────────────────────────────────────────────────────────────

@tool(
    "get_datetime",
    "Restituisce data e ora correnti del sistema. Usalo invece di indovinare la data.",
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
                return False, "nessuno strumento clipboard (installa xclip/xsel/wl-clipboard o pyperclip)"
        if r.returncode == 0:
            return True, r.stdout
        return False, r.stderr.strip() or "errore clipboard"
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
                return False, "nessuno strumento clipboard disponibile"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


@tool(
    "clipboard_get",
    "Legge il testo attualmente negli appunti di sistema.",
    {"type": "object", "properties": {}},
)
def clipboard_get(ctx: ToolContext) -> str:
    ok, val = _clipboard_get()
    if not ok:
        return f"❌ Impossibile leggere gli appunti: {val}"
    return f"Appunti:\n{val}" if val else "(appunti vuoti)"


@tool(
    "clipboard_set",
    "Scrive del testo negli appunti di sistema.",
    {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Testo da copiare negli appunti."}},
        "required": ["text"],
    },
)
def clipboard_set(ctx: ToolContext, text: str) -> str:
    ok, err = _clipboard_set(text)
    return "✓ Copiato negli appunti." if ok else f"❌ Impossibile scrivere negli appunti: {err}"


TOOLS = [
    open_url, open_path, open_application, search_files, list_directory,
    read_file, write_file, edit_file, run_command, run_powershell, system_info,
    get_datetime, clipboard_get, clipboard_set,
]
