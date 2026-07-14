"""Mappa compatta della struttura di un progetto.

Per ogni file sorgente elenca le definizioni di primo livello (funzioni, classi e
relative firme). Serve a orientarsi a basso costo — una sola chiamata invece di
molte `list_directory`/`grep`/`read_file` — restando una vista ADVISORY: il modello
legge comunque col `read_file` i file che gli servono.

Proprietà volute:
- sempre generata FRESCA dai file attuali → mai stale, non può fuorviare;
- confinata alla radice (usa lo stesso `resolve`/`display` degli altri tool);
- robusta: non solleva mai; su un file illeggibile o malformato passa oltre;
- limitata: tetto su file analizzati e dimensione dell'output (token-economica).

Copertura linguaggi (best-effort, è una MAPPA, non un parser): Python è analizzato
con `ast` (accurato). Per gli altri si usano regex per-linguaggio ancorate a inizio
riga, così si intercettano le DICHIARAZIONI e non gli usi a metà riga. Sono coperti
i linguaggi mainstream — JS/TS, Go, Rust, Java, C#, C/C++, Ruby, PHP, Swift, Kotlin,
Scala, shell, Lua, Dart, Elixir, Perl, R, Julia, Zig, Nim, Clojure, Haskell, SQL — e
per estensioni non riconosciute non si emette nulla (niente simboli inventati).
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from .fs import _BINARY_EXT, NOISE_DIRS, _trunc, display, resolve

_MAX_FILES = 800              # tetto di sicurezza sul numero di file analizzati
_MAX_FILE_BYTES = 1_000_000   # non aprire file enormi (probabili dati/minificati)
_MAX_SYMBOLS = 80             # tetto di simboli per file
_MAX_METHODS = 14             # metodi mostrati per classe Python (poi "…")

M = re.MULTILINE


def _compact_args(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:60] + "…") if len(s) > 60 else s


# ── Python: AST (accurato) ────────────────────────────────────────────────────
_PY_EXT = {".py", ".pyi"}


def _py_args(a: ast.arguments) -> str:
    parts: list[str] = []
    positional = list(a.posonlyargs) + list(a.args)
    ndef = len(a.defaults)
    for i, arg in enumerate(positional):
        parts.append(f"{arg.arg}=…" if i >= len(positional) - ndef else arg.arg)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(a.kwonlyargs):
        parts.append(arg.arg if a.kw_defaults[i] is None else f"{arg.arg}=…")
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return ", ".join(parts)


def _py_symbols(text: str) -> list[str] | None:
    """Simboli di primo livello via AST; None se non è Python valido (il chiamante
    ripiega sulla scansione generica)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    syms: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            syms.append(f"def {node.name}({_py_args(node.args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            label = f"class {node.name}"
            if methods:
                label += ": " + ", ".join(methods[:_MAX_METHODS]) + ("…" if len(methods) > _MAX_METHODS else "")
            syms.append(label)
    return syms


# ── Altri linguaggi: regex per-linguaggio ─────────────────────────────────────
# Ogni regex usa gruppi nominati: (?P<name>…) obbligatorio, (?P<kind>…) e (?P<args>…)
# opzionali. Ancorate a inizio riga (re.M). Negli alternation, le parole più lunghe
# vanno PRIMA (es. defmodule prima di def) per non catturare il prefisso.

_RULES: dict[str, list[re.Pattern]] = {
    "js": [
        re.compile(r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?P<kind>class)\s+(?P<name>\w+)", M),
        re.compile(r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?P<kind>function)\s*\*?\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)", M),
        re.compile(r"^[ \t]*(?:export\s+)?(?:default\s+)?(?P<kind>const|let)\s+(?P<name>\w+)\s*=\s*(?:async\s*)?\((?P<args>[^)]*)\)\s*=>", M),
        re.compile(r"^[ \t]*(?:export\s+)?(?P<kind>interface|enum|type)\s+(?P<name>\w+)", M),
    ],
    "go": [
        re.compile(r"^[ \t]*(?P<kind>func)\s+(?:\([^)]*\)\s*)?(?P<name>\w+)\s*\((?P<args>[^)]*)\)", M),
        re.compile(r"^[ \t]*(?P<kind>type)\s+(?P<name>\w+)\s+\S", M),
    ],
    "rust": [
        re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?(?P<kind>fn)\s+(?P<name>\w+)\s*(?:<[^>]*>)?\s*\((?P<args>[^)]*)\)", M),
        re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?P<kind>struct|enum|trait|union|mod|type)\s+(?P<name>\w+)", M),
    ],
    "java": [
        re.compile(r"^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*(?:(?:public|private|protected|abstract|final|static|sealed)\s+)*(?P<kind>class|interface|enum|record)\s+(?P<name>\w+)", M),
        re.compile(r"^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*(?:public|private|protected)\s+(?:(?:static|final|abstract|synchronized|native|default)\s+)*[\w.$<>\[\],\s]+?\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*(?:throws[\w\s,.]+)?[{;]", M),
    ],
    "csharp": [
        re.compile(r"^[ \t]*(?:\[[^\]]*\]\s*)*(?:(?:public|private|protected|internal|abstract|sealed|static|partial)\s+)*(?P<kind>class|struct|interface|enum|record)\s+(?P<name>\w+)", M),
        re.compile(r"^[ \t]*(?:\[[^\]]*\]\s*)*(?P<kind>namespace)\s+(?P<name>[\w.]+)", M),
        re.compile(r"^[ \t]*(?:(?:public|private|protected|internal)\s+)(?:(?:static|virtual|override|abstract|async|sealed|extern)\s+)*[\w.<>\[\],\s]+?\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)", M),
    ],
    "ruby": [
        re.compile(r"^[ \t]*(?P<kind>def|class|module)\s+(?P<name>[\w.:?!=]+)", M),
    ],
    "php": [
        re.compile(r"^[ \t]*(?:(?:public|private|protected|static|final|abstract)\s+)*(?P<kind>function)\s+&?\s*(?P<name>\w+)\s*\((?P<args>[^)]*)\)", M),
        re.compile(r"^[ \t]*(?:(?:abstract|final)\s+)*(?P<kind>class|interface|trait|enum)\s+(?P<name>\w+)", M),
    ],
    "swift": [
        re.compile(r"^[ \t]*(?:(?:public|private|internal|fileprivate|open|static|class|final|override|mutating)\s+|@\w+\s+)*(?P<kind>func)\s+(?P<name>\w+)\s*(?:<[^>]*>)?\s*\((?P<args>[^)]*)\)", M),
        re.compile(r"^[ \t]*(?:(?:public|private|internal|fileprivate|open|final)\s+|@\w+\s+)*(?P<kind>class|struct|enum|protocol|extension|actor)\s+(?P<name>\w+)", M),
    ],
    "kotlin": [
        re.compile(r"^[ \t]*(?:(?:public|private|internal|protected|open|override|suspend|inline|operator|abstract|final|external|tailrec)\s+)*(?P<kind>fun)\s+(?:<[^>]*>\s*)?(?:[\w.]+\.)?(?P<name>\w+)\s*\((?P<args>[^)]*)\)", M),
        re.compile(r"^[ \t]*(?:(?:public|private|internal|protected|sealed|open|abstract|data|enum|annotation|inner|final|value)\s+)*(?P<kind>class|object|interface)\s+(?P<name>\w+)", M),
    ],
    "scala": [
        re.compile(r"^[ \t]*(?:(?:private|protected|final|override|implicit|sealed|abstract|case|lazy)\s+)*(?P<kind>def|class|object|trait)\s+(?P<name>\w+)", M),
    ],
    "shell": [
        re.compile(r"^[ \t]*(?P<kind>function)\s+(?P<name>[\w\-]+)", M),
        re.compile(r"^[ \t]*(?P<name>[\w\-]+)\s*\(\)\s*\{", M),
    ],
    "lua": [
        re.compile(r"^[ \t]*(?:local\s+)?(?P<kind>function)\s+(?P<name>[\w.:]+)\s*\((?P<args>[^)]*)\)", M),
    ],
    "dart": [
        re.compile(r"^[ \t]*(?:abstract\s+)?(?P<kind>class|mixin|enum|extension)\s+(?P<name>\w+)", M),
    ],
    "elixir": [
        re.compile(r"^[ \t]*(?P<kind>defmodule|defprotocol|defimpl|defstruct|defmacro|defp|def)\s+(?P<name>[\w.?!]+)", M),
    ],
    "perl": [
        re.compile(r"^[ \t]*(?P<kind>sub|package)\s+(?P<name>[\w:]+)", M),
    ],
    "r": [
        re.compile(r"^[ \t]*(?P<name>[\w.]+)\s*(?:<-|=)\s*function\s*\((?P<args>[^)]*)\)", M),
    ],
    "julia": [
        re.compile(r"^[ \t]*(?P<kind>function|macro|module|struct|primitive\s+type|abstract\s+type)\s+(?P<name>\w+)", M),
        re.compile(r"^[ \t]*(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*=(?!=)", M),
    ],
    "zig": [
        re.compile(r"^[ \t]*(?:pub\s+)?(?:export\s+)?(?:extern\s+)?(?P<kind>fn)\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)", M),
        re.compile(r"^[ \t]*(?:pub\s+)?(?P<kind>const)\s+(?P<name>\w+)\s*=\s*(?:packed\s+|extern\s+)?(?:struct|enum|union)\b", M),
    ],
    "nim": [
        re.compile(r"^[ \t]*(?P<kind>proc|func|method|iterator|template|macro|converter)\s+(?P<name>[\w`]+)\s*\*?\s*\((?P<args>[^)]*)\)", M),
        re.compile(r"^[ \t]*(?:type\s+)?(?P<name>\w+)\*?\s*=\s*(?:ref\s+)?(?:object|enum|tuple|distinct)\b", M),
    ],
    "clojure": [
        re.compile(r"^[ \t]*\(\s*(?P<kind>defn-|defn|defmacro|defmulti|defmethod|defprotocol|defrecord|deftype|def)\s+(?P<name>[\w\-?!*+<>=/.]+)", M),
    ],
    "haskell": [
        re.compile(r"^(?P<name>[a-z][\w']*)\s*::", M),
        re.compile(r"^[ \t]*(?P<kind>data|newtype|class|instance|type)\s+(?P<name>[\w]+)", M),
    ],
    "sql": [
        re.compile(r"(?i)^[ \t]*create\s+(?:or\s+replace\s+)?(?:if\s+not\s+exists\s+)?(?P<kind>table|view|function|procedure|trigger|index|materialized\s+view)\s+(?:if\s+not\s+exists\s+)?[`\"\[]?(?P<name>[\w.]+)", M),
    ],
}

_EXT: dict[str, str] = {
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js", ".ts": "js", ".tsx": "js", ".mts": "js", ".cts": "js",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".rb": "ruby", ".rake": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".sc": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ksh": "shell",
    ".lua": "lua",
    ".dart": "dart",
    ".ex": "elixir", ".exs": "elixir",
    ".pl": "perl", ".pm": "perl", ".t": "perl",
    ".r": "r", ".jl": "julia",
    ".zig": "zig",
    ".nim": "nim", ".nims": "nim",
    ".clj": "clojure", ".cljs": "clojure", ".cljc": "clojure", ".edn": "clojure",
    ".hs": "haskell",
    ".sql": "sql",
}

# ── C / C++ : gestione dedicata (le funzioni non hanno keyword introduttiva) ───
_C_EXT = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".c++", ".h++", ".ino", ".m", ".mm"}
_C_KW = {
    "if", "for", "while", "switch", "return", "else", "catch", "sizeof", "do",
    "case", "typedef", "struct", "class", "enum", "union", "namespace", "template",
    "using", "static_assert", "decltype", "alignof", "new", "delete", "throw",
    "goto", "and", "or", "not", "constexpr",
}
_C_TYPE = re.compile(r"^[ \t]*(?:typedef\s+)?(?P<kind>struct|class|enum|union|namespace)\s+(?P<name>\w+)", M)
_C_FUNC = re.compile(
    r"^[ \t]*(?P<ret>[A-Za-z_][\w\s:<>,\*&]*?[\s\*&])(?P<name>[A-Za-z_]\w*)\s*"
    r"\((?P<args>[^;{)]*(?:\)[^;{)]*)*)\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:->\s*[\w\s:<>,\*&]+)?\{",
    M,
)
_C_PROTO = re.compile(
    r"^[ \t]*(?P<ret>[A-Za-z_][\w\s:<>,\*&]*?[\s\*&])(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^;{)]*)\)\s*;",
    M,
)


def _c_symbols(text: str) -> list[str]:
    found: list[tuple[int, str]] = [
        (m.start(), f"{m.group('kind')} {m.group('name')}") for m in _C_TYPE.finditer(text)
    ]
    for rx in (_C_FUNC, _C_PROTO):
        for m in rx.finditer(text):
            ret = m.group("ret").split()
            name = m.group("name")
            if not ret or name in _C_KW or ret[0] in _C_KW or ret[-1] in _C_KW:
                continue
            found.append((m.start(), f"{name}({_compact_args(m.group('args'))})"))
    return _dedup_ordered(found)


# Scansione generica (fallback per Python non valido o estensioni non in tabella).
_GENERIC = [
    re.compile(
        r"^[ \t]*(?:(?:pub|public|private|protected|internal|export|default|static|final|"
        r"abstract|async|inline|extern|virtual|override)\s+)*"
        r"(?P<kind>func|function|fn|def|proc|class|struct|interface|trait|enum|impl|module|object|type)\s+(?P<name>\w+)", M),
]

_MAPPED = _PY_EXT | _C_EXT | set(_EXT)


def _dedup_ordered(found: list[tuple[int, str]]) -> list[str]:
    found.sort(key=lambda t: t[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, lbl in found:
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def _scan(text: str, patterns: list[re.Pattern]) -> list[str]:
    found: list[tuple[int, str]] = []
    for rx in patterns:
        for m in rx.finditer(text):
            gd = m.groupdict()
            name = gd.get("name")
            if not name:
                continue
            kind = (gd.get("kind") or "").strip()
            label = f"{kind} {name}" if kind else name
            if gd.get("args") is not None:
                label += f"({_compact_args(gd['args'])})"
            found.append((m.start(), label))
    return _dedup_ordered(found)


def _symbols_for(path: Path, text: str) -> list[str]:
    ext = path.suffix.lower()
    if ext in _PY_EXT:
        syms = _py_symbols(text)
        return syms if syms is not None else _scan(text, _GENERIC)
    if ext in _C_EXT:
        return _c_symbols(text)
    key = _EXT.get(ext)
    if key:
        return _scan(text, _RULES[key])
    return _scan(text, _GENERIC)


def _iter_files(base: Path):
    if base.is_file():
        yield base
        return
    for dirpath, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in NOISE_DIRS]
        for name in sorted(files):
            yield Path(dirpath) / name


def build_repo_map(root: Path | None, rel_path: str, max_chars: int) -> str:
    """Outline compatto sotto `rel_path` (relativo alla radice). Non solleva mai."""
    base = resolve(root, rel_path)
    blocks: list[str] = []
    scanned = 0
    capped = False
    for full in _iter_files(base):
        if scanned >= _MAX_FILES:
            capped = True
            break
        ext = full.suffix.lower()
        if ext in _BINARY_EXT or ext not in _MAPPED:
            continue
        try:
            if full.stat().st_size > _MAX_FILE_BYTES:
                continue
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        syms = _symbols_for(full, text)
        if not syms:
            continue
        shown = syms[:_MAX_SYMBOLS]
        if len(syms) > _MAX_SYMBOLS:
            shown.append(f"… (+{len(syms) - _MAX_SYMBOLS} more)")
        body = "\n".join("    " + s for s in shown)
        blocks.append(f"{display(root, full)}\n{body}")

    if not blocks:
        return (f"No mappable source definitions under {display(root, base)}. "
                "Use list_directory/glob to see the files.")

    header = f"Project map ({len(blocks)} files with definitions):\n\n"
    note = "\n\n[Scansione limitata ai primi file per sicurezza.]" if capped else ""
    return _trunc(header + "\n\n".join(blocks) + note, max_chars,
                  hint="mappa una sottocartella con `path` per restringere")
