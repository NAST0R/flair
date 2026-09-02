#!/usr/bin/env python3
"""flair .env configurator — a standalone GUI editor for the flair .env file.

No third-party dependencies: Python stdlib + Tkinter only (imported lazily, so
`--check` and the pure parse/render/validate functions work without tkinter).

Usage
  python configurer.py [path/to/.env]
  python configurer.py --check [path/to/.env]     # headless validation report
"""
from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Tkinter is imported lazily in _gui() / main() so the pure functions below
# (parse_env, render_env, validate, --check) stay usable without python3-tk.

tk: Any = None
ttk: Any = None
filedialog: Any = None
messagebox: Any = None

UNSET = "(unset)"
DASH = "\u2500"   # used in f-string expressions (a backslash there is a SyntaxError before 3.12)
HEADER_RE = re.compile(r"^#\s*\u2500+\s*(.*?)\s*\u2500+\s*$")
ACTIVE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
# Commented-out entry: '# KEY=value' where the value is a plain token (no
# spaces/punctuation like '|') running to end-of-line or a trailing comment.
# Anything else mentioning KEY=... is prose (e.g. '# FLAIR_DEEPSEEK_FIRST_PARTY=
# true|false (default: auto)') and is never touched.
COMMENTED_RE = re.compile(r"^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_./~:\\-]*)(\s+#.*)?$")

# Section name in the GUI -> title text of the "── ... ──" header in the file.
SECTION_HEADERS = {
    "Provider": "Active provider",
    "DeepSeek": "DeepSeek",
    "OpenAI": "OpenAI",
    "Local": "Local inference",
    "Generation": "Generation / loop",
    "Vision": "Vision",
    "TLS": "TLS",
    "Context": "Context management",
    "Filesystem": "Filesystem",
    "Web": "Web search",
    "Observability": "Observability",
    "Sessions": "Sessions",
    "Safety": "Safety",
    "Pricing": "Cost estimate",
}


# Parameter catalog (mirrors .env.example + flair/config.py)

@dataclass(frozen=True)
class Field:
    section: str
    key: str
    kind: str = "str"              # str | secret | int | float | bool | choice | path
    default: str | None = None     # effective value when the key is absent from the file
    help: str = ""                 # 5th positional: the GUI help text for this field
    # kind == "choice" only — ALWAYS pass it by keyword, never positionally, so it
    # can never swap places with `help` (that silently emptied every help string).
    choices: tuple[str, ...] | None = None
    min: float | None = None       # numeric bounds for kind int/float (validated)
    max: float | None = None
    pick: str | None = None        # "file" | "dir" for kind == "path"


CATALOG: tuple[Field, ...] = (
    # ── Active provider ──────────────────────────────────────────────────────
    Field("Provider", "FLAIR_PROVIDER", "choice", "deepseek",
          "Active provider: which slot flair talks to — 'deepseek', 'openai' or 'local'.", choices=("deepseek", "openai", "local")),
    Field("Provider", "FLAIR_DEEPSEEK_FIRST_PARTY", "choice", "auto",
          "Force the DeepSeek protocol profile. auto (default) = deduced from "
          "DEEPSEEK_BASE_URL: api.deepseek.com speaks the first-party protocol "
          "(reasoning passback, thinking control, peak/off-peak billing); any "
          "other host gets the standard OpenAI-compat profile.", choices=("auto", "true", "false")),

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    Field("DeepSeek", "DEEPSEEK_API_KEY", "secret", None,
          "API key for the DeepSeek endpoint. Required when FLAIR_PROVIDER=deepseek."),
    Field("DeepSeek", "DEEPSEEK_BASE_URL", "str", "https://api.deepseek.com",
          "DeepSeek API endpoint — or any OpenAI-compatible host serving DeepSeek "
          "weights (OpenRouter, DeepInfra, ...)."),
    Field("DeepSeek", "DEEPSEEK_MODEL", "str", "deepseek-v4-flash",
          "Fast model for the tool loop (non-thinking). V4: flash = the workhorse."),
    Field("DeepSeek", "DEEPSEEK_THINK_MODEL", "str", "deepseek-v4-pro",
          "Thinking model used with --think. V4: pro = the reasoner."),
    Field("DeepSeek", "DEEPSEEK_TEMPERATURE", "float", "0.0",
          "Sampling temperature for DeepSeek requests.", min=0.0),
    Field("DeepSeek", "DEEPSEEK_REASONING_EFFORT", "choice", None,
          "Thinking depth with --think: 'high' (server default) or 'max' (recommended; "
          "field-measured cheapest AND most precise). The API maps low/medium to high "
          "and xhigh to max. Unset = server default.", choices=("low", "medium", "high", "xhigh", "max")),
    Field("DeepSeek", "DEEPSEEK_FAST_REASONING_EFFORT", "choice", None,
          "Opt-in: reasoning depth of the FAST model's default thinking (V4 already "
          "thinks at 'high' server-side). 'max' = deep reasoning at flash prices on "
          "every step of the regular loop. Unset = requests unchanged.", choices=("low", "medium", "high", "xhigh", "max")),
    Field("DeepSeek", "DEEPSEEK_VISION", "bool", "false",
          "Enable ONLY if the endpoint really accepts images (multipart OpenAI-style "
          "content). Enables the view_image tool and /img <path>."),

    # ── OpenAI ───────────────────────────────────────────────────────────────
    Field("OpenAI", "OPENAI_API_KEY", "secret", None,
          "OpenAI API key. Required when FLAIR_PROVIDER=openai."),
    Field("OpenAI", "OPENAI_BASE_URL", "str", None,
          "Optional: proxy / Azure-compatible endpoint."),
    Field("OpenAI", "OPENAI_MODEL", "str", "gpt-4.1-mini",
          "Fast model for the tool loop (~$0.40/$1.60 per 1M)."),
    Field("OpenAI", "OPENAI_THINK_MODEL", "str", "gpt-5-mini",
          "Thinking model for --think (~$0.25/$2.00 per 1M). Stay on gpt-5 / "
          "gpt-5-mini / gpt-5.1 or o3: gpt-5.4+ supports reasoning+tools only on "
          "the Responses API, not the Chat Completions API used here."),
    Field("OpenAI", "OPENAI_TEMPERATURE", "float", "0.0",
          "Sampling temperature for OpenAI requests.", min=0.0),
    Field("OpenAI", "OPENAI_REASONING_EFFORT", "choice", None,
          "Override reasoning_effort (with --think it is set to 'medium' "
          "automatically; valid values are model-dependent).", choices=("none", "minimal", "low", "medium", "high", "xhigh")),
    Field("OpenAI", "OPENAI_VISION", "bool", "false",
          "Enable ONLY if the endpoint accepts images. Enables the view_image tool."),

    # ── Local inference ──────────────────────────────────────────────────────
    Field("Local", "LOCAL_BASE_URL", "str", "http://127.0.0.1:8001/v1",
          "OpenAI-compatible local server (llama.cpp llama-server, vLLM, LM Studio, "
          "...). Set FLAIR_PROVIDER=local to use it."),
    Field("Local", "LOCAL_MODEL", "str", "local",
          "Display name of the local model (llama-server ignores it)."),
    Field("Local", "LOCAL_THINK_MODEL", "str", None,
          "Thinking model for the local slot. Unset = same as LOCAL_MODEL (one "
          "model in VRAM)."),
    Field("Local", "LOCAL_API_KEY", "str", "local",
          "Not verified by llama-server; some servers require a non-empty value."),
    Field("Local", "LOCAL_TEMPERATURE", "float", None,
          "Unset = the SERVER's sampling wins (recommended).", min=0.0),
    Field("Local", "LOCAL_VISION", "bool", "false",
          "Enable ONLY if the server was started with its mmproj projector loaded."),

    # ── Generation / loop ────────────────────────────────────────────────────
    Field("Generation", "FLAIR_MAX_TOKENS", "int", "8000",
          "Max output tokens per request. Raise it if max-effort reasoning chains "
          "get truncated (32000 is a safe working value for hard tasks).", min=1),
    Field("Generation", "FLAIR_TIMEOUT", "int", "300",
          "HTTP request timeout in seconds.", min=1),
    Field("Generation", "FLAIR_MAX_STEPS", "int", "60",
          "Maximum agent steps per turn.", min=1),
    Field("Generation", "FLAIR_THINK_STEPS", "choice", "first",
          "With --think: 'first' = thinking model only for the turn's opening step, "
          "then the loop continues on the fast model; 'all' = thinking model for "
          "EVERY step (deep reasoning throughout, ~3x pro cost).", choices=("first", "all")),
    Field("Generation", "FLAIR_EXPLORER_MAX_STEPS", "int", "20",
          "Step cap for the explorer sub-agent (explore tool, read-only).", min=1),
    Field("Generation", "FLAIR_PARALLEL_TOOLS", "bool", "true",
          "Run batches of non-destructive tool calls in parallel (lower latency, "
          "identical output). false = old sequential behavior."),
    Field("Generation", "FLAIR_PARALLEL_TOOLS_MAX", "int", "8",
          "Thread cap per parallel batch (avoids flooding the network).", min=1),
    Field("Generation", "FLAIR_STREAM", "bool", "true",
          "Real-time response streaming. --no-stream disables it per run."),

    # ── Vision ───────────────────────────────────────────────────────────────
    Field("Vision", "FLAIR_IMAGE_MAX_SIDE", "int", "1536",
          "Images larger than this (longest side, px) are downscaled before sending "
          "if Pillow is installed; without it, originals are sent as-is.", min=1),
    Field("Vision", "FLAIR_IMAGE_TOKENS", "int", "1200",
          "Estimated tokens per image for the context meter (drives proactive "
          "compaction; the real count comes back in the API usage).", min=1),
    Field("Context", "FLAIR_CTX_CALIBRATION", "bool", "true",
          "Learn the real chars-to-tokens ratio from the requests and correct the "
          "estimate of the not-yet-sent suffix with it (dense code tokenizes worse "
          "than prose). false = static chars/4 estimate."),

    # ── TLS ──────────────────────────────────────────────────────────────────
    Field("TLS", "FLAIR_CA_BUNDLE", "path", None,
          "Private CA bundle (PEM) used to verify TLS for ANY provider endpoint: "
          "self-signed llama-server certificates or corporate TLS inspection. "
          "Unset = default trust store (certifi).", pick="file"),

    # ── Context management ───────────────────────────────────────────────────
    Field("Context", "FLAIR_CONTEXT_WINDOW", "int", "120000",
          "Model context window (tokens). 120K sits on the good edge of "
          "DeepSeek-V4's retrieval plateau; raising it trades retrieval quality "
          "for capacity.", min=1),
    Field("Context", "FLAIR_COMPACT_RATIO", "float", "0.75",
          "The context is compacted once it passes this fraction of the window.",
          min=0.0, max=1.0),
    Field("Context", "FLAIR_COMPACT_KEEP", "int", "8",
          "Recent messages kept intact when compacting.", min=0),
    Field("Context", "FLAIR_COMPACT_SUMMARY_MAX", "int", "2000",
          "Token cap for the summary generated during compaction.", min=1),
    Field("Context", "FLAIR_COMPACT_PRUNE", "bool", "true",
          "Compaction stage 0: prune provably-superseded tool outputs (duplicates, "
          "reads of files later overwritten, partial reads covered by a later full "
          "read) BEFORE the LLM summary — for free. true recommended."),
    Field("Context", "FLAIR_PRUNE_HYSTERESIS", "float", "0.10",
          "Stage-0 hysteresis: pruning alone skips the LLM summary only if it lands "
          "the context BELOW threshold by this margin (fraction of the window). "
          "0 = old behavior (any prune-only exit).", min=0.0, max=1.0),

    # ── Filesystem / tools ───────────────────────────────────────────────────
    Field("Filesystem", "FLAIR_ROOT", "path", ".",
          "Working root the agent is sandboxed to. Relative paths are resolved "
          "against the .env location.", pick="dir"),
    Field("Filesystem", "FLAIR_READ_MAX", "int", "12000",
          "Max characters returned by the read_file tool.", min=1),
    Field("Filesystem", "FLAIR_GREP_MAX", "int", "6000",
          "Max characters returned by the grep tool.", min=1),
    Field("Filesystem", "FLAIR_CMD_MAX", "int", "8000",
          "Max characters of command output (run_command).", min=1),
    Field("Filesystem", "FLAIR_REPOMAP_MAX", "int", "8000",
          "Max size (chars) of the project map (repo_map tool).", min=1),
    Field("Filesystem", "FLAIR_LISTDIR_MAX", "int", "200",
          "Max entries returned by list_dir.", min=1),
    Field("Filesystem", "FLAIR_SEARCH_MAX", "int", "80",
          "Max results from the file search tool.", min=1),
    Field("Filesystem", "FLAIR_SEARCH_SCAN_MAX", "int", "200000",
          "Max files scanned by the search tool.", min=1),

    # ── Web search ───────────────────────────────────────────────────────────
    Field("Web", "TAVILY_API_KEY", "secret", None,
          "With a Tavily key, web search uses Tavily (recommended); without, it "
          "falls back to DuckDuckGo (best-effort)."),
    Field("Web", "FLAIR_WEB_MAX", "int", "5",
          "Max results per web search.", min=1),

    # ── Observability ────────────────────────────────────────────────────────
    Field("Observability", "FLAIR_LOG_DIR", "path", None,
          "Folder for the session log (JSONL) and flair.log. Empty = no log.",
          pick="dir"),

    # ── Sessions ─────────────────────────────────────────────────────────────
    Field("Sessions", "FLAIR_SESSION_DIR", "path", "~/.flair/sessions",
          "Where sessions are saved/resumed (--session, --continue, /save, /load).",
          pick="dir"),
    Field("Sessions", "FLAIR_MEMORY", "bool", "true",
          "Session memory: durable facts jotted with the remember tool, injected "
          "into the system prompt (stable prefix, cached) and saved as a "
          "<name>.memory.md sidecar. false = the tool does not exist at all."),
    Field("Sessions", "FLAIR_MEMORY_MAX_CHARS", "int", "4000",
          "HARD cap of the memory block in chars (~1k tokens at 4000); past it, "
          "remember refuses and pruning is a user choice (/memory).", min=1),

    # ── Safety ───────────────────────────────────────────────────────────────
    Field("Safety", "FLAIR_AUTO_APPROVE", "bool", "false",
          "false = ask confirmation for edit/write/run_command (with a diff "
          "preview). true = no confirmation."),
    Field("Safety", "FLAIR_READ_ONLY", "bool", "false",
          "Unattended execution: disables destructive tools entirely (writes/"
          "edits/commands) in both agents. Ideal for read-only scheduled jobs."),
    Field("Safety", "FLAIR_MAX_COST", "float", "0.0",
          "HARD session cost cap (USD): past it, the task stops. 0 = no limit.",
          min=0.0),
    Field("Safety", "FLAIR_COST_WARN", "float", None,
          "Warn when the estimated session cost exceeds this threshold (USD). "
          "0 = disabled.", min=0.0),

    # ── Cost estimate overrides ──────────────────────────────────────────────
    Field("Pricing", "FLAIR_PRICE_CACHE_HIT", "float", None,
          "Override: USD per 1M cache-hit tokens (applies in every band).", min=0.0),
    Field("Pricing", "FLAIR_PRICE_CACHE_MISS", "float", None,
          "Override: USD per 1M input / cache-miss tokens (applies in every band).",
          min=0.0),
    Field("Pricing", "FLAIR_PRICE_OUTPUT", "float", None,
          "Override: USD per 1M output tokens (applies in every band).", min=0.0),
    Field("Pricing", "FLAIR_PRICE_CACHE_HIT_PEAK", "float", None,
          "Override: cache-hit price during DeepSeek peak hours (01-04 and 06-10 UTC).",
          min=0.0),
    Field("Pricing", "FLAIR_PRICE_CACHE_MISS_PEAK", "float", None,
          "Override: input price during peak hours.", min=0.0),
    Field("Pricing", "FLAIR_PRICE_OUTPUT_PEAK", "float", None,
          "Override: output price during peak hours.", min=0.0),
    Field("Background jobs", "FLAIR_BG_MAX_JOBS", "int", "8",
          "How many background commands (run_background) can run at the same time.", min=1, max=64),
    Field("Background jobs", "FLAIR_BG_BUFFER_CHARS", "int", "262144",
          "Output kept per job, in characters. It is a ring: past the cap the oldest "
          "output is dropped and the job says how much.", min=4096),
    Field("Background jobs", "FLAIR_BG_MAX_WAIT", "int", "30",
          "Cap for wait_seconds on job(check): how long a single call may block "
          "waiting for new output instead of polling in a loop.", min=0, max=600),
    Field("Background jobs", "FLAIR_BG_START_GRACE", "float", "1.5",
          "Seconds waited right after starting a job, so a command that dies "
          "immediately reports the error in the same turn.", min=0.0, max=30.0),
    Field("Background jobs", "FLAIR_BG_STOP_GRACE", "float", "3.0",
          "Seconds between the graceful termination of a job's process tree and the "
          "forced kill.", min=0.0, max=60.0),
    Field("Background jobs", "FLAIR_BG_MAX_LIFETIME", "int", "3600",
          "Maximum lifetime of a background job in seconds: past it the job is "
          "terminated at the next interaction. 0 = no limit (not recommended).", min=0),
    Field("Background jobs", "FLAIR_BG_KEEP_FINISHED", "int", "5",
          "How many FINISHED jobs stay in the list. A finished job is still useful "
          "for a moment (its tail gets read after it ends), but keeping them all "
          "makes job(action=\"list\") grow with every long session and holds their "
          "output buffers in memory. Older ones are dropped and counted.", min=0),
    Field("Pricing", "FLAIR_PRICE_CACHE_HIT_THINK", "float", None,
          "Override for requests served by the THINKING model only (highest "
          "precedence). Needed when fast and thinking sit on different third-party "
          "hosts, where one flat set of prices would misprice --think turns.", min=0.0),
    Field("Pricing", "FLAIR_PRICE_CACHE_MISS_THINK", "float", None,
          "Override: input/cache-miss price of the thinking model.", min=0.0),
    Field("Pricing", "FLAIR_PRICE_OUTPUT_THINK", "float", None,
          "Override: output price of the thinking model.", min=0.0),
)

CATALOG_BY_KEY = {f.key: f for f in CATALOG}
SECTIONS = tuple(dict.fromkeys(f.section for f in CATALOG))
BOOL_WORDS = {"1", "true", "yes", "on"}


# .env parsing / rendering (comments and layout are preserved)

def _split_value(rest: str) -> tuple[str, str, str]:
    """Split 'VALUE  # comment' into (value, comment, gap). Follows dotenv rules:
    a '#' starts a comment only at the start of the value or after whitespace.
    `gap` is the original whitespace before the '#' (kept for alignment)."""
    rest = rest.strip()
    if rest.startswith("#"):
        return "", rest[1:].strip(), ""
    m = re.match(r"^(.*?)(\s+)(#\s*.*)$", rest)
    if m:
        return m.group(1), m.group(3)[1:].strip(), m.group(2)
    return rest, "", ""


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_env(text: str) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """Parse .env text. Returns (values, entries) where entries maps each key to
    its line records: {lineno, active, value, comment}."""
    values: dict[str, str] = {}
    entries: dict[str, list[dict]] = {}
    for i, line in enumerate(text.splitlines()):
        m = ACTIVE_RE.match(line)
        if m and not line.lstrip().startswith("#"):
            key, rest = m.group(1), m.group(2)
            value, comment, gap = _split_value(rest)
            values[key] = _unquote(value)
            entries.setdefault(key, []).append(
                {"lineno": i, "active": True, "value": values[key],
                 "comment": comment, "gap": gap})
            continue
        m = COMMENTED_RE.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            comment = (m.group(3) or "").strip().lstrip("#").strip()
            entries.setdefault(key, []).append(
                {"lineno": i, "active": False, "value": value,
                 "comment": comment, "gap": "  "})
    return values, entries


def quote_value(v: str) -> str:
    """Quote a value for writing back (dotenv-compatible)."""
    if v == "":
        return ""
    if re.search(r"[\s#\"']", v):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return v


def _eol(s: str) -> str:
    if s.endswith("\r\n"):
        return "\r\n"
    return "\n" if s.endswith("\n") else ""


def _read_text(path: Path) -> str:
    """Read .env text keeping the raw line endings. Path.read_text applies
    universal newlines (\r\n -> \n), which would hide the file's CRLF."""
    if not path.exists():
        return ""
    return path.read_bytes().decode("utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    """Write .env text with exactly the given line endings (no translation;
    write_text(newline=...) is Python 3.13+ only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def render_env(original: str, changes: dict[str, str | None]) -> str:
    """Apply `changes` (key -> new value, or None = unset) to the original text,
    preserving comments, layout and line endings. Keys absent from the file are
    inserted into their section (or appended at the end)."""
    lines = original.splitlines(keepends=True)   # keep endings: CRLF round-trips byte-exact
    _, entries = parse_env(original)
    headers = [(i, m.group(1).strip()) for i, ln in enumerate(lines)
               if (m := HEADER_RE.match(ln))]

    def section_end(section: str) -> int:
        title = SECTION_HEADERS.get(section, section)
        idx = [i for i, t in headers if t == title]
        if not idx:
            idx = [i for i, t in headers if t.lower().startswith(title.lower())]
        if not idx:
            return len(lines)
        h = idx[0]
        later = [i for i, _ in headers if i > h]
        end = later[0] if later else len(lines)
        while end > h + 1 and lines[end - 1].strip() == "":
            end -= 1          # insert after the section's content, not its blank tail
        return end

    inserts: dict[str, list[str]] = {}
    for key, new in changes.items():
        field = CATALOG_BY_KEY.get(key)
        ents = entries.get(key)
        if ents:
            active = [e for e in ents if e["active"]]
            if new is None:
                if active:
                    e = active[0]
                    tail = f"{e['gap']}# {e['comment']}" if e["comment"] else ""
                    lines[e["lineno"]] = (f"# {key}={quote_value(e['value'])}{tail}"
                                          + _eol(lines[e["lineno"]]))
                continue
            target = active[0] if active else ents[0]
            tail = f"{target['gap']}# {target['comment']}" if target["comment"] else ""
            lines[target["lineno"]] = (f"{key}={quote_value(new)}{tail}"
                                       + _eol(lines[target["lineno"]]))
            continue
        if new is None:
            continue
        if field is None:
            eol = _eol(lines[-1]) if lines else "\n"
            if lines and lines[-1].strip() == "":
                lines[-1] = f"{key}={quote_value(new)}" + eol
            else:
                lines.append(f"{key}={quote_value(new)}" + eol)
            continue
        inserts.setdefault(field.section, []).append(f"{key}={quote_value(new)}")
    # Insert new keys at the end of their section, in catalog order.
    for section, newlines in inserts.items():
        pos = section_end(section)
        eol = _eol(lines[pos - 1]) if pos else ""   # match the file's EOL
        for i, ln in enumerate(newlines):
            lines.insert(pos + i, ln + eol)
    return "".join(lines)


def build_default_env() -> str:
    """Generate a .env from the catalog defaults (used when no .env.example exists)."""
    out = ["# flair .env — generated by configurer.py from built-in defaults", ""]
    for section in SECTIONS:
        title = SECTION_HEADERS.get(section, section)
        out.append(f"# {DASH}{DASH} {title} {DASH * max(4, 60 - len(title))}")
        out += [f"# {f.key}=" if f.default is None else f"{f.key}={quote_value(f.default)}"
                for f in CATALOG if f.section == section]
        out.append("")
    return "\n".join(out)


# Validation

def _f(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def validate(values: dict[str, str | None], env_path: Path) -> tuple[list[str], list[str]]:
    """values: key -> current value or None (unset). Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    v = values.get

    provider = (v("FLAIR_PROVIDER") or "deepseek").strip().lower()
    if provider not in ("deepseek", "openai", "local"):
        errors.append(f"FLAIR_PROVIDER must be 'deepseek', 'openai' or 'local' (got {provider!r}).")

    for f in CATALOG:
        raw = v(f.key)
        if raw is None or f.kind not in ("int", "float"):
            continue
        try:
            x = float(int(raw)) if f.kind == "int" else float(raw)
        except ValueError:
            errors.append(f"{f.key}: {raw!r} is not a valid {f.kind}.")
            continue
        if f.min is not None and x < f.min:
            errors.append(f"{f.key}: {raw} is below the minimum ({f.min:g}).")
        if f.max is not None and x > f.max:
            errors.append(f"{f.key}: {raw} is above the maximum ({f.max:g}).")

    for key, msg in (("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY is empty — flair will refuse to start"),
                     ("OPENAI_API_KEY", "OPENAI_API_KEY is empty — flair will refuse to start"),
                     ("LOCAL_BASE_URL", "LOCAL_BASE_URL is empty")):
        if provider == key.split("_", 1)[0].lower() and not v(key):
            warnings.append(f"FLAIR_PROVIDER={provider} but {msg}.")

    def resolve(p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else (env_path.parent / path)

    root = v("FLAIR_ROOT")
    if root and not resolve(root).exists():
        warnings.append(f"FLAIR_ROOT does not exist: {resolve(root)}")
    ca = v("FLAIR_CA_BUNDLE")
    if ca and not resolve(ca).is_file():
        warnings.append(f"FLAIR_CA_BUNDLE does not exist: {resolve(ca)}")

    ratio, hyst = _f(v("FLAIR_COMPACT_RATIO")), _f(v("FLAIR_PRUNE_HYSTERESIS"))
    if ratio is not None and hyst is not None and hyst >= ratio:
        warnings.append(
            "FLAIR_PRUNE_HYSTERESIS >= FLAIR_COMPACT_RATIO: pruning can never land "
            "below the threshold, so stage 0 will never skip the LLM summary.")
    return errors, warnings


# GUI

class ScrollFrame:
    """A vertically scrollable ttk.Frame (canvas + scrollbar)."""

    def __init__(self, parent: tk.Widget) -> None:
        self.canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.frame = ttk.Frame(self.canvas)
        self.win = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.frame.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        for w in (self.canvas, self.frame):               # wheel: Windows + Linux
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                w.bind(ev, self._wheel)

    def _wheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(-1 if getattr(event, "delta", 0) > 0 or event.num == 4 else 1, "units")


class App:
    def __init__(self, root: tk.Tk, env_path: Path) -> None:
        self.root = root
        self.env_path = env_path
        self.modified = False
        self.file_values: dict[str, str] = {}
        self.eol = "\n"
        self.loaded: dict[str, str | None] = {}
        self.bool_vars: dict[str, tk.BooleanVar] = {}
        self.secret_vars: dict[str, tk.BooleanVar] = {}
        self.widgets: dict[str, tk.Widget] = {}
        self.help_labels: dict[str, ttk.Label] = {}
        self.current_help: dict[str, Field] = {}

        root.title("flair .env configurator")
        root.geometry("980x660")
        root.minsize(780, 480)
        self._build_topbar()
        self._build_notebook()
        self._build_statusbar()
        root.bind_all("<Control-s>", lambda e: self.save())
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load_env()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_topbar(self) -> None:
        top = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        top.pack(side="top", fill="x")
        ttk.Label(top, text=".env file:").pack(side="left")
        self.path_var = tk.StringVar(value=str(self.env_path))
        self.path_entry = ttk.Entry(top, textvariable=self.path_var)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.path_entry.bind("<Return>", lambda e: self.open_path(self.path_var.get()))
        ttk.Button(top, text="Browse...", command=self.browse).pack(side="left")

        bar2 = ttk.Frame(self.root, padding=(8, 2, 8, 4))
        bar2.pack(side="top", fill="x")
        for text, cmd in (("New from template", self.new_from_template), ("Reload", self.load_env),
                          ("Reset to defaults", self.reset_to_defaults), ("Preview...", self.preview)):
            ttk.Button(bar2, text=text, command=cmd).pack(side="left", padx=(6, 0))
        ttk.Button(bar2, text="Save as...", command=lambda: self.save(ask=True)).pack(side="right", padx=(6, 0))
        self.save_btn = ttk.Button(bar2, text="Save", command=self.save)
        self.save_btn.pack(side="right")

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        for section in SECTIONS:
            tab = ttk.Frame(self.notebook, padding=(4, 6, 4, 2))
            self.notebook.add(tab, text=f" {section} ")
            scroll = ScrollFrame(tab)
            scroll.frame.pack(fill="both", expand=True)
            parent = scroll.frame
            parent.columnconfigure(1, weight=1)
            help_label = ttk.Label(tab, wraplength=920, foreground="#555", padding=(10, 2, 10, 4))
            help_label.pack(side="bottom", fill="x")
            self.help_labels[section] = help_label
            tab.bind("<Configure>",
                     lambda e, lb=help_label: lb.configure(wraplength=max(200, e.width - 30)))
            for row, f in enumerate([f for f in CATALOG if f.section == section]):
                self._add_row(parent, f, row)

    def _add_row(self, parent: tk.Widget, f: Field, row: int) -> None:
        label = ttk.Label(parent, text=f.key, font=("Consolas", 9), anchor="e")
        label.grid(row=row, column=0, sticky="e", padx=(8, 10), pady=3)

        if f.kind == "bool":
            var = tk.BooleanVar(value=False)
            self.bool_vars[f.key] = var
            w = ttk.Checkbutton(parent, variable=var)
            w.grid(row=row, column=1, sticky="w")
        elif f.kind == "choice":
            opts = list(f.choices or ())
            if f.default is None and UNSET not in opts:
                opts = [UNSET] + opts
            w = ttk.Combobox(parent, values=opts, width=42)
            w.grid(row=row, column=1, sticky="ew")
        elif f.kind in ("int", "float"):
            w = ttk.Entry(parent, width=34)
            w.grid(row=row, column=1, sticky="w")
        else:  # str / secret / path
            w = ttk.Entry(parent)
            if f.kind == "secret":
                w.configure(show="\u2022")
                show_var = tk.BooleanVar(value=False)
                self.secret_vars[f.key] = show_var
                hint = ttk.Label(parent, text="(click to show)", foreground="#888")
                hint.grid(row=row, column=2, sticky="w", padx=(6, 0))

                def _toggle(e, w=w, sv=show_var, hint=hint) -> None:
                    sv.set(not sv.get())
                    revealed = sv.get()
                    w.configure(show="" if revealed else "\u2022")
                    hint.configure(text="(click to hide)" if revealed
                                   else "(click to show)")

                w.bind("<Button-1>", _toggle)
                hint.bind("<Button-1>", _toggle)
            w.grid(row=row, column=1, sticky="ew", padx=(0, 4))
            if f.kind == "path" and f.pick:
                ttk.Button(parent, text="Browse...",
                           command=lambda f=f: self.pick_path(f)).grid(row=row, column=2, sticky="w")

        self.widgets[f.key] = w
        for widget in (label, w):
            widget.bind("<FocusIn>", lambda e, f=f: self.show_help(f))
            widget.bind("<KeyRelease>", lambda e: self.mark_modified())
            if f.kind == "bool":
                widget.bind("<Button-1>", lambda e: self.mark_modified())
        if isinstance(w, ttk.Combobox):
            # mouse-driven dropdown selection fires no KeyRelease
            w.bind("<<ComboboxSelected>>", lambda e: self.mark_modified())

    def _build_statusbar(self) -> None:
        self.status = tk.Label(self.root, anchor="w", relief="sunken", padx=8, pady=3,
                               background="#f0f0f0", foreground="#333")
        self.status.pack(side="bottom", fill="x")
        self.refresh_status()

    # ── value model ──────────────────────────────────────────────────────────

    def field_value(self, f: Field) -> str | None:
        """Current GUI value for a field: string, or None when unset."""
        if f.kind == "bool":
            return "true" if self.bool_vars[f.key].get() else "false"
        raw = str(self.widgets[f.key].get()).strip()
        if f.kind == "choice" and raw in ("", UNSET):
            return None
        return raw or None

    def all_values(self) -> dict[str, str | None]:
        return {f.key: self.field_value(f) for f in CATALOG}

    def changes(self) -> dict[str, str | None]:
        return {f.key: cur for f in CATALOG
                if (cur := self.field_value(f)) != self.loaded.get(f.key)}

    def mark_modified(self) -> None:
        self.modified = True
        self.refresh_status()

    # ── file I/O ─────────────────────────────────────────────────────────────

    def load_env(self, path: str | Path | None = None) -> None:
        if path:
            self.env_path = Path(path).expanduser()
            self.path_var.set(str(self.env_path))
        text = _read_text(self.env_path)
        self.eol = "\r\n" if "\r\n" in text else "\n"
        values, _ = parse_env(text)
        self.file_values = values
        for f in CATALOG:
            in_file = f.key in values
            loaded = values[f.key] if in_file else f.default
            if f.kind == "bool" and loaded is not None:
                # normalize 'yes'/'on'/... to 'true'/'false' so an untouched
                # checkbox never shows up as a pending change
                loaded = "true" if loaded.strip().lower() in BOOL_WORDS else "false"
            self.loaded[f.key] = loaded
            self._set_widget(f, loaded, in_file)
        self.modified = False
        self.show_help(CATALOG[0])
        self.refresh_status()

    def _set_widget(self, f: Field, value: str | None, in_file: bool) -> None:
        w = self.widgets[f.key]
        if f.kind == "bool":
            return self.bool_vars[f.key].set((value or "false").strip().lower() in BOOL_WORDS)
        if f.kind == "choice":
            opts = list(w["values"])
            if value and value not in opts:
                opts.append(value)
                w.configure(values=opts)
            w.set(UNSET if value is None else value)
            return
        w.delete(0, "end")
        if value is not None:
            w.insert(0, value)
        if not in_file:
            self.show_help(f)

    def save(self, ask: bool = False) -> bool:
        if ask:
            target = filedialog.asksaveasfilename(
                title="Save .env as", defaultextension=".env",
                initialfile=self.env_path.name,
                filetypes=[("env file", "*.env"), ("all files", "*.*")])
            if not target:
                return False
            target = Path(target)
        else:
            target = self.env_path

        errors, warnings = validate(self.all_values(), target)
        if errors:
            messagebox.showerror(
                "Cannot save",
                "Fix these problems first:\n\n" + "\n".join(f"\u2022 {e}" for e in errors))
            return False

        _write_text(target, self._rendered_text(target))
        self.env_path = target
        self.path_var.set(str(target))
        self.load_env(target)
        if warnings:
            messagebox.showwarning(
                "Saved with warnings",
                f"Saved {target}\n\n" + "\n".join(f"\u2022 {w}" for w in warnings))
        return True

    def browse(self) -> None:
        p = filedialog.askopenfilename(
            title="Open .env file",
            initialdir=str(self.env_path.parent) if self.env_path.parent.exists() else None,
            filetypes=[("env file", "*.env"), ("all files", "*.*")])
        if p:
            self.load_env(p)

    def open_path(self, p: str) -> None:
        if p.strip():
            self.load_env(p.strip())

    def new_from_template(self) -> None:
        target = self.env_path
        if target.exists() and not messagebox.askyesno(
                "New from template",
                f"{target.name} already exists. Replace it with the template?"):
            return
        example = target.parent / ".env.example"
        if not example.exists():
            alt = Path(__file__).resolve().parent.parent / "flair" / ".env.example"
            if alt.exists():
                example = alt
        target.parent.mkdir(parents=True, exist_ok=True)
        if example.exists():
            shutil.copyfile(example, target)
            src = str(example)
        else:
            _write_text(target, build_default_env())
            src = "built-in defaults"
        self.load_env(target)
        messagebox.showinfo("New from template", f"Created {target}\nfrom {src}.")

    def reset_to_defaults(self) -> None:
        if not messagebox.askyesno(
                "Reset to defaults",
                "Set every parameter to its built-in default? "
                "Values currently in the file will be overwritten on Save."):
            return
        for f in CATALOG:
            self._set_widget(f, f.default, False)
            self.loaded[f.key] = f.default
        self.mark_modified()

    def pick_path(self, f: Field) -> None:
        w = self.widgets[f.key]
        if f.pick == "dir":
            p = filedialog.askdirectory(title=f"Choose {f.key}")
        else:
            p = filedialog.askopenfilename(
                title=f"Choose {f.key}",
                filetypes=[("PEM certificate", "*.pem *.crt *.cer"), ("all files", "*.*")])
        if p:
            w.delete(0, "end")
            w.insert(0, p)
            self.mark_modified()

    def _rendered_text(self, target: Path) -> str:
        """The .env text as it would be written to `target` right now.

        render_env already keeps the original line endings; this only normalizes
        lines that were newly inserted (they carry the file's dominant EOL or
        none) so the whole file uses one consistent EOL — never \r\r\n."""
        text = render_env(_read_text(target), self.changes())
        return text.replace("\r\n", "\n").replace("\n", self.eol)

    def preview(self) -> None:
        win = tk.Toplevel(self.root)
        win.title(f"Preview — {self.env_path}")
        win.geometry("860x600")
        frame = ttk.Frame(win, padding=4)
        frame.pack(fill="both", expand=True)
        scb = ttk.Scrollbar(frame)
        scb.pack(side="right", fill="y")
        txt = tk.Text(frame, wrap="none", font=("Consolas", 9), yscrollcommand=scb.set)
        txt.pack(side="left", fill="both", expand=True)
        scb.configure(command=txt.yview)
        txt.insert("1.0", self._rendered_text(self.env_path))
        txt.configure(state="disabled")

    # ── status / help / lifecycle ────────────────────────────────────────────

    def refresh_status(self) -> None:
        n_total = len(CATALOG)
        n_file = sum(1 for f in CATALOG if f.key in self.file_values)
        n_pending = len(self.changes())
        errors, warnings = validate(self.all_values(), self.env_path)
        bits = [str(self.env_path), f"{n_file}/{n_total} params in file",
                f"{n_pending} pending" if n_pending else "no pending changes",
                f"{len(errors)} error(s)" if errors else "",
                f"{len(warnings)} warning(s)" if warnings else "",
                "modified" if self.modified else "saved"]
        bits = [b for b in bits if b]
        self.status.configure(
            text="   \u00b7   ".join(bits),
            foreground="#b00020" if errors else ("#9a6b00" if warnings else "#333"))

    def show_help(self, f: Field) -> None:
        self.current_help[f.section] = f
        note = "" if f.key in self.file_values else "  [not in file — the built-in default applies]"
        self.help_labels[f.section].configure(text=f"{f.key}: {f.help}{note}")

    def on_close(self) -> None:
        if self.modified:
            r = messagebox.askyesnocancel("Quit", "Unsaved changes. Save before quitting?")
            if r is None:
                return
            if r and not self.save():
                return
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def default_env_path() -> Path:
    """Default target: .env in the current directory, else the first .env found
    walking up from this script (repo root), else ./flair/.env."""
    cwd = Path.cwd() / ".env"
    if cwd.exists():
        return cwd
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / ".env").exists():
            return parent / ".env"
    return here / "flair" / ".env"


def run_check(env_path: Path) -> int:
    """Headless validation report (no GUI)."""
    if not env_path.exists():
        print(f"{env_path}: not found (a new file would be created on save)")
        return 0
    values, _ = parse_env(_read_text(env_path))
    merged = {f.key: (values.get(f.key) if f.key in values else f.default) for f in CATALOG}
    errors, warnings = validate(merged, env_path)
    n_unknown = len(set(values) - set(CATALOG_BY_KEY))
    print(f"{env_path}: {len(values)} params in file, {n_unknown} unknown (kept as-is)")
    for e, w in (("  ERROR: ", errors), ("  WARN:  ", warnings)):
        for m in w:
            print(e + m)
    if not errors and not warnings:
        print("  OK - no problems found")
    return 1 if errors else 0


def _gui():
    """Import tkinter on demand (keeps the module importable without python3-tk)."""
    global tk, ttk, filedialog, messagebox
    import tkinter as _tk
    from tkinter import filedialog as _fd
    from tkinter import messagebox as _mb
    from tkinter import ttk as _ttk
    tk, ttk, filedialog, messagebox = _tk, _ttk, _fd, _mb
    return _tk, _ttk


def main(argv: list[str]) -> int:
    args = [a for a in argv if a]   # tolerate empty entries
    if args and args[0] == "--check":
        path = Path(args[1]).expanduser() if len(args) > 1 else default_env_path()
        return run_check(path)
    path = Path(args[0]).expanduser() if args else default_env_path()
    tk, ttk = _gui()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root, path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
