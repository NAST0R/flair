"""Helper filesystem condivisi tra i tool.

`resolve(root, raw)` confina i path dentro `root` quando `root` è impostato
(agente coding → radice del progetto). Con `root=None` (agente generico) il
path viene solo espanso/risolto, senza confinamento: serve a operare su tutta
la macchina ("apri quel file sul Desktop", "trova una canzone in Musica").

Le funzioni `read_file_impl` / `list_dir_impl` sono usate da entrambi gli
agenti, evitando duplicazione: cambia solo il `root` passato.
"""

from __future__ import annotations

from pathlib import Path

from ..core.tool import ToolError

_BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".woff", ".woff2",
    ".ttf", ".mp3", ".mp4", ".mov", ".flac", ".wav", ".jar", ".class",
}

NOISE_DIRS = {
    "__pycache__", ".git", "node_modules", "venv", ".venv", "build", "dist",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
    "target", ".next", ".gradle", "$RECYCLE.BIN", "System Volume Information",
}


class PathOutsideRoot(ToolError):
    pass


def resolve(root: Path | None, raw: str) -> Path:
    """Risolve `raw`. Se `root` è dato, garantisce che resti al suo interno."""
    p = Path(raw).expanduser()
    if root is None:
        return p.resolve()
    candidate = (p if p.is_absolute() else root / p).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathOutsideRoot(f"Path fuori dalla radice di lavoro ({root}): {raw}") from exc
    return candidate


def display(root: Path | None, p: Path) -> str:
    """Path leggibile: relativo alla radice se possibile, altrimenti assoluto."""
    if root is not None:
        try:
            return str(p.relative_to(root))
        except ValueError:
            pass
    return str(p)


def add_line_numbers(text: str, start: int = 1) -> str:
    lines = text.splitlines()
    width = len(str(start + len(lines) - 1)) if lines else 1
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, start=start))


def _trunc(text: str, limit: int, hint: str = "") -> str:
    if len(text) <= limit:
        return text
    extra = f"\n...[output troncato a {limit} caratteri{'; ' + hint if hint else ''}]"
    return text[:limit] + extra


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _reindent(new: str, old_base: str, file_base: str) -> str:
    """Ri-basa l'indentazione di `new` da `old_base` a `file_base`, preservando
    l'indentazione relativa interna al blocco."""
    out = []
    for line in new.split("\n"):
        if not line.strip():
            out.append("")
            continue
        rel = line[len(old_base):] if line.startswith(old_base) else line.lstrip()
        out.append(file_base + rel)
    return "\n".join(out)


def _unique_window(text_lines: list[str], old_lines: list[str], key) -> int | None:
    """Indice dell'unica finestra che combacia secondo `key`, o None se zero/ambigua."""
    n, L = len(text_lines), len(old_lines)
    matches = [
        i for i in range(0, n - L + 1)
        if all(key(text_lines[i + k]) == key(old_lines[k]) for k in range(L))
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ToolError(
            f"old_string corrisponde a {len(matches)} blocchi (a meno di spazi). "
            "Aggiungi contesto per renderlo univoco."
        )
    return None


def apply_edit(text: str, old: str, new: str, replace_all: bool = False) -> tuple[str, str]:
    """Applica una sostituzione `old`→`new` con fallback tolleranti agli spazi.

    Ritorna (nuovo_testo, strategia). Solleva ToolError se non trova un match
    univoco. Strategie, in ordine di preferenza: esatto, spazi esterni, fine-riga,
    indentazione.
    """
    if not old:
        raise ToolError("old_string vuoto: specifica il testo da sostituire.")

    # 1. Match esatto.
    n = text.count(old)
    if n >= 1:
        if n > 1 and not replace_all:
            raise ToolError(
                f"old_string compare {n} volte. Aggiungi contesto per renderlo "
                "univoco, oppure usa replace_all=true."
            )
        count = -1 if replace_all else 1
        return text.replace(old, new, count), "esatto"

    if replace_all:
        raise ToolError("old_string non trovato (con replace_all serve il match esatto).")

    # 2. Spazi esterni dell'intero blocco ignorati.
    stripped = old.strip()
    if stripped and text.count(stripped) == 1:
        return text.replace(stripped, new, 1), "spazi esterni ignorati"

    # 3/4. Match per riga: prima ignorando il fine-riga, poi l'indentazione.
    text_lines = text.split("\n")
    old_lines = old.split("\n")
    if old_lines and old_lines[-1] == "":
        old_lines = old_lines[:-1]  # ignora newline finale dell'old_string
    if not old_lines:
        raise ToolError("old_string non trovato.")

    for key, label in ((lambda s: s.rstrip(), "fine-riga tollerato"),
                       (lambda s: s.strip(), "indentazione tollerata")):
        idx = _unique_window(text_lines, old_lines, key)
        if idx is None:
            continue
        starts = _line_starts(text)
        start_off = starts[idx]
        end_line = idx + len(old_lines)
        end_off = starts[end_line] if end_line < len(starts) else len(text)
        matched = text[start_off:end_off]

        old_base = _leading_ws(old_lines[0])
        file_base = _leading_ws(text_lines[idx])
        replacement = _reindent(new, old_base, file_base)
        if matched.endswith("\n") and not replacement.endswith("\n"):
            replacement += "\n"
        return text[:start_off] + replacement + text[end_off:], label

    raise ToolError(
        "old_string non trovato. Rileggi il file con read_file e copia il testo "
        "esatto da sostituire (inclusa l'indentazione)."
    )


def read_file_impl(root: Path | None, path: str, offset: int, limit: int | None, max_chars: int) -> str:
    p = resolve(root, path)
    if not p.exists():
        return f"❌ Il file non esiste: {display(root, p)}"
    if p.is_dir():
        return f"❌ È una directory, non un file: {display(root, p)} (usa list_directory)"
    if p.suffix.lower() in _BINARY_EXT:
        return f"❌ File binario non leggibile come testo: {display(root, p)}"

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)
    offset = max(1, offset)
    end = total if limit is None else min(total, offset - 1 + limit)
    chunk = "\n".join(lines[offset - 1:end])

    header = f"{display(root, p)}  (righe {offset}-{end} di {total})\n"
    out = header + add_line_numbers(chunk, start=offset)
    if end < total:
        out += f"\n...[restano {total - end} righe; continua con read_file(path, offset={end + 1})]"
    return _trunc(out, max_chars, hint="leggi un range più piccolo con offset/limit")


def list_dir_impl(root: Path | None, path: str, max_entries: int) -> str:
    p = resolve(root, path)
    if not p.exists():
        return f"❌ La directory non esiste: {display(root, p)}"
    if not p.is_dir():
        return f"❌ Non è una directory: {display(root, p)}"

    entries: list[str] = []
    try:
        children = sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower()))
    except PermissionError:
        return f"❌ Permesso negato: {display(root, p)}"
    for child in children:
        if child.name in NOISE_DIRS:
            continue
        if child.is_dir():
            entries.append(f"{child.name}/")
        else:
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entries.append(f"{child.name}  ({size} B)")

    if not entries:
        return f"{display(root, p)}/  (vuota)"
    shown = entries[:max_entries]
    out = f"{display(root, p)}/\n" + "\n".join(shown)
    if len(entries) > len(shown):
        out += f"\n...[altre {len(entries) - len(shown)} voci]"
    return out
