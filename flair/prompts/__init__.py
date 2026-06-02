"""Caricamento dei prompt di sistema e delle istruzioni di progetto."""

from __future__ import annotations

from pathlib import Path

from ..config import PROJECT_INSTRUCTION_FILES

_DIR = Path(__file__).resolve().parent
_MAX_PROJECT_CHARS = 8000


def load(name: str) -> str:
    path = _DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt non trovato: {path}")
    return path.read_text(encoding="utf-8")


def project_instructions(root: Path) -> str:
    """Legge il primo file di istruzioni di progetto presente nella root (o '')."""
    for name in PROJECT_INSTRUCTION_FILES:
        p = root / name
        if p.exists() and p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if not text:
                continue
            if len(text) > _MAX_PROJECT_CHARS:
                text = text[:_MAX_PROJECT_CHARS] + "\n…[istruzioni troncate]"
            return f"\n\n## Istruzioni specifiche del progetto (da {name})\n\n{text}"
    return ""
