"""Persistenza delle sessioni: salva e riprende lo stato della conversazione.

Una sessione è un file JSON `<nome>.json` nella cartella sessioni. Contiene lo
stato di entrambi gli agenti (messaggi + uso cumulativo) e l'ultimo agente
attivo, così da poter chiudere flair e riprendere esattamente da dove si era.

Tutto best-effort: un errore di salvataggio viene segnalato ma non interrompe
mai il lavoro. I segreti non vengono salvati (solo i messaggi della chat).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger("flair")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    cleaned = _SAFE.sub("-", name.strip()).strip("-")
    return cleaned or "default"


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def _path(self, name: str) -> Path:
        return self.dir / f"{_safe_name(name)}.json"

    def save(self, name: str, state: dict) -> Path | None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            payload = {"saved_at": datetime.now().isoformat(timespec="seconds"), **state}
            self._path(name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return self._path(name)
        except OSError as exc:
            log.warning("Salvataggio sessione '%s' fallito: %s", name, exc)
            return None

    def load(self, name: str) -> dict | None:
        p = self._path(name)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Caricamento sessione '%s' fallito: %s", name, exc)
            return None

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def list(self) -> list[tuple[str, str]]:
        """[(nome, timestamp)] ordinato dal più recente."""
        if not self.dir.exists():
            return []
        items = []
        for p in self.dir.glob("*.json"):
            try:
                saved = json.loads(p.read_text(encoding="utf-8")).get("saved_at", "")
            except (OSError, json.JSONDecodeError):
                saved = ""
            items.append((p.stem, saved, p.stat().st_mtime))
        items.sort(key=lambda t: t[2], reverse=True)
        return [(name, saved) for name, saved, _ in items]

    def latest(self) -> str | None:
        items = self.list()
        return items[0][0] if items else None
