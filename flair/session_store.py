"""Persistenza delle sessioni: salva e riprende lo stato della conversazione.

Una sessione è un file JSON `<nome>.json` nella cartella sessioni. Contiene la
conversazione condivisa dai due agenti (messaggi + uso cumulativo) e l'ultimo
agente attivo, così da poter chiudere flair e riprendere da dove si era.

Tutto best-effort: un errore di salvataggio viene segnalato ma non interrompe
mai il lavoro. I segreti non vengono salvati (solo i messaggi della chat).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
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
            target = self._path(name)
            data = json.dumps(payload, ensure_ascii=False)
            # Scrittura ATOMICA: file temporaneo nella stessa cartella + os.replace (atomico
            # anche su Windows). Così un kill a metà scrittura, o due run concorrenti sulla
            # stessa sessione, non lasciano mai un JSON troncato: resta il vecchio o c'è il
            # nuovo completo. Il temp (.<nome>-*.tmp, nascosto) non è un *.json → list() lo ignora.
            fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{target.stem}-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(data)
                os.replace(tmp, target)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            return target
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
