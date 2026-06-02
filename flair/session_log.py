"""Osservabilità: log di sessione in JSONL e log su file degli eventi interni.

Il `SessionLogger` scrive un record per turno (task, risposta, tool usati, usage)
così da poter analizzare a posteriori dove vanno i token. Vive nella CLI, che
intercetta già i callback dei tool — l'agente resta disaccoppiato dal logging.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path


def setup_file_logging(log_dir: Path) -> Path:
    """Aggancia un handler su file al logger 'flair' (warning interni, retry, ecc.)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "flair.log"
    logger = logging.getLogger("flair")
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_flair", False) for h in logger.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler._flair = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return path


def _trunc(value, n: int):
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


class SessionLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"session-{ts}.jsonl"

    def log_turn(self, agent: str, task: str, result, tool_events: list[dict]) -> None:
        usage = result.usage
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "task": _trunc(task, 2000),
            "response": _trunc(result.content or "", 4000),
            "steps": result.steps,
            "stopped_reason": result.stopped_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cache_hit_tokens": usage.cache_hit_tokens,
                "cache_miss_tokens": usage.cache_miss_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            },
            "tools": tool_events,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
