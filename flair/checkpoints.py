"""Checkpoint dei file modificati da flair, per turno: `/rewind` e `/diff`.

Un agente di coding sbaglia edit multi-file, soprattutto un modello locale: senza
un modo di tornare indietro l'unica rete è git, e solo se si è committato prima.
Qui, prima di ogni scrittura fatta dai tool di flair, si salva il contenuto
precedente del file — una volta per file per turno — così che `/rewind` possa
riportare i file (e la conversazione) a prima dell'ultimo turno.

Tre scelte deliberate:

1. **Conferma solo a scrittura riuscita.** Lo snapshot viene preparato prima e
   confermato solo se il tool non ha fallito: un edit rifiutato non lascia un
   checkpoint fasullo.
2. **La conversazione si accorcia, non si riscrive.** Riavvolgere taglia la coda
   della storia, quindi ciò che resta è un prefisso di quanto già inviato: la
   cache del provider continua a valere. Se nel frattempo c'è stata una
   compattazione (che sostituisce i messaggi), tagliare la storia sarebbe errato:
   allora si riavvolgono solo i file, e lo si dice.
3. **Confini dichiarati.** Si tracciano solo i file scritti dai tool di flair,
   nella sessione corrente: una modifica fatta da `run_command` (un `sed`, un
   formatter) non è vista. I checkpoint vivono nel processo: non sopravvivono a
   `/load`, `/reset` o all'uscita.
"""

from __future__ import annotations

import difflib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Quanti turni si conservano, e oltre quale dimensione un file non viene
# fotografato (un dump da 50 MB riscritto dall'agente non deve finire in RAM).
MAX_TURNS = 20
MAX_FILE_BYTES = 2_000_000
# Tetto complessivo: senza, venti turni che riscrivono file da 2 MB arrivavano a
# mezzo giga di RAM. Oltre il tetto si scartano i turni più vecchi (mai quello in
# corso), così il costo resta proporzionato all'uso reale.
MAX_TOTAL_BYTES = 64_000_000


@dataclass
class _Turn:
    start_len: int                 # lunghezza della conversazione all'inizio del turno
    anchor: object | None          # ultimo messaggio prima del turno (identità)
    ops: list[tuple] = field(default_factory=list)   # ("write", path, prev) | ("move", src, dst)
    touched: set[str] = field(default_factory=set)
    skipped: list[str] = field(default_factory=list)  # file troppo grandi per lo snapshot


@dataclass
class RewindReport:
    restored: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unmoved: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conversation_rewound: bool = False
    turns: int = 0


class Checkpoints:
    """Registro dei checkpoint della sessione, condiviso fra gli agenti come la
    memoria e i job (vive nel contesto dei tool)."""

    def __init__(self, max_turns: int = MAX_TURNS, max_file_bytes: int = MAX_FILE_BYTES,
                 max_total_bytes: int = MAX_TOTAL_BYTES) -> None:
        self.max_turns = max_turns
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self._turns: list[_Turn] = []

    # ── ciclo di vita ───────────────────────────────────────────────────────
    def begin_turn(self, messages: list) -> None:
        """Apre il turno: ricorda dove inizia nella conversazione, per poterla
        riportare qui. Un turno senza scritture non lascia traccia."""
        anchor = messages[-1] if messages else None
        self._turns.append(_Turn(start_len=len(messages), anchor=anchor))
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def clear(self) -> None:
        self._turns.clear()

    @property
    def rewindable(self) -> int:
        """Turni che hanno modificato almeno un file — compresi quelli che hanno
        toccato solo file troppo grandi per lo snapshot: riavvolgerli non li
        ripristina, ma va detto, ed è /rewind a dirlo (stesso criterio di rewind())."""
        return sum(1 for t in self._turns if t.ops or t.skipped)

    # ── registrazione (chiamata dai tool) ───────────────────────────────────
    def prepare_write(self, path: Path) -> tuple | None:
        """Snapshot del file PRIMA della scrittura, da confermare con `commit`
        se la scrittura riesce. None se non c'è un turno aperto o il file è già
        stato fotografato in questo turno (vale il contenuto di inizio turno)."""
        turn = self._turns[-1] if self._turns else None
        key = str(path)
        if turn is None or key in turn.touched:
            return None
        if path.is_file():
            if path.stat().st_size > self.max_file_bytes:
                return ("skip", key)
            return ("write", key, path.read_bytes())
        return ("write", key, None)          # file nuovo: riavvolgere = rimuoverlo

    def prepare_move(self, src: Path, dst: Path) -> tuple | None:
        return ("move", str(src), str(dst)) if self._turns else None

    def commit(self, op: tuple | None) -> None:
        if op is None or not self._turns:
            return
        turn = self._turns[-1]
        if op[0] == "skip":
            turn.skipped.append(op[1])
            turn.touched.add(op[1])
            return
        turn.ops.append(op)
        if op[0] == "write":
            turn.touched.add(op[1])
        self._enforce_budget()

    def _bytes(self) -> int:
        return sum(len(op[2]) for t in self._turns for op in t.ops
                   if op[0] == "write" and op[2] is not None)

    def _enforce_budget(self) -> None:
        while len(self._turns) > 1 and self._bytes() > self.max_total_bytes:
            self._turns.pop(0)

    # ── riavvolgimento ──────────────────────────────────────────────────────
    def rewind(self, messages: list, turns: int = 1) -> RewindReport:
        """Riporta file e conversazione a prima degli ultimi `turns` turni che
        hanno modificato file. Le operazioni si annullano in ordine INVERSO."""
        report = RewindReport()
        cut_all = True                       # ogni turno riavvolto ha potuto tagliare la storia?
        while turns > 0 and self._turns:
            turn = self._turns.pop()
            if not turn.ops and not turn.skipped:
                continue                     # turno senza scritture: non conta
            turns -= 1
            report.turns += 1
            report.skipped.extend(turn.skipped)
            for op in reversed(turn.ops):
                self._undo(op, report)
            # La conversazione si taglia solo se il punto di taglio è ancora quello
            # registrato: una compattazione sostituisce i messaggi e lo invalida.
            if turn.start_len <= len(messages) and (
                    turn.start_len == 0 or messages[turn.start_len - 1] is turn.anchor):
                del messages[turn.start_len:]
            else:
                cut_all = False
        # «Riavvolta» solo se TUTTI i turni hanno potuto tagliarla: un taglio
        # parziale lascerebbe la storia più avanti dei file, e va detto.
        report.conversation_rewound = report.turns > 0 and cut_all
        return report

    @staticmethod
    def _undo(op: tuple, report: RewindReport) -> None:
        try:
            if op[0] == "write":
                _, key, prev = op
                p = Path(key)
                if prev is None:
                    if p.is_file():
                        p.unlink()
                        report.removed.append(key)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_name(p.name + ".flair-rewind")
                    tmp.write_bytes(prev)
                    os.replace(tmp, p)
                    report.restored.append(key)
            elif op[0] == "move":
                _, src, dst = op
                if Path(dst).exists() and not Path(src).exists():
                    shutil.move(dst, src)
                    report.unmoved.append(f"{dst} → {src}")
        except OSError as exc:
            report.failed.append(f"{op[1]}: {exc}")

    # ── diff della sessione ────────────────────────────────────────────────
    def session_diff(self, max_chars: int = 20_000) -> str:
        """Diff unificato fra lo stato di ogni file a inizio sessione (il primo
        snapshot registrato) e quello attuale."""
        first: dict[str, bytes | None] = {}
        for turn in self._turns:
            for op in turn.ops:
                if op[0] == "write" and op[1] not in first:
                    first[op[1]] = op[2]
        chunks: list[str] = []
        for key, before in first.items():
            p = Path(key)
            after = p.read_bytes() if p.is_file() else None
            if before == after:
                continue
            a = (before or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            b = (after or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            label_a = key if before is not None else "/dev/null"
            label_b = key if after is not None else "/dev/null"
            chunks.append("".join(difflib.unified_diff(a, b, label_a, label_b)))
        text = "".join(chunks)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [diff truncated at {max_chars} chars]"
        return text


# ── Aggancio per i tool ──────────────────────────────────────────────────────
# Un solo punto di passaggio per tutte le scritture dei tool: prepara lo
# snapshot, esegue, conferma solo se la scrittura è riuscita, e (per il coding
# agent) accoda l'esito del controllo dopo l'edit.

def _failed(out: str) -> bool:
    """L'esito di un tool che NON ha scritto nulla: errore o nessuna modifica."""
    return out.startswith(("❌", "⚠️"))


def tracked_write(ctx, root, path: str, write, check: bool = False) -> str:
    """Esegue `write()` (la scrittura vera) dentro il checkpoint del turno."""
    from .tools import fs, post_edit  # locale: i moduli dei tool importano questo modulo
    ckpt = getattr(ctx, "checkpoints", None)
    try:
        target: Path | None = fs.resolve(root, path)
    except Exception:  # noqa: BLE001 — path fuori sandbox o malformato: lo dirà il tool stesso
        target = None
    op = ckpt.prepare_write(target) if (ckpt is not None and target is not None) else None
    out = write()
    if _failed(out):
        return out
    if ckpt is not None:
        ckpt.commit(op)
    if check and target is not None:
        out += post_edit.run_check(ctx.cfg, target)
    return out


def tracked_move(ctx, root, src: str, dst: str, move) -> str:
    from .tools import fs
    ckpt = getattr(ctx, "checkpoints", None)
    op = None
    if ckpt is not None:
        try:
            op = ckpt.prepare_move(fs.resolve(root, src), fs.resolve(root, dst))
        except Exception:  # noqa: BLE001
            op = None
    out = move()
    if not _failed(out) and ckpt is not None:
        ckpt.commit(op)
    return out
