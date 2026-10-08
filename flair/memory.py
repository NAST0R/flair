"""Memoria di sessione: fatti DUREVOLI che sopravvivono a compaction e riavvii.

Complementare alla conversazione (il JSON di sessione), su un asse ortogonale:
la conversazione porta la narrativa del lavoro (compattabile, quindi lossy);
la memoria porta i fatti sul progetto/macchina/preferenze (poche righe, lossless).

Design (e suoi perché):
- Vive nel SYSTEM PROMPT, accanto alle istruzioni di progetto → prefisso stabile
  → CACHE del provider: dopo la prima chiamata costa i prezzi cache-hit. Vuota
  non inietta nulla (zero token per chi non la usa).
- Viene (ri)composta SOLO ai confini di sessione (avvio, /load, /memory clear):
  mai a metà lavoro, così il prefisso in cache non si rompe. Un `remember`
  durante la sessione aggiorna lo stato (e il prossimo salvataggio), non il
  prompt in corso: il fatto appena appreso è già nella conversazione corrente.
- Compaction e pruning operano SOLO su convo.messages: il system prompt è fuori
  dalla loro portata per costruzione, quindi le note non vengono mai riassunte
  né potate.
- Su disco è un sidecar markdown accanto al JSON di sessione (<nome>.memory.md),
  leggibile e modificabile a mano: trasparenza prima di tutto.
- Tetto DURO senza magie: al superamento si rifiuta con messaggio azionabile.
  Niente eviction automatica (le note vecchie sono spesso le più importanti) né
  distillazione LLM decisa dalla macchina: la potatura è una scelta dell'utente
  (/memory, o l'editor sul file).
"""

from __future__ import annotations

import re

_HEADER = "## Session memory (facts learned in previous work)"

# Pattern ovvi di credenziali/segreti: una nota che li contiene viene rifiutata.
# Volutamente conservativo (pochi falsi positivi): non è una barriera perfetta,
# è una rete di sicurezza contro l'errore in buona fede.
_SECRET_RX = re.compile(
    r"(sk-[A-Za-z0-9]{8,}"                      # chiavi stile OpenAI/DeepSeek
    r"|ghp_[A-Za-z0-9]{20,}"                    # GitHub PAT
    r"|AKIA[0-9A-Z]{16}"                        # AWS access key id
    r"|-----BEGIN [A-Z ]*PRIVATE KEY"           # chiavi PEM
    r"|\bbearer\s+[A-Za-z0-9._\-]{12,}"         # header Authorization
    r"|\b(api[_-]?key|password|passwd|secret|token)\s*[=:]\s*\S+)",  # coppie chiave=valore
    re.IGNORECASE,
)


class SessionMemory:
    """Lista di note brevi (una riga ciascuna) con dedup, filtro segreti e tetto."""

    def __init__(self, max_chars: int = 4000, max_note_chars: int = 200) -> None:
        self.max_chars = max(200, int(max_chars))
        self.max_note_chars = max(40, int(max_note_chars))
        self.notes: list[str] = []

    # ── interni ──────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(note: str) -> str:
        """Chiave di dedup: spazi normalizzati, case-insensitive."""
        return " ".join(note.split()).lower()

    def used_chars(self) -> int:
        """Occupazione attuale, misurata come apparirà nel blocco ('- nota\\n')."""
        return sum(len(n) + 3 for n in self.notes)

    # ── scrittura ────────────────────────────────────────────────────────────

    def add(self, note: str) -> tuple[bool, str]:
        """Aggiunge una nota. Ritorna (ok, messaggio per il modello). Deterministico,
        zero chiamate LLM: dedup, filtro segreti e tetto sono regole fisse."""
        note = " ".join(str(note).split())  # una riga, spazi normalizzati
        if not note:
            return False, "empty note: nothing to store."
        if len(note) > self.max_note_chars:
            return False, (f"note too long ({len(note)} chars, max {self.max_note_chars}): "
                           "condense the fact into one line.")
        if _SECRET_RX.search(note):
            return False, "the note appears to contain credentials or secrets: not stored."
        if self._norm(note) in {self._norm(n) for n in self.notes}:
            return False, "already in memory."
        if self.used_chars() + len(note) + 3 > self.max_chars:
            return False, (f"memory is full ({self.used_chars()}/{self.max_chars} chars): "
                           "be more selective; the user can prune it with /memory.")
        self.notes.append(note)
        return True, f"stored ({len(self.notes)} notes in memory)."

    def remove(self, selector: str, by_index: bool = True) -> tuple[bool, str, str | None]:
        """Rimuove UNA nota. Ritorna (ok, messaggio, nota rimossa).

        `selector` è il numero mostrato da /memory (1-based, solo se `by_index`)
        oppure un testo: la nota esatta — a meno di maiuscole e spazi — o un
        frammento che ne identifichi UNA sola. Mai più di una nota per chiamata e
        mai a caso: un frammento ambiguo viene rifiutato elencando le candidate,
        così chi chiede restringe invece di cancellare la nota sbagliata.

        `by_index=False` è per il modello: nel system prompt le note non sono
        numerate, e dopo una rimozione a metà sessione la lista che vede non
        coincide più con quella reale — un indice cancellerebbe la nota sbagliata.
        Il testo, invece, resta un riferimento stabile."""
        sel = (selector or "").strip()
        if not sel:
            return False, "say which note to forget (its number or its text).", None
        if not self.notes:
            return False, "memory is empty: nothing to forget.", None
        if by_index and sel.isdigit():
            i = int(sel)
            if not 1 <= i <= len(self.notes):
                return False, f"there is no note number {i}: memory holds {len(self.notes)}.", None
            return True, f"forgotten ({len(self.notes) - 1} left).", self.notes.pop(i - 1)
        key = self._norm(sel)
        exact = [n for n in self.notes if self._norm(n) == key]
        matches = exact or [n for n in self.notes if key in self._norm(n)]
        if not matches:
            return False, "no note matches that text.", None
        if len(matches) > 1:
            listed = "; ".join(f"«{m}»" for m in matches[:5])
            return False, f"{len(matches)} notes match, be more specific: {listed}", None
        self.notes.remove(matches[0])
        return True, f"forgotten ({len(self.notes)} left).", matches[0]

    def clear(self) -> None:
        self.notes = []

    # ── lettura / iniezione ──────────────────────────────────────────────────

    def block(self) -> str:
        """Blocco da appendere al system prompt. Vuota → stringa vuota (zero token)."""
        if not self.notes:
            return ""
        return f"\n\n{_HEADER}\n\n" + "\n".join(f"- {n}" for n in self.notes)

    # ── serializzazione sidecar ──────────────────────────────────────────────

    def to_text(self) -> str:
        """Contenuto del sidecar markdown. Dedup difensivo (preservando l'ordine):
        ripulisce l'eventuale doppione teorico di un batch parallelo."""
        seen: set[str] = set()
        out: list[str] = []
        for n in self.notes:
            k = self._norm(n)
            if k in seen:
                continue
            seen.add(k)
            out.append(n)
        self.notes = out
        if not out:
            return ""
        return ("# Session memory (flair)\n"
                "# One line per note; this file can be edited by hand.\n\n"
                + "\n".join(f"- {n}" for n in out) + "\n")

    def load_text(self, text: str) -> tuple[int, bool]:
        """Carica le note da un sidecar (anche editato a mano): tollerante ma con le
        stesse regole di sicurezza. Righe valide: '- nota' o '* nota'; il resto è
        ignorato. Note oltre il limite per-nota vengono troncate; al superamento del
        tetto totale si smette di caricare. Ritorna (note_caricate, troncato)."""
        self.notes = []
        truncated = False
        for line in (text or "").splitlines():
            s = line.strip()
            if not s.startswith(("- ", "* ")):
                continue
            note = " ".join(s[2:].split())
            if not note or _SECRET_RX.search(note):
                continue
            if len(note) > self.max_note_chars:
                note = note[: self.max_note_chars]
                truncated = True
            if self._norm(note) in {self._norm(n) for n in self.notes}:
                continue
            if self.used_chars() + len(note) + 3 > self.max_chars:
                truncated = True
                break
            self.notes.append(note)
        return len(self.notes), truncated
