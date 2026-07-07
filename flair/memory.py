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

_HEADER = "## Memoria di sessione (fatti appresi in lavori precedenti)"

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
            return False, "nota vuota: niente da memorizzare."
        if len(note) > self.max_note_chars:
            return False, (f"nota troppo lunga ({len(note)} caratteri, max {self.max_note_chars}): "
                           "sintetizza il fatto in una riga.")
        if _SECRET_RX.search(note):
            return False, "la nota sembra contenere credenziali o segreti: non memorizzabile."
        if self._norm(note) in {self._norm(n) for n in self.notes}:
            return False, "fatto già in memoria."
        if self.used_chars() + len(note) + 3 > self.max_chars:
            return False, (f"memoria piena ({self.used_chars()}/{self.max_chars} caratteri): "
                           "sii più selettivo; l'utente può ripulirla con /memory.")
        self.notes.append(note)
        return True, f"memorizzato ({len(self.notes)} note in memoria)."

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
        return ("# Memoria di sessione (flair)\n"
                "# Una riga per nota; il file è modificabile a mano.\n\n"
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
