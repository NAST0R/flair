"""Router minimale per scegliere l'agente.

Strategia (token-conscious, niente Frankenstein):
1. Euristica su parole chiave/segnali → 'coding' o 'general' nella maggior parte dei casi.
2. Se ambiguo, resta "appiccicato" all'ultimo agente usato (continuità di sessione).
3. Solo se davvero indeciso e senza storia, una singola chiamata LLM da pochi token.

Nel dubbio, default 'general' (un assistente generico sa anche conversare).
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_CODE_HINTS = re.compile(
    r"\b(funzione|classe|metodo|modulo|refactor|rifattorizz|bug|debug|errore|eccezione|"
    r"codebase|repository|repo|codice|import|compil|stacktrace|traceback|lint|test|"
    r"function|class|method|module|refactor|exception|stack.?trace|unittest|pytest|"
    r"git|commit|branch|merge|diff|endpoint|api|migrazione|migration|sql|query|"
    r"variabile|parametro|dipendenz|build)\b"
    r"|\.(py|js|ts|jsx|tsx|java|cpp|c|cs|go|rs|rb|php|swift|kt|sh|sql|html|css|json|yaml|yml|toml)\b"
    r"|[/\\][\w./\\-]+\.\w+",
    re.IGNORECASE,
)

_SYSTEM_HINTS = re.compile(
    r"\b(?:"
    # Verbi d'azione desktop — stem con confine iniziale e suffisso libero,
    # così prende le coniugazioni e i clitici: apri/aprire/aprimi, avvia/avviare,
    # lancia/lanciare, riprodu(ci/rre), ascolt(a/ami), suona/suonare, metti su.
    r"apri|avvia|lancia|riprodu|ascolt|suona|metti\s*su|"
    r"open|launch|play\b|"
    # Oggetti chiaramente "desktop" (parole intere).
    r"browser|sito\b|news\b|canzone|brano|musica|playlist|"
    r"appunti|clipboard|incolla|paste\b|screenshot|schermata|volume\b|"
    # Info di sistema / orario / data.
    r"system\s*info|info\s*(?:di|sul)\s*sistema|spazio\s*su\s*disco|"
    r"memoria\s*ram|\bram\b|processi\b|che\s*ore|che\s*ora|data\s*(?:di\s*)?oggi|data\s*odierna|"
    r"what\s*time|today|"
    # Ricerca di file: il verbo da solo è debole (può essere coding), quindi
    # lo si lega a un oggetto chiaramente da disco/multimediale.
    r"(?:trova|cerca|find)\w*[^.]{0,20}?(?:canzone|brano|musica|file|cartella|"
    r"documento|foto|immagine|video|song|folder|photo)"
    r")",
    re.IGNORECASE,
)

_ROUTER_PROMPT = (
    "Classifica la richiesta dell'utente in una sola parola: 'coding' se riguarda "
    "leggere/scrivere/analizzare codice o file di progetto; 'general' per tutto il "
    "resto (aprire programmi, trovare file sul PC, riprodurre media, info di sistema, "
    "domande generiche, conversazione). Rispondi SOLO con 'coding' o 'general'."
)


def classify(text: str, provider, last_agent: str | None = None) -> str:
    code = bool(_CODE_HINTS.search(text))
    system = bool(_SYSTEM_HINTS.search(text))

    if code and not system:
        return "coding"
    if system and not code:
        return "general"

    # Ambiguo o nessun segnale: continuità di sessione.
    if last_agent in ("coding", "general"):
        return last_agent

    # Ultima risorsa: una micro-chiamata all'LLM (output limitato a pochi token).
    try:
        resp = provider.complete(
            [{"role": "system", "content": _ROUTER_PROMPT},
             {"role": "user", "content": text[:400]}],
            tools=None,
            think=False,
            max_tokens=4,
        )
        ans = (resp.content or "").strip().lower()
        return "coding" if "cod" in ans else "general"
    except Exception as exc:  # noqa: BLE001
        log.warning("Router LLM fallito: %s — default 'general'", exc)
        return "general"
