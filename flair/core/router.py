"""Router per scegliere l'agente (modalità) di un turno.

Strategia:
1. L'LLM decide direttamente, con UNA sola chiamata economica (modello non-thinking,
   output di pochi token). È il decisore primario perché più robusto di regole fisse:
   capisce quando una richiesta ha bisogno di accesso GLOBALE (→ general) anche se
   parla di file, evitando di incastrare l'utente nella modalità coding (confinata).
2. Se la chiamata fallisce (rete) o dà una risposta inattesa, si ripiega su
   un'euristica a parole chiave + continuità di sessione (sticky), così il routing
   non crasha mai ed è ragionevole anche offline.

Nel dubbio, default 'general' (un assistente generico sa anche conversare).
Le scelte esplicite dell'utente (/code, /do) non passano di qui: la CLI le forza.
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
    r"apri|avvia|lancia|riprodu|ascolt|suona|metti\s*su|"
    r"open|launch|play\b|"
    r"browser|sito\b|news\b|canzone|brano|musica|playlist|"
    r"appunti|clipboard|incolla|paste\b|screenshot|schermata|volume\b|"
    r"system\s*info|info\s*(?:di|sul)\s*sistema|spazio\s*su\s*disco|"
    r"memoria\s*ram|\bram\b|processi\b|che\s*ore|che\s*ora|data\s*(?:di\s*)?oggi|data\s*odierna|"
    r"what\s*time|today|"
    r"(?:trova|cerca|find)\w*[^.]{0,20}?(?:canzone|brano|musica|file|cartella|"
    r"documento|foto|immagine|video|song|folder|photo)"
    r")",
    re.IGNORECASE,
)

_ROUTER_PROMPT = (
    "Sei il router di un assistente. Scegli la modalità giusta per la richiesta e "
    "rispondi con UNA sola parola: 'coding' oppure 'general'.\n"
    "- 'coding': lavorare sul CODICE/PROGETTO nella cartella corrente — leggere, "
    "cercare, modificare o creare file del progetto, eseguire test, refactoring, git. "
    "Questa modalità è CONFINATA alla cartella del progetto.\n"
    "- 'general': qualsiasi cosa sull'INTERO computer o di conversazione — aprire "
    "app/URL, cercare o scrivere file FUORI dal progetto, info di sistema, appunti, "
    "ricerche sul web, domande generiche.\n"
    "Se la richiesta richiede accesso fuori dalla cartella del progetto, scegli "
    "'general'. Nel dubbio, o se è conversazione, scegli 'general'."
)


def _classify_heuristic(text: str, last_agent: str | None) -> str:
    """Fallback senza rete: parole chiave + continuità di sessione, default general."""
    code = bool(_CODE_HINTS.search(text))
    system = bool(_SYSTEM_HINTS.search(text))
    if code and not system:
        return "coding"
    if system and not code:
        return "general"
    if last_agent in ("coding", "general"):
        return last_agent
    return "general"


def classify(text: str, provider, last_agent: str | None = None) -> str:
    """Decide l'agente con una singola chiamata LLM economica; ripiega sull'euristica
    in caso di errore o risposta inattesa."""
    hint = ""
    if last_agent in ("coding", "general"):
        hint = f"\n\n(Modalità attuale: {last_agent}. Mantienila se la richiesta è coerente.)"
    try:
        resp = provider.complete(
            [{"role": "system", "content": _ROUTER_PROMPT},
             {"role": "user", "content": text[:600] + hint}],
            tools=None,
            think=False,
            max_tokens=2,
        )
        ans = (resp.content or "").strip().lower()
        if "cod" in ans:
            return "coding"
        if "gen" in ans:
            return "general"
        return _classify_heuristic(text, last_agent)  # risposta inattesa
    except Exception as exc:  # noqa: BLE001
        log.warning("Router LLM non disponibile (%s) — uso l'euristica.", exc)
        return _classify_heuristic(text, last_agent)
