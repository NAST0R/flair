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
    r"browser|canzone|brano|musica|playlist|"
    r"appunti|clipboard|incolla|paste\b|screenshot|schermata|volume\b|"
    r"system\s*info|info\s*(?:di|sul)\s*sistema|spazio\s*su\s*disco|"
    r"memoria\s*ram|\bram\b|processi\b|che\s*ore|che\s*ora|data\s*(?:di\s*)?oggi|data\s*odierna|"
    r"what\s*time|today|"
    r"(?:trova|cerca|find)\w*[^.]{0,20}?(?:canzone|brano|musica|file|cartella|"
    r"documento|foto|immagine|video|song|folder|photo)"
    r")",
    re.IGNORECASE,
)

# "Costruire software": un verbo di creazione vicino a un artefatto sw è un segnale
# di coding inequivocabile (vale anche per un progetto nuovo o in una sottocartella),
# e ha la precedenza — risolve il caso "programmami un sito" che altrimenti, vedendo
# la parola "sito", finirebbe su general. Verbi volutamente NON ambigui (niente "fai",
# che vale anche "fai partire l'app"); i sostantivi usano i confini di parola, così
# "make an appointment" non scatta su "app".
_BUILD_RX = re.compile(
    r"\b(?:programm\w+|svilupp\w+|realizz\w+|implement\w+|cod(?:a|are|ifica\w*)|"
    r"crea\w*|scriv\w+|costru\w+|gener\w+|build|creat\w*|develop\w*|coding|make|write)\b"
    r"[^.\n]{0,40}?"
    r"\b(?:sito|site|web[\s-]?site|website|web[\s-]?app|webapp|applicazione|app|pagina|page|"
    r"landing|front[\s-]?end|back[\s-]?end|full[\s-]?stack|script|programma|software|gioco|"
    r"game|bot|cli|tool|libreria|library|pacchetto|package|componente|component|dashboard|"
    r"api|server|endpoint|interfaccia|widget)\b",
    re.IGNORECASE,
)

_ROUTER_PROMPT = (
    "Sei il router di un assistente con due modalità. Rispondi con UNA parola: "
    "'coding' o 'general'.\n"
    "- 'coding': SCRIVERE SOFTWARE — creare, leggere o modificare codice e costruire un "
    "progetto (sito web, app, script, gioco, tool…) nella cartella di lavoro, anche da zero "
    "o in una nuova sottocartella; più test, refactoring, git.\n"
    "- 'general': tutto il resto sul computer o di conversazione — aprire app/URL, "
    "riprodurre media, cercare/aprire file fuori dal progetto, info di sistema, appunti, "
    "ricerche web, scrivere testi non-codice, domande generiche.\n"
    "Programmare o costruire software → 'coding'. Nel dubbio, o se è conversazione → 'general'."
)


def _classify_heuristic(text: str, last_agent: str | None) -> str:
    """Fallback senza rete: parole chiave + continuità di sessione, default general."""
    if _BUILD_RX.search(text):       # "programma/costruisci un sito/app/script…" = coding
        return "coding"
    code = bool(_CODE_HINTS.search(text))
    system = bool(_SYSTEM_HINTS.search(text))
    if code and not system:
        return "coding"
    if system and not code:
        return "general"
    if last_agent in ("coding", "general"):
        return last_agent
    return "general"


def classify(text: str, provider, last_agent: str | None = None, convo=None) -> str:
    """Decide l'agente con una singola chiamata LLM economica; ripiega sull'euristica
    in caso di errore o risposta inattesa. Se è data una Conversation (`convo`),
    l'usage della chiamata viene sommato al totale di sessione: è piccolo (prompt
    corto e in cache, 2 token di output) ma è costo reale e va contato."""
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
        if convo is not None:
            convo.total_usage = convo.total_usage + resp.usage
        ans = (resp.content or "").strip().lower()
        if "cod" in ans:
            return "coding"
        if "gen" in ans:
            return "general"
        return _classify_heuristic(text, last_agent)  # risposta inattesa
    except Exception as exc:  # noqa: BLE001
        log.warning("Router LLM non disponibile (%s) — uso l'euristica.", exc)
        return _classify_heuristic(text, last_agent)
