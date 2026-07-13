"""Router per scegliere l'agente (modalità) di un turno.

Strategia:
0. Le continuazioni nude ("procedi", "ok", "vai", "go ahead"…) restano
   deterministicamente sull'agente corrente, SENZA chiamata LLM: non contengono
   alcun segnale di routing, e farle decidere a un modello può mandarle — a metà
   task — sull'agente sbagliato (sandbox, prompt e tool diversi) invalidando
   anche il prefisso in cache. In più si risparmiano la chiamata e la latenza.
1. Per il resto l'LLM decide direttamente, con UNA sola chiamata economica (modello non-thinking,
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
    "You are the router of an assistant with two modes. Reply with ONE word: "
    "'coding' or 'general'.\n"
    "- 'coding': WRITING SOFTWARE — creating, reading or modifying code and building a "
    "project (website, app, script, game, tool…) in the working folder, even from scratch "
    "or in a new subfolder; plus tests, refactoring, git.\n"
    "- 'general': everything else on the computer or conversational — opening apps/URLs, "
    "playing media, searching/opening files outside the project, system info, clipboard, "
    "web searches, writing non-code text, generic questions.\n"
    "Programming or building software → 'coding'. When in doubt, or if it is conversation → 'general'."
)


# Continuazioni/conferme "nude": messaggi composti SOLO da queste parole (e corti)
# non portano alcun segnale di routing → si resta sull'agente corrente, senza LLM.
# La regola è prudente per costruzione: basta UNA parola fuori lessico ("vai su
# google", "procedi col refactor di auth") e si torna al routing normale; e nel
# peggiore dei casi l'effetto è solo mantenere la modalità corrente, che l'utente
# può sempre forzare con /code o //do.
_CONTINUATION_WORDS = {
    # italiano
    "ok", "okay", "va", "bene", "sì", "si", "certo", "certamente", "perfetto",
    "ottimo", "esatto", "giusto", "d'accordo", "daccordo", "procedi", "procediamo",
    "prosegui", "proseguiamo", "continua", "continuiamo", "vai", "andiamo", "avanti",
    "dai", "fallo", "falla", "esegui", "eseguilo", "eseguila", "riprendi", "riprova",
    "ritenta", "ancora", "pure", "così", "cosi", "confermo", "confermato", "grazie",
    "prego", "favore", "per", "me",
    # inglese
    "yes", "yep", "yeah", "sure", "go", "on", "ahead", "proceed", "continue", "do",
    "it", "keep", "going", "carry", "next", "more", "again", "resume", "please", "for",
}

_WORD_RX = re.compile(r"[a-zà-ÿ']+")


def _is_bare_continuation(text: str) -> bool:
    """True se il messaggio è SOLO una continuazione/conferma (corto e composto
    interamente da parole del lessico): nessun contenuto da instradare."""
    t = text.strip().lower()
    if not t or len(t) > 40:
        return False
    words = _WORD_RX.findall(t)
    return bool(words) and all(w in _CONTINUATION_WORDS for w in words)


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
    """Decide l'agente del turno. Le continuazioni nude ("procedi", "ok", "vai"…)
    restano deterministicamente sull'agente corrente, senza chiamata LLM (vedi
    docstring del modulo). Per il resto decide l'LLM con una singola chiamata
    economica; in caso di errore o risposta inattesa si ripiega sull'euristica.
    Se è data una Conversation (`convo`), l'usage della chiamata viene sommato al
    totale di sessione: è piccolo (prompt corto e in cache, 2 token di output) ma
    è costo reale e va contato."""
    if last_agent in ("coding", "general") and _is_bare_continuation(text):
        return last_agent
    hint = ""
    if last_agent in ("coding", "general"):
        hint = f"\n\n(Current mode: {last_agent}. Keep it if the request is consistent.)"
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
        log.warning("LLM router unavailable (%s) — falling back to the heuristic.", exc)
        return _classify_heuristic(text, last_agent)
