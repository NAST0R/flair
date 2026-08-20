"""Configurazione centralizzata.

Una `Config` costruita da variabili d'ambiente (.env). I parametri specifici di
ciascun provider vivono in `ProviderConfig` annidate. Aggiunge: finestra di
contesto e soglie di compaction, streaming, logging di sessione, prezzi
per-modello e chiavi per la ricerca web.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv assente: si usano le env già presenti
    pass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Host che parlano il protocollo first-party DeepSeek (passback del reasoning,
# thinking via extra_body, listino a fasce orarie). Tutti gli altri endpoint
# OpenAI-compatibili che servono i pesi DeepSeek ricevono il profilo standard.
_FIRST_PARTY_HOSTS = ("api.deepseek.com",)


def deepseek_first_party(base_url: str | None) -> bool:
    """True se l'endpoint parla il protocollo first-party DeepSeek. Dedotto
    dall'HOST dell'URL (match esatto sull'hostname, MAI substring: un proxy
    ostile 'api.deepseek.com.evil.io' non deve passare), con override esplicito
    FLAIR_DEEPSEEK_FIRST_PARTY=true|false (default: auto). L'override copre ciò
    che la deduzione non può conoscere — un endpoint ufficiale futuro non ancora
    in lista, o un reverse-proxy interno davanti all'API ufficiale — così un
    eventuale cambio di dominio si risolve con una riga di .env, zero codice."""
    override = os.getenv("FLAIR_DEEPSEEK_FIRST_PARTY", "auto").strip().lower()
    if override in ("true", "1", "yes"):
        return True
    if override in ("false", "0", "no"):
        return False
    host = (urlparse(base_url or "").hostname or "").lower()
    return host in _FIRST_PARTY_HOSTS


def _model_key(model: str) -> str:
    """Chiave di listino dal nome modello: gli slug dei reseller portano prefisso
    vendor e maiuscole ('deepseek-ai/DeepSeek-V4-Flash-0731') — si normalizza
    all'ultimo segmento minuscolo; il match per prefisso digerisce da solo i
    suffissi di versione (-0731, -0813)."""
    return model.rsplit("/", 1)[-1].strip().lower()


# Prezzi indicativi (USD per 1M token: cache-hit, input/cache-miss, output).
# Sono STIME e cambiano spesso: servono solo a mostrare un costo approssimato e
# sono sovrascrivibili via env (FLAIR_PRICE_*). Match per prefisso del nome.
#
# Dal 2026-08-16 (16:00 UTC) l'API ufficiale DeepSeek fattura per FASCE ORARIE:
# questa tabella è il listino OFF-PEAK (la base); le varianti PEAK (= 2x) vivono
# in MODEL_PRICING_PEAK e vengono selezionate da resolve_pricing in base all'ora
# UTC della richiesta. I listini senza fasce (OpenAI, local) restano piatti.
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # DeepSeek (USD/1M: cache-hit, input, output), listino OFF-PEAK. V4-flash è
    # il workhorse; V4-pro il reasoner di punta. Gli alias legacy mappano su flash.
    # Fonte: api-docs.deepseek.com/quick_start/pricing (verificati 2026-08-17).
    "deepseek-v4-flash": (0.007, 0.22, 0.66),
    "deepseek-v4-pro": (0.022, 0.66, 1.98),
    "deepseek-chat": (0.007, 0.22, 0.66),
    "deepseek-reasoner": (0.007, 0.22, 0.66),
    "deepseek-v4": (0.007, 0.22, 0.66),
    # OpenAI (approssimati, USD/1M; verificati 2026-07, sovrascrivibili via env)
    "gpt-4.1-nano": (0.025, 0.10, 0.40),
    "gpt-4.1-mini": (0.10, 0.40, 1.60),
    "gpt-4.1": (0.50, 2.00, 8.00),
    "gpt-4o-mini": (0.075, 0.15, 0.60),
    "gpt-4o": (1.25, 2.50, 10.0),
    "gpt-5-nano": (0.005, 0.05, 0.40),
    "gpt-5-mini": (0.025, 0.25, 2.00),
    "gpt-5.4-nano": (0.02, 0.20, 1.25),
    "gpt-5.4-mini": (0.075, 0.75, 4.50),
    "gpt-5.4": (0.25, 2.50, 15.0),
    "gpt-5.5-pro": (30.0, 30.0, 180.0),   # nessuno sconto cache sul tier Pro
    "gpt-5.5": (0.50, 5.00, 30.0),
    "gpt-5": (0.125, 1.25, 10.0),
    "o4-mini": (0.275, 1.10, 4.40),
    "o3-mini": (0.55, 1.10, 4.40),
    "o3": (0.50, 2.00, 8.00),
}
# Varianti PEAK (esattamente 2x l'off-peak, dal listino ufficiale). Solo i
# listini a fasce hanno una voce qui; chi manca resta piatto in ogni ora.
# Nota: vale per l'API ufficiale — i reseller (OpenRouter e simili) usano nomi
# con prefisso vendor che non matchano queste chiavi, e correttamente ricadono
# sul fallback piatto del provider.
MODEL_PRICING_PEAK: dict[str, tuple[float, float, float]] = {
    "deepseek-v4-flash": (0.014, 0.44, 1.32),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
    "deepseek-chat": (0.014, 0.44, 1.32),
    "deepseek-reasoner": (0.014, 0.44, 1.32),
    "deepseek-v4": (0.014, 0.44, 1.32),
}
_PROVIDER_FALLBACK = {
    "deepseek": (0.007, 0.22, 0.66),      # off-peak flash
    "openai": (0.075, 0.15, 0.60),
    "local": (0.0, 0.0, 0.0),   # inference locale: il costo vero è la bolletta
}
_PROVIDER_FALLBACK_PEAK = {
    "deepseek": (0.014, 0.44, 1.32),
}

# Fasce peak del listino DeepSeek, [inizio, fine) in ore UTC. Definite in UTC
# dal listino ufficiale — quindi immuni all'ora legale per costruzione (in
# Italia: 03-06 e 08-12 col DST estivo, un'ora prima in inverno).
_PEAK_RANGES_UTC: tuple[tuple[int, int], ...] = ((1, 4), (6, 10))


def is_peak_hour(when: datetime | None = None) -> bool:
    """True se `when` (default: adesso) cade nelle fasce peak DeepSeek.
    I datetime naive sono interpretati come UTC; quelli aware vengono convertiti."""
    now = when if when is not None else datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc)
    return any(a <= now.hour < b for a, b in _PEAK_RANGES_UTC)


def resolve_pricing(provider: str, model: str, when: datetime | None = None,
                    banded: bool = True) -> tuple[float, float, float]:
    """Listino per (provider, modello) nell'istante `when` (default: adesso).
    Match per prefisso più lungo sul nome NORMALIZZATO (gli slug vendor dei
    reseller agganciano la famiglia giusta). `banded` dice se il chiamante compra
    dall'API ufficiale DeepSeek: le fasce peak/off-peak sono un attributo di QUEL
    listino — i terzi prezzano flat e passano False (lo decide il provider dal
    proprio endpoint), altrimenti in peak la stima raddoppierebbe a torto."""
    m = _model_key(model)
    best_key: str | None = None
    best_len = -1
    for key in MODEL_PRICING:
        if m.startswith(key) and len(key) > best_len:
            best_key, best_len = key, len(key)
    peak = banded and is_peak_hour(when)
    if best_key is not None:
        if peak and best_key in MODEL_PRICING_PEAK:
            return MODEL_PRICING_PEAK[best_key]
        return MODEL_PRICING[best_key]
    if peak and provider in _PROVIDER_FALLBACK_PEAK:
        return _PROVIDER_FALLBACK_PEAK[provider]
    return _PROVIDER_FALLBACK.get(provider, _PROVIDER_FALLBACK["deepseek"])


def price_for(provider: str, model: str, when: datetime | None = None,
              banded: bool = True) -> tuple[float, float, float]:
    """Prezzi effettivi per UNA richiesta: listino del modello indicato nella
    fascia oraria corrente (se `banded`), con gli override env sempre vincenti
    campo per campo. In fascia peak vale la catena FLAIR_PRICE_*_PEAK >
    FLAIR_PRICE_* > listino peak; in off-peak (o su listini flat) FLAIR_PRICE_* >
    listino base. Serve all'attribuzione dei costi per-richiesta: in un turno
    --think si alternano fast e thinking, e un listino unico sottostimerebbe."""
    hit, miss, out = resolve_pricing(provider, model, when, banded=banded)
    hit = _float("FLAIR_PRICE_CACHE_HIT", hit)
    miss = _float("FLAIR_PRICE_CACHE_MISS", miss)
    out = _float("FLAIR_PRICE_OUTPUT", out)
    if banded and is_peak_hour(when):
        hit = _float("FLAIR_PRICE_CACHE_HIT_PEAK", hit)
        miss = _float("FLAIR_PRICE_CACHE_MISS_PEAK", miss)
        out = _float("FLAIR_PRICE_OUTPUT_PEAK", out)
    return (hit, miss, out)


# Nomi dei file di istruzioni di progetto caricati nel prompt dell'agente coding.
PROJECT_INSTRUCTION_FILES = ("AGENTS.md", "FLAIR.md", "CLAUDE.md", ".flair.md")


@dataclass
class ProviderConfig:
    api_key: str
    model: str            # modello "veloce" per il loop a tool (non-thinking)
    think_model: str      # modello "thinking" usato quando serve ragionare
    base_url: str | None = None
    temperature: float = 0.0
    reasoning_effort: str | None = None  # profondità per --think (deepseek: high|max; openai: low|medium|high)
    # Opt-in: profondità del ragionamento di DEFAULT del modello veloce (il flash
    # V4 pensa già a 'high' lato server anche senza parametri). None = intatto.
    fast_reasoning_effort: str | None = None
    # L'endpoint accetta immagini (content multipart OpenAI-style)? Esplicito
    # per-slot, MAI dedotto: solo chi gestisce l'endpoint sa se il server ha
    # l'encoder visivo caricato (es. llama-server con --mmproj). Default: no.
    vision: bool = False


@dataclass
class Config:
    provider: str
    deepseek: ProviderConfig
    openai: ProviderConfig
    local: ProviderConfig | None = None   # server locale OpenAI-compatibile (llama-server e simili)

    # Generazione
    max_tokens: int = 8000
    request_timeout: int = 300
    stream: bool = True
    # Bundle CA privato (PEM) per la verifica TLS di QUALSIASI provider: endpoint
    # self-hosted con certificato self-signed (llama-server --ssl-*) o reti con
    # TLS inspection che ri-firmano il traffico. Sostituisce SSL_CERT_FILE, che
    # httpx >= 0.28 non onora più. None = trust di default (certifi), invariato.
    ca_bundle: Path | None = None

    # Loop agentico
    max_steps: int = 60
    explorer_max_steps: int = 20
    # Con --think: 'first' (default) = modello thinking solo alla mossa d'apertura,
    # poi il loop prosegue sul veloce; 'all' = thinking per TUTTO il turno (ha
    # senso col passback del reasoning: ogni step eredita il filo). Opt-in.
    think_steps: str = "first"
    # Esecuzione concorrente dei tool: quando il modello emette più tool call in un
    # turno e sono TUTTE non distruttive (read-only), vengono eseguite in parallelo
    # (latenza ridotta su letture/ricerche/explore). I batch con tool distruttivi, o
    # singoli, restano sequenziali. L'output è identico (stesso ordine): cambia solo la
    # velocità. parallel_tools=False forza il vecchio comportamento sequenziale.
    parallel_tools: bool = True
    parallel_tools_max_workers: int = 8     # tetto di thread per batch (evita di saturare la rete)

    # Gestione del contesto (compaction)
    context_window: int = 120_000
    compact_threshold_ratio: float = 0.75   # compatta oltre questa frazione della finestra
    compact_keep_recent: int = 8            # messaggi recenti tenuti integri
    compact_summary_max_tokens: int = 2000
    compact_prune: bool = True              # stadio 0: pota gli output di tool superati prima del riassunto
    # Immagini (view_image / /img): lato massimo in px oltre cui ridimensionare
    # PRIMA dell'invio (richiede Pillow, opzionale: senza, si invia com'è) e stima
    # token per immagine usata dal contatore di contesto — il costo vero lo decide
    # l'encoder del server e torna nell'usage; questa serve solo alla compaction.
    image_max_side: int = 1536
    image_token_estimate: int = 1200

    # Isteresi dello stadio 0: la potatura da sola evita il riassunto SOLO se porta
    # il contesto sotto soglia con questo margine (frazione della finestra). La
    # mutazione ha comunque rotto il prefisso in cache: uscire a ridosso della
    # soglia significherebbe ricascarci in pochi step e pagare una SECONDA rottura
    # ravvicinata — a quel punto conviene compattare subito, nello stesso respiro.
    prune_hysteresis_ratio: float = 0.10

    # Filesystem / tool
    root: Path = Path(".")
    read_file_max_chars: int = 12000
    grep_max_chars: int = 6000
    command_max_chars: int = 8000
    repomap_max_chars: int = 8000
    list_dir_max_entries: int = 200
    search_max_results: int = 80
    search_max_scanned: int = 200_000

    # Ricerca web (agente generico)
    tavily_api_key: str | None = None
    web_max_results: int = 5

    # Osservabilità
    log_dir: Path | None = None
    cost_warn: float = 0.0                  # avviso quando il costo sessione supera questa soglia (USD); 0 = off
    session_dir: Path | None = None         # dove salvare/riprendere le sessioni
    # Memoria di sessione: fatti durevoli iniettati nel system prompt (prefisso
    # stabile → cache) e persistiti come sidecar accanto al JSON di sessione.
    # Vuota non inietta nulla; a False il tool `remember` non esiste proprio.
    memory_enabled: bool = True
    memory_max_chars: int = 4000            # tetto DURO del blocco memoria (~1k token)

    # Sicurezza
    auto_approve: bool = False
    read_only: bool = False                 # esecuzione non presidiata: nessun tool distruttivo
    max_cost: float = 0.0                   # tetto HARD di costo sessione (USD); 0 = nessun limite

    # Pricing (stima di costo, risolto dal modello veloce attivo)
    price_cache_hit: float = 0.028
    price_cache_miss: float = 0.28
    price_output: float = 0.42

    @property
    def active(self) -> ProviderConfig:
        if self.provider == "local" and self.local is not None:
            return self.local
        return self.deepseek if self.provider == "deepseek" else self.openai

    @property
    def compact_threshold(self) -> int:
        return int(self.context_window * self.compact_threshold_ratio)

    def refresh_pricing(self) -> None:
        """Riallinea i prezzi al modello attivo; gli override via env (anche di un
        singolo campo) hanno la precedenza. NOTA: è uno snapshot della fascia
        oraria del momento — serve solo da fallback per Usage senza attribuzione;
        il costo autorevole è quello per-richiesta (price_for a request-time)."""
        hit, miss, out = resolve_pricing(
            self.provider, self.active.model,
            banded=(self.provider != "deepseek" or deepseek_first_party(self.deepseek.base_url)))
        self.price_cache_hit = _float("FLAIR_PRICE_CACHE_HIT", hit)
        self.price_cache_miss = _float("FLAIR_PRICE_CACHE_MISS", miss)
        self.price_output = _float("FLAIR_PRICE_OUTPUT", out)

    def validate(self) -> None:
        if self.provider not in ("deepseek", "openai", "local"):
            raise RuntimeError(f"Invalid provider: {self.provider} (use 'deepseek', 'openai' or 'local').")
        if self.provider != "local" and not self.active.api_key:
            key_name = "DEEPSEEK_API_KEY" if self.provider == "deepseek" else "OPENAI_API_KEY"
            raise RuntimeError(
                f"{key_name} missing. Create a .env file (see .env.example) "
                "or export the environment variable."
            )
        if not self.root.exists():
            raise RuntimeError(f"FLAIR_ROOT does not exist: {self.root}")
        if self.ca_bundle is not None and not self.ca_bundle.is_file():
            raise RuntimeError(f"FLAIR_CA_BUNDLE does not exist: {self.ca_bundle}")


def _think_steps() -> str:
    val = (os.getenv("FLAIR_THINK_STEPS") or "first").strip().lower()
    if val not in ("first", "all"):
        raise ValueError(f"Invalid FLAIR_THINK_STEPS: {val!r} (use 'first' or 'all').")
    return val


def load_config() -> Config:
    provider = os.getenv("FLAIR_PROVIDER", "deepseek").strip().lower()

    deepseek = ProviderConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        think_model=os.getenv("DEEPSEEK_THINK_MODEL", "deepseek-v4-pro"),
        temperature=_float("DEEPSEEK_TEMPERATURE", 0.0),
        reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT") or None,
        fast_reasoning_effort=os.getenv("DEEPSEEK_FAST_REASONING_EFFORT") or None,
        vision=_bool("DEEPSEEK_VISION", False),
    )
    openai = ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        think_model=os.getenv("OPENAI_THINK_MODEL", "gpt-5-mini"),
        temperature=_float("OPENAI_TEMPERATURE", 0.0),
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT") or None,
        vision=_bool("OPENAI_VISION", False),
    )

    local_model = os.getenv("LOCAL_MODEL", "local")
    local = ProviderConfig(
        api_key=os.getenv("LOCAL_API_KEY", "local"),   # llama-server non la verifica
        base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8001/v1"),
        model=local_model,
        think_model=os.getenv("LOCAL_THINK_MODEL", local_model),   # un solo modello in VRAM
        temperature=_float("LOCAL_TEMPERATURE", -1.0),   # -1 = non inviare: vince il server
        vision=_bool("LOCAL_VISION", False),
    )

    log_dir = os.getenv("FLAIR_LOG_DIR")
    # Risolto SUBITO (prima del chdir alla root di lavoro): un path relativo
    # resta ancorato alla directory di lancio, prevedibile per l'utente.
    ca_bundle = os.getenv("FLAIR_CA_BUNDLE")

    cfg = Config(
        provider=provider,
        deepseek=deepseek,
        openai=openai,
        local=local,
        max_tokens=_int("FLAIR_MAX_TOKENS", 8000),
        request_timeout=_int("FLAIR_TIMEOUT", 300),
        stream=_bool("FLAIR_STREAM", True),
        ca_bundle=Path(ca_bundle).expanduser().resolve() if ca_bundle else None,
        max_steps=_int("FLAIR_MAX_STEPS", 60),
        explorer_max_steps=_int("FLAIR_EXPLORER_MAX_STEPS", 20),
        think_steps=_think_steps(),
        parallel_tools=_bool("FLAIR_PARALLEL_TOOLS", True),
        parallel_tools_max_workers=_int("FLAIR_PARALLEL_TOOLS_MAX", 8),
        context_window=_int("FLAIR_CONTEXT_WINDOW", 120_000),
        compact_threshold_ratio=_float("FLAIR_COMPACT_RATIO", 0.75),
        compact_keep_recent=_int("FLAIR_COMPACT_KEEP", 8),
        compact_summary_max_tokens=_int("FLAIR_COMPACT_SUMMARY_MAX", 2000),
        compact_prune=_bool("FLAIR_COMPACT_PRUNE", True),
        prune_hysteresis_ratio=_float("FLAIR_PRUNE_HYSTERESIS", 0.10),
        image_max_side=_int("FLAIR_IMAGE_MAX_SIDE", 1536),
        image_token_estimate=_int("FLAIR_IMAGE_TOKENS", 1200),
        root=Path(os.getenv("FLAIR_ROOT", ".")).expanduser().resolve(),
        read_file_max_chars=_int("FLAIR_READ_MAX", 12000),
        grep_max_chars=_int("FLAIR_GREP_MAX", 6000),
        command_max_chars=_int("FLAIR_CMD_MAX", 8000),
        repomap_max_chars=_int("FLAIR_REPOMAP_MAX", 8000),
        list_dir_max_entries=_int("FLAIR_LISTDIR_MAX", 200),
        search_max_results=_int("FLAIR_SEARCH_MAX", 80),
        search_max_scanned=_int("FLAIR_SEARCH_SCAN_MAX", 200_000),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        web_max_results=_int("FLAIR_WEB_MAX", 5),
        log_dir=Path(log_dir).expanduser().resolve() if log_dir else None,
        cost_warn=_float("FLAIR_COST_WARN", 0.0),
        session_dir=Path(os.getenv("FLAIR_SESSION_DIR", str(Path.home() / ".flair" / "sessions"))).expanduser().resolve(),
        memory_enabled=_bool("FLAIR_MEMORY", True),
        memory_max_chars=_int("FLAIR_MEMORY_MAX_CHARS", 4000),
        auto_approve=_bool("FLAIR_AUTO_APPROVE", False),
        read_only=_bool("FLAIR_READ_ONLY", False),
        max_cost=_float("FLAIR_MAX_COST", 0.0),
    )
    # Prezzi: un'unica fonte di verità (modello attivo + override FLAIR_PRICE_*),
    # la stessa usata quando si cambia provider/modello a runtime.
    cfg.refresh_pricing()
    return cfg
