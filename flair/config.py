"""Configurazione centralizzata.

Una `Config` costruita da variabili d'ambiente (.env). I parametri specifici di
ciascun provider vivono in `ProviderConfig` annidate. Aggiunge: finestra di
contesto e soglie di compaction, streaming, logging di sessione, prezzi
per-modello e chiavi per la ricerca web.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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


# Prezzi indicativi (USD per 1M token: cache-hit, input/cache-miss, output).
# Sono STIME e cambiano spesso: servono solo a mostrare un costo approssimato e
# sono sovrascrivibili via env (FLAIR_PRICE_*). Match per prefisso del nome.
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # DeepSeek (USD/1M: cache-hit, input, output). V4-flash è il workhorse;
    # V4-pro il reasoner di punta. Gli alias chat/reasoner mappano su V4-flash.
    "deepseek-v4-flash": (0.014, 0.14, 0.28),
    "deepseek-v4-pro": (0.0435, 0.435, 0.87),
    "deepseek-chat": (0.014, 0.14, 0.28),
    "deepseek-reasoner": (0.014, 0.14, 0.28),
    "deepseek-v4": (0.014, 0.14, 0.28),
    # OpenAI (approssimati, USD/1M; aggiornati ~2026, sovrascrivibili via env)
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
    "gpt-5.5-pro": (0.50, 5.00, 30.0),
    "gpt-5.5": (0.50, 5.00, 30.0),
    "gpt-5": (0.125, 1.25, 10.0),
    "o4-mini": (0.275, 1.10, 4.40),
    "o3-mini": (0.55, 1.10, 4.40),
    "o3": (0.50, 2.00, 8.00),
}
_PROVIDER_FALLBACK = {
    "deepseek": (0.014, 0.14, 0.28),
    "openai": (0.075, 0.15, 0.60),
}


def resolve_pricing(provider: str, model: str) -> tuple[float, float, float]:
    m = model.lower()
    best: tuple[float, float, float] | None = None
    best_len = -1
    for key, price in MODEL_PRICING.items():
        if m.startswith(key) and len(key) > best_len:
            best, best_len = price, len(key)
    return best or _PROVIDER_FALLBACK.get(provider, _PROVIDER_FALLBACK["deepseek"])


# Nomi dei file di istruzioni di progetto caricati nel prompt dell'agente coding.
PROJECT_INSTRUCTION_FILES = ("AGENTS.md", "FLAIR.md", "CLAUDE.md", ".flair.md")


@dataclass
class ProviderConfig:
    api_key: str
    model: str            # modello "veloce" per il loop a tool (non-thinking)
    think_model: str      # modello "thinking" usato quando serve ragionare
    base_url: str | None = None
    temperature: float = 0.0
    reasoning_effort: str | None = None  # solo reasoning model (low|medium|high)


@dataclass
class Config:
    provider: str
    deepseek: ProviderConfig
    openai: ProviderConfig

    # Generazione
    max_tokens: int = 8000
    request_timeout: int = 300
    stream: bool = True

    # Loop agentico
    max_steps: int = 60

    # Gestione del contesto (compaction)
    context_window: int = 120_000
    compact_threshold_ratio: float = 0.75   # compatta oltre questa frazione della finestra
    compact_keep_recent: int = 8            # messaggi recenti tenuti integri
    compact_summary_max_tokens: int = 2000

    # Filesystem / tool
    root: Path = Path(".")
    read_file_max_chars: int = 12000
    grep_max_chars: int = 6000
    command_max_chars: int = 8000
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

    # Sicurezza
    auto_approve: bool = False

    # Pricing (stima di costo, risolto dal modello veloce attivo)
    price_cache_hit: float = 0.028
    price_cache_miss: float = 0.28
    price_output: float = 0.42

    @property
    def active(self) -> ProviderConfig:
        return self.deepseek if self.provider == "deepseek" else self.openai

    @property
    def compact_threshold(self) -> int:
        return int(self.context_window * self.compact_threshold_ratio)

    def refresh_pricing(self) -> None:
        """Riallinea i prezzi al modello attivo; gli override via env (anche di un
        singolo campo) hanno la precedenza."""
        hit, miss, out = resolve_pricing(self.provider, self.active.model)
        self.price_cache_hit = _float("FLAIR_PRICE_CACHE_HIT", hit)
        self.price_cache_miss = _float("FLAIR_PRICE_CACHE_MISS", miss)
        self.price_output = _float("FLAIR_PRICE_OUTPUT", out)

    def validate(self) -> None:
        if self.provider not in ("deepseek", "openai"):
            raise RuntimeError(f"Provider non valido: {self.provider} (usa 'deepseek' o 'openai').")
        if not self.active.api_key:
            key_name = "DEEPSEEK_API_KEY" if self.provider == "deepseek" else "OPENAI_API_KEY"
            raise RuntimeError(
                f"{key_name} mancante. Crea un file .env (vedi .env.example) "
                "o esporta la variabile d'ambiente."
            )
        if not self.root.exists():
            raise RuntimeError(f"FLAIR_ROOT non esiste: {self.root}")


def load_config() -> Config:
    provider = os.getenv("FLAIR_PROVIDER", "deepseek").strip().lower()

    deepseek = ProviderConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        think_model=os.getenv("DEEPSEEK_THINK_MODEL", "deepseek-v4-pro"),
        temperature=_float("DEEPSEEK_TEMPERATURE", 0.0),
    )
    openai = ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        think_model=os.getenv("OPENAI_THINK_MODEL", "gpt-5-mini"),
        temperature=_float("OPENAI_TEMPERATURE", 0.0),
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT") or None,
    )

    log_dir = os.getenv("FLAIR_LOG_DIR")
    hit, miss, out = resolve_pricing(provider, deepseek.model if provider == "deepseek" else openai.model)

    cfg = Config(
        provider=provider,
        deepseek=deepseek,
        openai=openai,
        max_tokens=_int("FLAIR_MAX_TOKENS", 8000),
        request_timeout=_int("FLAIR_TIMEOUT", 300),
        stream=_bool("FLAIR_STREAM", True),
        max_steps=_int("FLAIR_MAX_STEPS", 60),
        context_window=_int("FLAIR_CONTEXT_WINDOW", 120_000),
        compact_threshold_ratio=_float("FLAIR_COMPACT_RATIO", 0.75),
        compact_keep_recent=_int("FLAIR_COMPACT_KEEP", 8),
        compact_summary_max_tokens=_int("FLAIR_COMPACT_SUMMARY_MAX", 2000),
        root=Path(os.getenv("FLAIR_ROOT", ".")).expanduser().resolve(),
        read_file_max_chars=_int("FLAIR_READ_MAX", 12000),
        grep_max_chars=_int("FLAIR_GREP_MAX", 6000),
        command_max_chars=_int("FLAIR_CMD_MAX", 8000),
        list_dir_max_entries=_int("FLAIR_LISTDIR_MAX", 200),
        search_max_results=_int("FLAIR_SEARCH_MAX", 80),
        search_max_scanned=_int("FLAIR_SEARCH_SCAN_MAX", 200_000),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        web_max_results=_int("FLAIR_WEB_MAX", 5),
        log_dir=Path(log_dir).expanduser().resolve() if log_dir else None,
        cost_warn=_float("FLAIR_COST_WARN", 0.0),
        session_dir=Path(os.getenv("FLAIR_SESSION_DIR", str(Path.home() / ".flair" / "sessions"))).expanduser().resolve(),
        auto_approve=_bool("FLAIR_AUTO_APPROVE", False),
        price_cache_hit=_float("FLAIR_PRICE_CACHE_HIT", hit),
        price_cache_miss=_float("FLAIR_PRICE_CACHE_MISS", miss),
        price_output=_float("FLAIR_PRICE_OUTPUT", out),
    )
    return cfg
