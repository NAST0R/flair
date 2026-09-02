"""Motore agentico generico — uno solo, riusato da tutti gli agenti.

Ciclo:
    while step < max_steps:
        risposta = provider.complete(messaggi, tools)
        se non ci sono tool call → risposta finale, stop.
        altrimenti → esegui i tool, accoda i risultati, continua.

Efficienza e robustezza sui token:
- Messaggi APPEND-ONLY → la testa non cambia mai → cache del prefisso attiva.
- COMPACTION: quando il contesto supera una soglia (frazione della finestra del
  modello) la parte vecchia viene riassunta in UN messaggio e si riparte con un
  nuovo prefisso stabile. Si paga il cache-miss una volta per compaction, non a
  ogni turno — a differenza degli approcci naïf che invalidano la cache a ogni turno.
- La dimensione del contesto è misurata in modo esatto dai prompt_tokens
  restituiti dall'API (più una stima per i messaggi accodati dopo l'ultima
  chiamata): niente tokenizer da installare.
- Se il provider segnala comunque un overflow, si compatta in modo aggressivo e
  si ritenta una volta.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..llm import LLMProvider, LLMResponse, ToolCall, Usage, is_context_overflow
from . import prune
from .tool import ToolContext, ToolError, Toolset

log = logging.getLogger(__name__)

OnTool = Callable[[str, dict], None]
OnResult = Callable[[str, str, bool], None]
OnReasoning = Callable[[str], None]
OnDelta = Callable[[str], None]
OnCompact = Callable[[int, int], None]
OnPrune = Callable[[int], None]
Approve = Callable[[str, dict], bool | str]  # True = procedi, False = nega, "stop" = ferma il flusso


class StoppedByUser(Exception):
    """Sollevata quando l'utente sceglie 'stop' al prompt di conferma: il flusso
    agentico si ferma subito e il controllo torna all'utente."""

# Testi iniettati in conversazione dalla compaction (superficie model-facing:
# la guardia test_english_surface li asserisce direttamente).
_SUMMARY_HEADER = "[Summary of the work done so far]\n\n"
_ATTACHED_IMAGES_PREFIX = prune.ATTACHED_IMAGES_PREFIX  # unica fonte: la potatura lo riconosce
_IMAGE_REJECTED_NOTE = "[image removed: the endpoint rejected it]"

# Calibrazione del fattore caratteri→token (v. Conversation.token_ratio):
# peso del campione più recente, campione minimo per essere informativo, e limiti
# oltre i quali non si va nemmeno se le misure sono strane.
_RATIO_ALPHA = 0.3
_RATIO_MIN_SAMPLE = 200
_RATIO_MIN, _RATIO_MAX = 0.7, 2.5
_SUMMARIZE_PREAMBLE = "Conversation to summarize:\n\n"

_COMPACT_PROMPT = (
    "You are a context compressor for an AI assistant. Summarize the conversation "
    "below in a self-sufficient way, so the assistant can continue the work without "
    "having read the original. Keep: the goal/request, the files examined with the "
    "relevant contents and signatures, the edits already applied, the decisions "
    "made, the errors encountered, any plan/TODO with the status of each step, and "
    "the current status with the next steps. "
    "If the conversation ALREADY contains a previous summary, incorporate all of its "
    "information into the new summary without losing any of it. Be complete on "
    "technical facts but concise. Do not invent anything."
)

# Istruzione per il riassunto SUL PREFISSO IN CACHE: viene accodata alla
# conversazione COSÌ COM'È (system prompt dell'agente + storia byte-identica a
# quella già inviata) → il provider riusa la cache e si paga miss solo su queste
# righe, invece dell'intera storia ri-renderizzata: ~5x in meno sulle storie
# tool-heavy (dove il render tronca molto), fino a ~30x su quelle poco
# troncabili. In più il riassuntore vede gli output tool INTEGRALI, non gli
# stub a 800 caratteri (misurato: il render copre anche solo il 15% della storia).
def content_text(content, image_placeholder: str = "[image]") -> str:
    """Testo di un `content` che può essere una stringa (il caso normale) o una
    LISTA di parti multimodali OpenAI-style (messaggi con immagini: /img e
    view_image). Le parti immagine diventano un placeholder. È l'UNICO punto in
    cui il resto di flair — stima del contesto, render della compaction, recap,
    display — deve conoscere il formato multipart: tutto il resto chiama questo."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    out: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            out.append(part.get("text") or "")
        elif part.get("type") == "image_url":
            out.append(image_placeholder)
    return "\n".join(out)


def _count_images(content) -> int:
    """Numero di parti immagine in un content (0 per le stringhe)."""
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")


_SUMMARIZE_ON_PREFIX = (
    "Stop the current work. Summarize the ENTIRE conversation above in a "
    "self-sufficient way, so an assistant can continue without having read it. "
    "Keep: the goal/request, the files examined with the relevant contents and "
    "signatures, the edits already applied, the decisions made, the errors "
    "encountered, any plan/TODO with the status of each step, and the current "
    "status with the next steps. If the conversation already contains a previous "
    "summary, incorporate ALL of its information without losing any of it. Be "
    "complete on technical facts but concise. Do not invent anything. Reply with "
    "the summary as PLAIN TEXT ONLY: do not call any tools."
)


@dataclass
class AgentResult:
    content: str
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    stopped_reason: str = "done"   # done | max_steps | loop | stopped | budget
    truncated: bool = False        # True se la risposta finale è stata tagliata dal limite di output


_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens",
                 "cache_hit_tokens", "cache_miss_tokens", "reasoning_tokens")


@dataclass
class Conversation:
    """Memoria CONDIVISA dai due agenti: una sola conversazione, così passare da
    coding a general (o viceversa) NON perde il contesto né forza l'utente a
    ripetersi. Il system prompt NON sta qui — ogni agente antepone il proprio alla
    chiamata, mantenendo focalizzazione e (per coding) confinamento.

    Tiene anche il tracking esatto del contesto (token dell'ultima chiamata + indice
    di quanto era già stato inviato) e l'uso cumulativo della sessione, perché sono
    proprietà della conversazione, non del singolo agente.
    """
    messages: list[dict] = field(default_factory=list)
    last_prompt_tokens: int = 0   # dimensione esatta dell'ultimo contesto inviato
    sent_upto: int = 0            # indice di `messages` fin dove era già stato inviato
    total_usage: Usage = field(default_factory=Usage)
    # Rotture del prefisso in cache nella sessione (una per RIPRISTINO, non per
    # mutazione: prune+compaction nello stesso respiro = un solo evento). Col
    # listino a fasce ogni rottura è un evento economico: contarle le rende
    # visibili (/cost, JSONL) invece di doverle dedurre dal cache-hit% che cala.
    cache_breaks: int = 0
    # Fattore caratteri→token APPRESO dalle richieste reali: la stima statica
    # (chars//4) sottostima il codice denso, che tokenizza peggio della prosa — con
    # la soglia di compaction creduta lontana e il muro del modello raggiunto per
    # davvero (overflow → compaction d'emergenza, la più lossy). Qui si misura il
    # rapporto tra i prompt_tokens VERI e la nostra stima, sul DELTA tra due
    # chiamate: così le costanti presenti in entrambe (system prompt, schemi tool)
    # si cancellano e resta solo la conversione che ci interessa. 1.0 = nessuna
    # correzione (stato iniziale, e valore di ogni sessione senza dati).
    token_ratio: float = 1.0

    def reset(self) -> None:
        self.messages = []
        self.last_prompt_tokens = 0
        self.sent_upto = 0
        self.total_usage = Usage()
        self.cache_breaks = 0
        self.token_ratio = 1.0

    def dump(self) -> dict:
        """Stato serializzabile (JSON) della conversazione e dell'uso cumulativo."""
        u = self.total_usage
        return {"messages": self.messages,
                "usage": {k: getattr(u, k) for k in _USAGE_FIELDS},
                "cache_breaks": self.cache_breaks}

    def load(self, state: dict) -> None:
        msgs = state.get("messages")
        if isinstance(msgs, list):
            self.messages = list(msgs)
        u = state.get("usage") or {}
        self.total_usage = Usage(**{k: int(u.get(k, 0)) for k in _USAGE_FIELDS})
        try:
            self.cache_breaks = int(state.get("cache_breaks", 0))
        except (TypeError, ValueError):
            self.cache_breaks = 0
        self.last_prompt_tokens = 0
        self.sent_upto = 0


class Agent:
    def __init__(
        self,
        name: str,
        cfg,
        provider: LLMProvider,
        toolset: Toolset,
        system_prompt: str,
        conversation: Conversation | None = None,
        on_tool: OnTool | None = None,
        on_result: OnResult | None = None,
        on_reasoning: OnReasoning | None = None,
        on_reasoning_delta: OnReasoning | None = None,
        on_delta: OnDelta | None = None,
        on_compact: OnCompact | None = None,
        on_prune: OnPrune | None = None,
        approve: Approve | None = None,
    ) -> None:
        self.name = name
        self.cfg = cfg
        self.provider = provider
        self.toolset = toolset
        self.system_prompt = system_prompt

        self.on_tool = on_tool
        self.on_result = on_result
        self.on_reasoning = on_reasoning
        self.on_reasoning_delta = on_reasoning_delta
        self.on_delta = on_delta
        self.on_compact = on_compact
        self.on_prune = on_prune
        self.approve = approve

        # La memoria è condivisa: chi passa la stessa Conversation ai due agenti li fa
        # ragionare sulla stessa storia. Il system prompt è anteposto alla chiamata.
        self.convo = conversation if conversation is not None else Conversation()

        # Stato condiviso passato ai tool. Il provider serve ai tool che delegano a
        # un sub-agente (es. `explore`) per costruirlo; `delegated_usage` è il canale
        # con cui il tool riporta l'usage del sub-agente, che l'agente somma a turno
        # e sessione.
        self.ctx = ToolContext(cfg=cfg, provider=provider)
        self.ctx.delegated_usage = Usage()
        self.ctx.pending_images = []

    @property
    def messages(self) -> list[dict]:
        """La conversazione COSÌ COME viene inviata al modello: system prompt (di
        QUESTO agente) + storia condivisa. Vista di sola lettura."""
        return [{"role": "system", "content": self.system_prompt}, *self.convo.messages]

    def reset(self) -> None:
        self.convo.reset()

    # ── compaction / contesto ───────────────────────────────────────────────

    def compact(self) -> bool:
        """Compatta su richiesta esplicita (REPL /compact): prima la potatura
        deterministica (gratis), poi il riassunto LLM."""
        pruned = self._prune_superseded()
        return self._compact() or pruned > 0

    def context_fill(self) -> tuple[int, float]:
        """(token dell'ultimo contesto inviato, frazione della finestra) per la UI."""
        tokens = self._ctx_estimate()
        window = max(1, self.cfg.context_window)
        return tokens, min(1.0, tokens / window)

    # ── esecuzione ──────────────────────────────────────────────────────────

    def _answer_unanswered(self, resp: LLMResponse) -> None:
        """Risponde "interrotto" a ogni tool_call della risposta ancora senza esito.
        Ogni tool_call DEVE avere un messaggio 'tool', altrimenti la prossima chiamata
        API fallisce: così la conversazione resta valida e l'agente sa dove si è fermato."""
        answered = {m.get("tool_call_id") for m in self.convo.messages if m.get("role") == "tool"}
        for tc in resp.tool_calls:
            if tc.id not in answered:
                self.convo.messages.append({"role": "tool", "tool_call_id": tc.id, "content": (
                    f"⛔ Stopped by the user: «{tc.name}» was not executed. "
                    "Control has returned to the user; wait for new instructions."
                )})

    def run(self, task: str | list, think: bool = False, max_steps: int | None = None) -> AgentResult:
        # `task` è normalmente una stringa; con allegati (/img) è una LISTA di parti
        # multimodali OpenAI-style, che viaggia nel content così com'è.
        self.convo.messages.append({"role": "user", "content": task})
        schemas = self.toolset.schemas()
        recent: dict[str, int] = {}
        step = 0
        step_limit = max_steps if max_steps is not None else self.cfg.max_steps
        turn_usage = Usage()
        resp: LLMResponse | None = None

        try:
            while step < step_limit:
                # Budget hard: se il costo di sessione ha raggiunto il tetto, fermati
                # PRIMA della prossima chiamata a pagamento (no-op se max_cost=0). È il
                # freno che evita spese fuori controllo in esecuzione non presidiata.
                if self._over_budget():
                    return AgentResult("", self._fold_delegated(turn_usage), step, "budget")
                # Con --think, di default il modello thinking guida solo la mossa
                # d'apertura (step 0); con FLAIR_THINK_STEPS=all resta in cabina per
                # tutto il turno. Il knob modula SOLO i turni --think: senza --think
                # non forza mai nulla.
                deep = think and (step == 0 or self.cfg.think_steps == "all")
                resp = self._complete(tools=schemas, think=deep)
                turn_usage = turn_usage + resp.usage

                if resp.reasoning and self.on_reasoning and not self._streaming():
                    self.on_reasoning(resp.reasoning)

                if not resp.has_tool_calls:
                    truncated = resp.finish_reason == "length"
                    has_content = bool((resp.content or "").strip())
                    # Marcatore di continuazione SOLO se c'è del contenuto da proseguire.
                    # Se il troncamento è avvenuto nel ragionamento (contenuto vuoto), il
                    # marcatore non servirebbe (il reasoning non si riporta tra i turni) e
                    # anzi, accumulandosi, confonderebbe il modello: meglio non aggiungerlo.
                    stored = resp.content
                    if truncated and has_content:
                        stored = (stored or "") + (
                            "\n\n[⚠ Output cut here by the length limit, not by choice. "
                            "If the user asks to continue, RESUME exactly from this point, "
                            "without starting over or repeating what is already written above.]"
                        )
                    self.convo.messages.append({"role": "assistant", "content": stored})
                    return AgentResult(resp.content, turn_usage, step, "done", truncated=truncated)

                step += 1
                self.convo.messages.append(self._assistant_msg(resp))
                try:
                    if self._should_parallelize(resp.tool_calls):
                        # Batch di soli tool read-only e indipendenti → esecuzione
                        # concorrente (latenza ridotta su letture/ricerche/explore). Append,
                        # callback e usage restano nel thread principale, in ordine.
                        self._run_batch_parallel(resp.tool_calls, recent)
                    else:
                        for tc in resp.tool_calls:
                            output, _ok = self._run_tool(tc, recent)
                            self.convo.messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                except StoppedByUser:
                    self._answer_unanswered(resp)
                    self.ctx.pending_images = []   # niente allegati fuori contesto al turno dopo
                    return AgentResult("", self._fold_delegated(turn_usage), step, "stopped")
                turn_usage = self._fold_delegated(turn_usage)
                self._flush_pending_images()

                if any(c >= 4 for c in recent.values()):
                    content, delta = self._force_final()
                    return AgentResult(content, turn_usage + delta, step, "loop")
                if any(c == 3 for c in recent.values()):
                    self.convo.messages.append({"role": "user", "content": (
                        "You are repeating the same call without making progress. Stop "
                        "and answer with what you have gathered so far, stating what you "
                        "could not determine."
                    )})
        except KeyboardInterrupt:
            # Ctrl-C in qualsiasi punto (anche a metà di un tool): manteniamo valida la
            # conversazione rispondendo agli eventuali tool_call ancora pendenti.
            if resp is not None and resp.has_tool_calls:
                self._answer_unanswered(resp)
            self.ctx.pending_images = []
            return AgentResult("", self._fold_delegated(turn_usage), step, "stopped")

        content, delta = self._force_final()
        return AgentResult(content, turn_usage + delta, step, "max_steps")

    def _fold_delegated(self, turn_usage: Usage) -> Usage:
        """Somma UNA volta (turno + sessione) l'usage riportato dai tool che delegano
        a un sub-agente (ctx.delegated_usage), poi lo azzera. Va chiamata su OGNI
        uscita dal batch di tool — normale, stop dell'utente, Ctrl-C — perché i token
        delegati sono costo reale e non devono perdersi né finire attribuiti al turno
        sbagliato. A zero (nessuna delega) è un no-op."""
        d = self.ctx.delegated_usage
        if d is None:
            return turn_usage
        self.convo.total_usage = self.convo.total_usage + d
        self.ctx.delegated_usage = Usage()
        return turn_usage + d

    def _flush_pending_images(self) -> None:
        """Consegna al modello le immagini depositate dai tool (view_image). Il canale
        tool è solo-testo per protocollo: il tool risponde con una conferma e QUI il
        framework accoda UN messaggio utente multipart con le immagini vere — DOPO i
        risultati del batch (pairing tool_call/tool intatto), append-only come nudge
        e inventario. Al passo successivo il modello se le trova nel contesto."""
        staged = self.ctx.pending_images or []
        if not staged:
            return
        self.ctx.pending_images = []
        labels = ", ".join(s["label"] for s in staged)
        parts: list[dict] = [{"type": "text", "text": f"{_ATTACHED_IMAGES_PREFIX}{labels}]"}]
        parts.extend(s["part"] for s in staged)
        self.convo.messages.append({"role": "user", "content": parts})

    def _neutralize_rejected_images(self) -> None:
        """Rete di sicurezza per l'irriducibile: un'immagine che passa le validazioni
        locali ma che l'ENDPOINT rigetta con un 400 (es. un webp valido su llama.cpp,
        che non lo decodifica) resterebbe nella storia append-only e murerebbe la
        sessione — ogni richiesta successiva la rimanderebbe, rifallendo identica.
        Qui, SOLO su 400 non-overflow (deciso dal chiamante) e SOLO nel suffisso MAI
        inviato con successo, le parti immagine diventano un placeholder testuale:
        il turno fallisce una volta con l'errore vero a schermo, la sessione si
        auto-ripara. Nessuna rottura di cache: il suffisso mutato non era mai stato
        accettato dal server."""
        for m in self.convo.messages[self.convo.sent_upto:]:
            content = m.get("content")
            if m.get("role") == "user" and isinstance(content, list) and _count_images(content):
                m["content"] = [
                    {"type": "text", "text": _IMAGE_REJECTED_NOTE}
                    if isinstance(p, dict) and p.get("type") == "image_url" else p
                    for p in content
                ]
                log.warning("Endpoint rejected an image attachment: replaced with a placeholder.")

    def _over_budget(self) -> bool:
        """True se il costo cumulativo di sessione ha raggiunto il tetto `max_cost`
        (USD). A 0 (default) è disattivato: la modalità interattiva non è toccata.
        Il controllo usa il totale di sessione — la spesa reale mostrata all'utente —
        così il tetto vale sia per il singolo task sia per una sessione ripresa."""
        cap = getattr(self.cfg, "max_cost", 0.0) or 0.0
        if cap <= 0:
            return False
        return self.provider.estimate_cost(self.convo.total_usage, self.cfg) >= cap

    # ── chiamata al modello (con compaction e gestione overflow) ──────────────

    def _streaming(self) -> bool:
        return bool(self.cfg.stream and self.on_delta)

    def _learn_token_ratio(self, before_tokens: int, before_upto: int, real_tokens: int) -> None:
        """Aggiorna il fattore caratteri→token confrontando la crescita REALE del
        prompt con quella STIMATA per gli stessi messaggi. Si impara solo quando il
        confronto è sensato: prefisso precedente noto (nessuna compaction in mezzo),
        delta stimato non trascurabile e delta reale positivo. Media esponenziale
        (pesa il recente senza saltare su un singolo campione) e clamp: una stima
        sballata deve poter correggere, mai impazzire."""
        if before_tokens <= 0 or real_tokens <= 0:
            return                                  # nessun prefisso di riferimento
        est_delta = self._estimate_tokens(
            self.convo.messages[before_upto:],
            image_tokens=getattr(self.cfg, "image_token_estimate", 1200))
        real_delta = real_tokens - before_tokens
        if est_delta < _RATIO_MIN_SAMPLE or real_delta <= 0:
            return                                  # campione troppo piccolo per essere informativo
        sample = real_delta / est_delta
        blended = (1 - _RATIO_ALPHA) * self.convo.token_ratio + _RATIO_ALPHA * sample
        self.convo.token_ratio = min(_RATIO_MAX, max(_RATIO_MIN, blended))

    def _complete(self, tools, think, tool_choice: str | None = None) -> LLMResponse:
        self._maybe_compact()
        before_tokens, before_upto = self.convo.last_prompt_tokens, self.convo.sent_upto
        try:
            resp = self._raw_complete(tools, think, tool_choice)
        except Exception as exc:  # noqa: BLE001
            if is_context_overflow(exc):
                # Prima la potatura (gratis), poi il riassunto aggressivo: in overflow
                # ogni carattere conta e la potatura riduce anche l'input del riassunto.
                shrunk = self._prune_superseded() > 0
                shrunk = self._compact(aggressive=True) or shrunk
                if shrunk:
                    log.warning("Overflow di contesto: compattato e ritento.")
                    resp = self._raw_complete(tools, think, tool_choice)
                else:
                    raise
            else:
                if getattr(exc, "status_code", None) == 400:
                    self._neutralize_rejected_images()
                raise
        self.convo.total_usage = self.convo.total_usage + resp.usage
        if resp.usage.prompt_tokens:
            if getattr(self.cfg, "context_calibration", True):
                self._learn_token_ratio(before_tokens, before_upto, resp.usage.prompt_tokens)
            self.convo.last_prompt_tokens = resp.usage.prompt_tokens
        self.convo.sent_upto = len(self.convo.messages)
        return resp

    def _raw_complete(self, tools, think, tool_choice: str | None = None) -> LLMResponse:
        streaming = self._streaming()
        return self.provider.complete(
            self.messages,
            tools=tools,
            think=think,
            tool_choice=tool_choice,
            stream=streaming,
            on_delta=self.on_delta if streaming else None,
            on_reasoning=self.on_reasoning if streaming else None,
            on_reasoning_delta=self.on_reasoning_delta if streaming else None,
        )

    # ── compaction ────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(msgs: list[dict], image_tokens: int = 1200) -> int:
        chars = 0
        images = 0
        for m in msgs:
            # content_text, NON len(content): un'immagine base64 sono megabyte di
            # caratteri ma ~1-2K token reali (li decide l'encoder del server) — la
            # stima a caratteri esploderebbe innescando compaction a vuoto. Le
            # immagini si contano a costo fisso (image_tokens, FLAIR_IMAGE_TOKENS).
            content = m.get("content")
            chars += len(content_text(content, image_placeholder=""))
            images += _count_images(content)
            chars += len(m.get("reasoning_content") or "")   # tracce nei turni con tool
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                chars += len(fn.get("arguments", "")) + len(fn.get("name", ""))
        return chars // 4 + images * image_tokens

    def _ctx_estimate(self) -> int:
        """Token del contesto: la parte già inviata è ESATTA (prompt_tokens dell'ultima
        risposta), il suffisso accodato dopo è stimato — e corretto col fattore appreso
        dalle richieste reali di questa sessione (v. _learn_token_ratio)."""
        suffix = self._estimate_tokens(
            self.convo.messages[self.convo.sent_upto:],
            image_tokens=getattr(self.cfg, "image_token_estimate", 1200))
        ratio = self.convo.token_ratio if getattr(self.cfg, "context_calibration", True) else 1.0
        return self.convo.last_prompt_tokens + int(suffix * ratio)

    def _maybe_compact(self) -> None:
        if self._ctx_estimate() <= self.cfg.compact_threshold:
            return
        # Stadio 0: potatura deterministica degli output superati — nessuna chiamata
        # LLM e nessuna perdita di fedeltà sul resto. ATTENZIONE alla cache: la prima
        # mutazione ha GIÀ rotto il prefisso, quindi accettarla come unica misura
        # conviene solo se libera margine VERO (isteresi): a ridosso della soglia si
        # ricadrebbe in pochi step, pagando una SECONDA rottura ravvicinata — meglio
        # compattare ora, nello stesso respiro (la rottura è ormai pagata).
        if self._prune_superseded():
            margin = int(self.cfg.context_window * getattr(self.cfg, "prune_hysteresis_ratio", 0.10))
            if self._ctx_estimate() <= self.cfg.compact_threshold - margin:
                return
        self._compact()

    def _note_prefix_break(self) -> None:
        """Conta le rotture del prefisso in cache: una per RIPRISTINO, non per
        mutazione. Si conta solo se c'era davvero un prefisso inviato da rompere
        (contatori non già azzerati): così prune+compaction consecutivi — o una
        compaction a conversazione mai inviata — non gonfiano il contatore."""
        if self.convo.sent_upto or self.convo.last_prompt_tokens:
            self.convo.cache_breaks += 1

    def _prune_superseded(self) -> int:
        """Stub-ba gli output di tool provabilmente superati (vedi core/prune.py).
        La prima mutazione spezza il prefisso in cache da quel punto: azzeriamo i
        contatori così la stima del contesto riparte onesta (come per la compaction,
        che il prefisso lo spezzerebbe comunque)."""
        if not getattr(self.cfg, "compact_prune", True):
            return 0
        pruned = prune.prune_superseded(self.convo.messages)
        if pruned:
            self._note_prefix_break()
            self.convo.last_prompt_tokens = 0
            self.convo.sent_upto = 0
            if self.on_prune:
                self.on_prune(pruned)
        return pruned

    def _safe_split(self, keep_recent: int) -> int:
        """Indice (nella storia condivisa) da cui inizia la coda da preservare; mai su
        un messaggio 'tool' orfano (romperebbe il pairing tool_call/tool dell'API)."""
        msgs = self.convo.messages
        split = max(0, len(msgs) - keep_recent)
        while split < len(msgs) and msgs[split]["role"] == "tool":
            split += 1
        return split

    def _compact(self, aggressive: bool = False) -> bool:
        keep = 2 if aggressive else self.cfg.compact_keep_recent
        split = self._safe_split(keep)
        to_summarize = self.convo.messages[:split]
        if len(to_summarize) < 2:
            return False  # niente di sostanziale da comprimere

        try:
            summary = self._summarize(to_summarize, aggressive=aggressive)
        except Exception as exc:  # noqa: BLE001
            log.warning("Compaction fallita (%s): mantengo il contesto invariato.", exc)
            return False
        # Il riassunto LLM è lossy e NON conserva l'inventario: senza questo, dopo una
        # compattazione il modello ricostruisce "cosa esiste" da glob parziali e finisce
        # per asserire assenze false (es. "non ci sono test" su test letti un'ora prima).
        # L'inventario è estratto MECCANICAMENTE dalle tool call potate: esatto, zero LLM.
        inventory = self._read_inventory(to_summarize)
        if inventory:
            summary += "\n\nFiles already read (mechanical inventory, not from the summary): " + inventory

        before = len(self.convo.messages)
        tail = self.convo.messages[split:]
        # Il system prompt non è nella storia (lo antepone ogni agente): qui sostituiamo
        # solo la parte vecchia con UN messaggio di riassunto. La testa resta stabile.
        self._note_prefix_break()
        self.convo.messages = (
            [{"role": "user", "content": _SUMMARY_HEADER + summary}]
            + tail
        )
        self.convo.last_prompt_tokens = 0
        self.convo.sent_upto = 0
        if self.on_compact:
            self.on_compact(before, len(self.convo.messages))
        return True

    def _summarize(self, msgs: list[dict], aggressive: bool = False) -> str:
        """Riassunto della testa della conversazione, in due strategie.

        Percorso primario (cache-friendly): si accoda l'istruzione di riassunto alla
        conversazione COSÌ COM'È — stesso system prompt, stessa storia byte-identica,
        STESSI schemi tool del loop (partecipano al prefisso renderizzato) — così il
        provider riusa la cache e si paga miss solo sull'istruzione, non sull'intera
        storia (~5-30x in meno a compaction secondo quanto il render avrebbe
        troncato, e il riassuntore vede gli output tool integrali). Fallback al render legacy (compresso, prompt a sé): quando il
        modello risponde in modo anomalo (tool call / vuoto), quando il prefisso non
        entra nella finestra (overflow) e in modalità aggressive — dove per definizione
        il contesto è GIÀ troppo grande e solo il render compresso può passare."""
        if not aggressive:
            try:
                resp = self.provider.complete(
                    [{"role": "system", "content": self.system_prompt},
                     *msgs,
                     {"role": "user", "content": _SUMMARIZE_ON_PREFIX}],
                    tools=self.toolset.schemas(),
                    # Schemi INVIATI (prefisso identico → cache riusata) ma chiamate
                    # VIETATE: i modelli thinking, dopo decine di step a tool, tendono
                    # a rispondere all'istruzione con una tool call — e ogni volta si
                    # ripiegava sul render legacy (output tool troncati a 800 char),
                    # cioè il riassunto peggiore proprio dove serve il migliore.
                    tool_choice="none",
                    think=False,
                    max_tokens=self.cfg.compact_summary_max_tokens,
                )
                self.convo.total_usage = self.convo.total_usage + resp.usage
                if (resp.content or "").strip() and not resp.tool_calls:
                    return resp.content
                log.warning("Riassunto sul prefisso anomalo (tool call o vuoto): ripiego sul render.")
            except Exception as exc:  # noqa: BLE001
                if not is_context_overflow(exc):
                    raise
                log.warning("Riassunto sul prefisso in overflow: ripiego sul render compresso.")
        return self._summarize_rendered(msgs)

    def _summarize_rendered(self, msgs: list[dict]) -> str:
        blob = self._render_for_summary(msgs)
        resp = self.provider.complete(
            [{"role": "system", "content": _COMPACT_PROMPT},
             {"role": "user", "content": _SUMMARIZE_PREAMBLE + blob}],
            tools=None,
            think=False,
            max_tokens=self.cfg.compact_summary_max_tokens,
        )
        self.convo.total_usage = self.convo.total_usage + resp.usage
        return resp.content or "(summary unavailable)"

    @staticmethod
    def _read_inventory(msgs: list[dict], cap: int = 900) -> str:
        """Elenco deterministico dei path letti con read_file nei messaggi dati, in ordine
        di prima lettura; "(parziale)" se NESSUNA lettura di quel path è risultata completa
        (marcatori 'restano N righe' / 'output troncato' nel risultato). Tetto in caratteri:
        oltre, si chiude con il conteggio dei rimanenti."""
        id_to_path: dict[str, str] = {}
        status: dict[str, bool] = {}            # path -> almeno una lettura completa
        order: list[str] = []
        for m in msgs:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    if fn.get("name") != "read_file":
                        continue
                    try:
                        path = str(json.loads(fn.get("arguments") or "{}").get("path", "")).strip()
                    except (TypeError, ValueError):
                        path = ""
                    if path:
                        id_to_path[tc.get("id", "")] = path
                        if path not in status:
                            status[path] = False
                            order.append(path)
            elif m.get("role") == "tool":
                tpath = id_to_path.get(m.get("tool_call_id", ""))
                if tpath is not None:
                    content = m.get("content") or ""
                    if " more lines;" not in content and "output truncated" not in content:
                        status[tpath] = True
        parts: list[str] = []
        used = 0
        for i, path in enumerate(order):
            item = path + ("" if status.get(path) else " (partial)")
            if used + len(item) + 2 > cap:
                parts.append(f"… and {len(order) - i} more")
                break
            parts.append(item)
            used += len(item) + 2
        return ", ".join(parts)

    @staticmethod
    def _render_for_summary(msgs: list[dict]) -> str:
        parts: list[str] = []
        for m in msgs:
            role = m["role"]
            if role == "assistant" and m.get("tool_calls"):
                calls = ", ".join(
                    f"{tc['function']['name']}({tc['function']['arguments'][:200]})"
                    for tc in m["tool_calls"]
                )
                if m.get("content"):
                    parts.append(f"[assistant] {m['content']}")
                parts.append(f"[assistant→tool] {calls}")
            elif role == "tool":
                content = m.get("content") or ""
                if len(content) > 800:
                    content = content[:800] + " …[truncated]"
                parts.append(f"[tool result] {content}")
            else:
                # content_text: i messaggi utente possono essere multipart (immagini);
                # nel blob del riassunto un'immagine è solo il suo placeholder.
                parts.append(f"[{role}] {content_text(m.get('content'))}")
        return "\n".join(parts)

    # ── interni ─────────────────────────────────────────────────────────────

    def _assistant_msg(self, resp: LLMResponse) -> dict:
        msg: dict = {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in resp.tool_calls
            ],
        }
        # Protocollo thinking V4 (DeepSeek): nei turni CON tool call il reasoning
        # dell'assistant deve tornare all'API nelle richieste successive, così il
        # modello RIPRENDE il filo invece di ri-ragionare da zero sui soli artefatti
        # visibili (e i doc minacciano un 400 se manca). Nei turni SENZA tool il
        # server lo ignora: lo omettiamo per non pagare token inutili. I provider
        # che non accettano il campo lo spogliano a request-time (mai mutando la
        # cronologia: append-only e cache del prefisso restano intatti).
        if resp.tool_calls and resp.reasoning:
            msg["reasoning_content"] = resp.reasoning
        return msg

    def _loop_exempt(self, name: str) -> bool:
        """I tool di GESTIONE dei job in background sono interrogati più volte con
        gli stessi argomenti per natura (`job(action="check", id="j1")`): il
        rilevatore di loop, che conta le chiamate identiche e chiude il turno alla
        quarta, li fermerebbe proprio mentre stanno facendo il loro lavoro. L'avvio
        (distruttivo) resta invece sotto il rilevatore: quattro avvii identici sono
        quattro processi identici, e quello è un loop vero."""
        t = self.toolset.get(name)
        return bool(t and t.background and not t.destructive)

    @staticmethod
    def _sig(name: str, args: dict) -> str:
        """Firma stabile di una chiamata (nome + argomenti) per il rilevamento dei loop."""
        return hashlib.md5(json.dumps([name, args], sort_keys=True, default=str).encode()).hexdigest()[:12]

    @staticmethod
    def _raw_args_error(name: str) -> str:
        """Messaggio azionabile quando gli argomenti sono ininterpretabili (di solito
        troncati perché l'output era troppo lungo)."""
        return (
            f"❌ I could not parse the arguments for «{name}»: they were probably "
            "truncated because the output was too long (usually when writing a very "
            "large file in a single call). Retry with more concise content, or write "
            "the file in parts: first write_file with the first part, then the rest "
            "with write_file and append=true."
        )

    def _execute_pure(self, name: str, args: dict) -> tuple[str, bool, Usage]:
        """Esecuzione PURA di un tool, senza stato condiviso: usa un ToolContext ISOLATO
        (così i tool che delegano — explore — sommano l'usage su un canale proprio, mai
        in race con altri worker), non tocca self e non chiama callback. Pensata per
        girare in un thread del pool. Ritorna (output, ok, usage_delegato)."""
        t = self.toolset.get(name)
        if t is None:  # invariante garantita dal chiamante; guardia esplicita per sicurezza e tipi
            return f"❌ Unknown tool: {name}", False, Usage()
        ctx = ToolContext(cfg=self.cfg, provider=self.provider)
        ctx.delegated_usage = Usage()
        # Unica eccezione DELIBERATA alla purezza dei worker: la memoria di sessione è
        # condivisa per riferimento (serve a `remember` anche nei batch paralleli).
        # È sicura: list.append è atomico sotto il GIL; il caso patologico (due note
        # identiche nello stesso batch che scavalcano il dedup) produce al più un
        # doppione, ripulito alla serializzazione (SessionMemory.to_text).
        ctx.memory = self.ctx.memory
        # Stessa ragione, stessa eccezione: il registro dei job in background è UNO
        # per sessione e va condiviso per riferimento, altrimenti un batch di
        # `job(action="check")` — non distruttivo, quindi parallelizzabile — finisce
        # su un contesto isolato senza registro e risponde "no registry attached"
        # mentre i job stanno girando. È sicuro: BackgroundJobs ha il suo lock, e
        # ogni Job protegge il proprio buffer.
        ctx.jobs = self.ctx.jobs
        try:
            out = t(ctx, **args)
            ok = not out.startswith("❌")
        except ToolError as exc:
            out, ok = f"❌ {exc}", False
        except TypeError as exc:
            out, ok = f"❌ Invalid arguments for {name}: {exc}", False
        except Exception as exc:  # noqa: BLE001
            out, ok = f"❌ Error in {name}: {type(exc).__name__}: {exc}", False
            log.exception("Unexpected error in tool %s", name)
        return out, ok, (ctx.delegated_usage or Usage())

    def _should_parallelize(self, tcs: list[ToolCall]) -> bool:
        """Vero solo se conviene ed è SICURO eseguire il batch in parallelo: più di una
        chiamata e OGNI tool non distruttivo (read-only). Così niente gate di approvazione
        concorrente, niente dipendenze d'ordine (es. due edit sullo stesso file) e niente
        effetti collaterali da serializzare. Batch con anche un solo tool distruttivo, o
        singoli, restano sequenziali: comportamento identico a prima."""
        if not getattr(self.cfg, "parallel_tools", True) or len(tcs) < 2:
            return False
        for tc in tcs:
            if "_raw" in tc.arguments:
                continue                       # errore gestito a valle, non esegue nulla
            t = self.toolset.get(tc.name)
            if t is None:
                continue                       # sconosciuto: errore a valle, non esegue
            if t.destructive:
                return False
            if t.stages_media:
                # Deposita nella staging del ctx CONDIVISO (view_image): i worker
                # paralleli usano ctx isolati e l'allegato andrebbe perso.
                return False
        return True

    def _run_batch_parallel(self, tcs: list[ToolCall], recent: dict[str, int]) -> None:
        """Esegue in parallelo un batch di tool tutti non distruttivi e indipendenti.
        I worker fanno SOLO esecuzione pura (ctx isolato); TUTTO ciò che tocca stato
        condiviso — contatori anti-loop, callback UI, append dei messaggi, somma
        dell'usage — avviene qui nel thread principale, IN ORDINE. Niente callback dai
        thread → nessun output interlacciato né associazione errata dei risultati. I
        messaggi 'tool' sono accodati nell'ordine delle tool_call (non di completamento),
        preservando il pairing dell'API e una trascrizione deterministica."""
        cap = max(1, int(getattr(self.cfg, "parallel_tools_max_workers", 8) or 8))
        # Pre-fase (in ordine): chi non esegue nulla (argomenti illeggibili / tool
        # sconosciuto) ha un output di errore pronto; gli altri vanno eseguiti, e solo
        # per loro si incrementa il contatore anti-loop (come nel percorso sequenziale).
        precomputed: dict[str, str] = {}
        to_run: list[ToolCall] = []
        for tc in tcs:
            if "_raw" in tc.arguments:
                precomputed[tc.id] = self._raw_args_error(tc.name)
            elif self.toolset.get(tc.name) is None:
                precomputed[tc.id] = f"❌ Unknown tool: {tc.name}"
            else:
                if not self._loop_exempt(tc.name):
                    sig = self._sig(tc.name, tc.arguments)
                    recent[sig] = recent.get(sig, 0) + 1
                to_run.append(tc)

        results: dict[str, tuple[str, bool, Usage]] = {}
        if to_run:
            pool = ThreadPoolExecutor(max_workers=min(len(to_run), cap))
            try:
                futs = {pool.submit(self._execute_pure, tc.name, tc.arguments): tc for tc in to_run}
                try:
                    for f in as_completed(futs):
                        results[futs[f].id] = f.result()
                except BaseException:
                    # Ctrl-C nel thread principale: annulla i pendenti e rilancia. I worker
                    # già in esecuzione sono PURI (nessuna scrittura su stato condiviso né a
                    # video) → finiscono in background innocui, il loro esito viene scartato.
                    for f in futs:
                        f.cancel()
                    raise
            finally:
                pool.shutdown(wait=False)

        # Report (in ordine, main thread): callback appaiati per-tool (così l'handler che
        # aggiorna l'ultimo elemento resta corretto), append e somma dell'usage delegato.
        delegated = Usage()
        for tc in tcs:
            if self.on_tool:
                self.on_tool(tc.name, tc.arguments)
            if tc.id in precomputed:
                out, ok = precomputed[tc.id], False
            else:
                out, ok, used = results[tc.id]
                delegated = delegated + used
            if self.on_result:
                self.on_result(tc.name, out, ok)
            self.convo.messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        if delegated != Usage():
            self.ctx.delegated_usage = (self.ctx.delegated_usage or Usage()) + delegated

    def _run_tool(self, tc: ToolCall, recent: dict[str, int]) -> tuple[str, bool]:
        name, args = tc.name, tc.arguments
        if self.on_tool:
            self.on_tool(name, args)

        # Argomenti non interpretabili: parse_tool_args ripiega su {"_raw": ...} quando
        # il JSON degli argomenti è malformato o, molto più spesso, TRONCATO perché
        # l'output ha superato il limite di token (tipico scrivendo un file grande in
        # una sola chiamata). Diamo al modello un messaggio azionabile, prima del gate
        # di approvazione, così smette di ripetere la stessa chiamata destinata a fallire.
        if "_raw" in args:
            out = self._raw_args_error(name)
            if self.on_result:
                self.on_result(name, out, False)
            return out, False

        t = self.toolset.get(name)
        if t is None:
            out = f"❌ Unknown tool: {name}"
            if self.on_result:
                self.on_result(name, out, False)
            return out, False

        if not self._loop_exempt(name):
            sig = self._sig(name, args)
            recent[sig] = recent.get(sig, 0) + 1

        if t.destructive and not self.cfg.auto_approve and self.approve:
            decision = self.approve(name, args)
            if decision == "stop":
                if self.on_result:
                    self.on_result(name, "⛔ stopped by the user", False)
                raise StoppedByUser(name)
            if not decision:
                out = f"⚠️ Operation '{name}' was cancelled by the user."
                if self.on_result:
                    self.on_result(name, out, False)
                return out, False

        try:
            out = t(self.ctx, **args)
            ok = not out.startswith("❌")
        except ToolError as exc:
            out, ok = f"❌ {exc}", False
        except TypeError as exc:
            out, ok = f"❌ Invalid arguments for {name}: {exc}", False
        except Exception as exc:  # noqa: BLE001
            out, ok = f"❌ Error in {name}: {type(exc).__name__}: {exc}", False
            log.exception("Unexpected error in tool %s", name)

        if self.on_result:
            self.on_result(name, out, ok)
        return out, ok

    def _force_final(self) -> tuple[str, Usage]:
        self.convo.messages.append({"role": "user", "content": (
            "Conclude now: write the final answer based only on what you actually "
            "did/read. No more tool calls."
        )})
        try:
            # Gli schemi restano nella richiesta e le chiamate sono vietate da
            # tool_choice: il prefisso renderizzato non cambia, quindi la sintesi
            # finale non paga un cache-miss integrale (togliere `tools` lo pagava).
            resp = self._complete(tools=self.toolset.schemas(), think=False, tool_choice="none")
            self.convo.messages.append({"role": "assistant", "content": resp.content})
            return resp.content or "(no answer produced)", resp.usage
        except Exception as exc:  # noqa: BLE001
            return f"Stopped. Error in the final synthesis: {exc}", Usage()
