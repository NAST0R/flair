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
  ogni turno. È l'opposto del vecchio Flair, che mutava il prefisso ogni turno.
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

_COMPACT_PROMPT = (
    "Sei un compressore di contesto per un assistente AI. Riassumi la conversazione "
    "seguente in modo autosufficiente, così che l'assistente possa proseguire il "
    "lavoro senza aver letto l'originale. Mantieni: l'obiettivo/richiesta, i file "
    "esaminati con i contenuti e le firme rilevanti, le modifiche già applicate, le "
    "decisioni prese, gli errori incontrati, l'eventuale piano/TODO con lo stato di "
    "ogni passo, e lo stato attuale con i prossimi passi. "
    "Se la conversazione contiene GIÀ un riassunto precedente, incorporane tutte le "
    "informazioni nel nuovo riassunto senza perderle. Sii completo sui fatti tecnici "
    "ma conciso. Non inventare nulla."
)


@dataclass
class AgentResult:
    content: str
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    stopped_reason: str = "done"   # done | max_steps | loop | stopped
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

    def reset(self) -> None:
        self.messages = []
        self.last_prompt_tokens = 0
        self.sent_upto = 0
        self.total_usage = Usage()

    def dump(self) -> dict:
        """Stato serializzabile (JSON) della conversazione e dell'uso cumulativo."""
        u = self.total_usage
        return {"messages": self.messages,
                "usage": {k: getattr(u, k) for k in _USAGE_FIELDS}}

    def load(self, state: dict) -> None:
        msgs = state.get("messages")
        if isinstance(msgs, list):
            self.messages = list(msgs)
        u = state.get("usage") or {}
        self.total_usage = Usage(**{k: int(u.get(k, 0)) for k in _USAGE_FIELDS})
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
                    f"⛔ Interrotto dall'utente: «{tc.name}» non è stato eseguito. "
                    "Il controllo è tornato all'utente; attendi nuove istruzioni."
                )})

    def run(self, task: str, think: bool = False, max_steps: int | None = None) -> AgentResult:
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
                resp = self._complete(tools=schemas, think=think and step == 0)
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
                            "\n\n[⚠ Output interrotto qui dal limite di lunghezza, non per scelta. "
                            "Se l'utente chiede di continuare, RIPRENDI esattamente da questo punto, "
                            "senza ricominciare né ripetere ciò che è già scritto sopra.]"
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
                    return AgentResult("", self._fold_delegated(turn_usage), step, "stopped")
                turn_usage = self._fold_delegated(turn_usage)

                if any(c >= 4 for c in recent.values()):
                    content, delta = self._force_final()
                    return AgentResult(content, turn_usage + delta, step, "loop")
                if any(c == 3 for c in recent.values()):
                    self.convo.messages.append({"role": "user", "content": (
                        "Stai ripetendo la stessa chiamata senza progredire. Fermati e "
                        "rispondi con quanto hai raccolto finora, indicando cosa non sei "
                        "riuscito a determinare."
                    )})
        except KeyboardInterrupt:
            # Ctrl-C in qualsiasi punto (anche a metà di un tool): manteniamo valida la
            # conversazione rispondendo agli eventuali tool_call ancora pendenti.
            if resp is not None and resp.has_tool_calls:
                self._answer_unanswered(resp)
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

    def _complete(self, tools, think) -> LLMResponse:
        self._maybe_compact()
        try:
            resp = self._raw_complete(tools, think)
        except Exception as exc:  # noqa: BLE001
            if is_context_overflow(exc):
                # Prima la potatura (gratis), poi il riassunto aggressivo: in overflow
                # ogni carattere conta e la potatura riduce anche l'input del riassunto.
                shrunk = self._prune_superseded() > 0
                shrunk = self._compact(aggressive=True) or shrunk
                if shrunk:
                    log.warning("Overflow di contesto: compattato e ritento.")
                    resp = self._raw_complete(tools, think)
                else:
                    raise
            else:
                raise
        self.convo.total_usage = self.convo.total_usage + resp.usage
        if resp.usage.prompt_tokens:
            self.convo.last_prompt_tokens = resp.usage.prompt_tokens
        self.convo.sent_upto = len(self.convo.messages)
        return resp

    def _raw_complete(self, tools, think) -> LLMResponse:
        streaming = self._streaming()
        return self.provider.complete(
            self.messages,
            tools=tools,
            think=think,
            stream=streaming,
            on_delta=self.on_delta if streaming else None,
            on_reasoning=self.on_reasoning if streaming else None,
        )

    # ── compaction ────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(msgs: list[dict]) -> int:
        chars = 0
        for m in msgs:
            chars += len(m.get("content") or "")
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                chars += len(fn.get("arguments", "")) + len(fn.get("name", ""))
        return chars // 4

    def _ctx_estimate(self) -> int:
        return self.convo.last_prompt_tokens + self._estimate_tokens(self.convo.messages[self.convo.sent_upto:])

    def _maybe_compact(self) -> None:
        if self._ctx_estimate() <= self.cfg.compact_threshold:
            return
        # Stadio 0: potatura deterministica degli output superati — gratis (nessuna
        # chiamata LLM) e senza perdita di fedeltà sul resto. Se basta a rientrare
        # sotto soglia, il riassunto (che sostituirebbe il dettaglio) non serve.
        if self._prune_superseded() and self._ctx_estimate() <= self.cfg.compact_threshold:
            return
        self._compact()

    def _prune_superseded(self) -> int:
        """Stub-ba gli output di tool provabilmente superati (vedi core/prune.py).
        La prima mutazione spezza il prefisso in cache da quel punto: azzeriamo i
        contatori così la stima del contesto riparte onesta (come per la compaction,
        che il prefisso lo spezzerebbe comunque)."""
        if not getattr(self.cfg, "compact_prune", True):
            return 0
        pruned = prune.prune_superseded(self.convo.messages)
        if pruned:
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
            summary = self._summarize(to_summarize)
        except Exception as exc:  # noqa: BLE001
            log.warning("Compaction fallita (%s): mantengo il contesto invariato.", exc)
            return False

        before = len(self.convo.messages)
        tail = self.convo.messages[split:]
        # Il system prompt non è nella storia (lo antepone ogni agente): qui sostituiamo
        # solo la parte vecchia con UN messaggio di riassunto. La testa resta stabile.
        self.convo.messages = (
            [{"role": "user", "content": "[Riassunto del lavoro svolto finora]\n\n" + summary}]
            + tail
        )
        self.convo.last_prompt_tokens = 0
        self.convo.sent_upto = 0
        if self.on_compact:
            self.on_compact(before, len(self.convo.messages))
        return True

    def _summarize(self, msgs: list[dict]) -> str:
        blob = self._render_for_summary(msgs)
        resp = self.provider.complete(
            [{"role": "system", "content": _COMPACT_PROMPT},
             {"role": "user", "content": "Conversazione da riassumere:\n\n" + blob}],
            tools=None,
            think=False,
            max_tokens=self.cfg.compact_summary_max_tokens,
        )
        self.convo.total_usage = self.convo.total_usage + resp.usage
        return resp.content or "(riassunto non disponibile)"

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
                    content = content[:800] + " …[troncato]"
                parts.append(f"[risultato tool] {content}")
            else:
                parts.append(f"[{role}] {m.get('content') or ''}")
        return "\n".join(parts)

    # ── interni ─────────────────────────────────────────────────────────────

    def _assistant_msg(self, resp: LLMResponse) -> dict:
        return {
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

    @staticmethod
    def _sig(name: str, args: dict) -> str:
        """Firma stabile di una chiamata (nome + argomenti) per il rilevamento dei loop."""
        return hashlib.md5(json.dumps([name, args], sort_keys=True, default=str).encode()).hexdigest()[:12]

    @staticmethod
    def _raw_args_error(name: str) -> str:
        """Messaggio azionabile quando gli argomenti sono ininterpretabili (di solito
        troncati perché l'output era troppo lungo)."""
        return (
            f"❌ Non sono riuscito a interpretare gli argomenti di «{name}»: probabilmente "
            "sono stati troncati perché l'output era troppo lungo (di solito quando si "
            "scrive un file molto grande in una sola chiamata). Riprova con un contenuto "
            "più conciso, oppure scrivi il file in più parti: prima write_file con la "
            "prima parte, poi le successive con write_file e append=true."
        )

    def _execute_pure(self, name: str, args: dict) -> tuple[str, bool, Usage]:
        """Esecuzione PURA di un tool, senza stato condiviso: usa un ToolContext ISOLATO
        (così i tool che delegano — explore — sommano l'usage su un canale proprio, mai
        in race con altri worker), non tocca self e non chiama callback. Pensata per
        girare in un thread del pool. Ritorna (output, ok, usage_delegato)."""
        t = self.toolset.get(name)
        ctx = ToolContext(cfg=self.cfg, provider=self.provider)
        ctx.delegated_usage = Usage()
        try:
            out = t(ctx, **args)
            ok = not out.startswith("❌")
        except ToolError as exc:
            out, ok = f"❌ {exc}", False
        except TypeError as exc:
            out, ok = f"❌ Argomenti non validi per {name}: {exc}", False
        except Exception as exc:  # noqa: BLE001
            out, ok = f"❌ Errore in {name}: {type(exc).__name__}: {exc}", False
            log.exception("Errore inatteso nel tool %s", name)
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
                precomputed[tc.id] = f"❌ Tool sconosciuto: {tc.name}"
            else:
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
            out = f"❌ Tool sconosciuto: {name}"
            if self.on_result:
                self.on_result(name, out, False)
            return out, False

        sig = self._sig(name, args)
        recent[sig] = recent.get(sig, 0) + 1

        if t.destructive and not self.cfg.auto_approve and self.approve:
            decision = self.approve(name, args)
            if decision == "stop":
                if self.on_result:
                    self.on_result(name, "⛔ interrotto dall'utente", False)
                raise StoppedByUser(name)
            if not decision:
                out = f"⚠️ Operazione '{name}' annullata dall'utente."
                if self.on_result:
                    self.on_result(name, out, False)
                return out, False

        try:
            out = t(self.ctx, **args)
            ok = not out.startswith("❌")
        except ToolError as exc:
            out, ok = f"❌ {exc}", False
        except TypeError as exc:
            out, ok = f"❌ Argomenti non validi per {name}: {exc}", False
        except Exception as exc:  # noqa: BLE001
            out, ok = f"❌ Errore in {name}: {type(exc).__name__}: {exc}", False
            log.exception("Errore inatteso nel tool %s", name)

        if self.on_result:
            self.on_result(name, out, ok)
        return out, ok

    def _force_final(self) -> tuple[str, Usage]:
        self.convo.messages.append({"role": "user", "content": (
            "Concludi ora: scrivi la risposta finale basandoti solo su ciò che hai "
            "effettivamente fatto/letto. Niente altre tool call."
        )})
        try:
            resp = self._complete(tools=None, think=False)
            self.convo.messages.append({"role": "assistant", "content": resp.content})
            return resp.content or "(nessuna risposta prodotta)", resp.usage
        except Exception as exc:  # noqa: BLE001
            return f"Interrotto. Errore nella sintesi finale: {exc}", Usage()
