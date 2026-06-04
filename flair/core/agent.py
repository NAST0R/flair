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
from dataclasses import dataclass, field

from ..llm import LLMProvider, LLMResponse, ToolCall, Usage, is_context_overflow
from .tool import ToolContext, ToolError, Toolset

log = logging.getLogger(__name__)

OnTool = Callable[[str, dict], None]
OnResult = Callable[[str, str, bool], None]
OnReasoning = Callable[[str], None]
OnText = Callable[[str], None]
OnDelta = Callable[[str], None]
OnCompact = Callable[[int, int], None]
Approve = Callable[[str, dict], bool | str]  # True = procedi, False = nega, "stop" = ferma il flusso


class StoppedByUser(Exception):
    """Sollevata quando l'utente sceglie 'stop' al prompt di conferma: il flusso
    agentico si ferma subito e il controllo torna all'utente."""

_COMPACT_PROMPT = (
    "Sei un compressore di contesto per un assistente AI. Riassumi la conversazione "
    "seguente in modo autosufficiente, così che l'assistente possa proseguire il "
    "lavoro senza aver letto l'originale. Mantieni: l'obiettivo/richiesta, i file "
    "esaminati con i contenuti e le firme rilevanti, le modifiche già applicate, le "
    "decisioni prese, gli errori incontrati, e lo stato attuale con i prossimi passi. "
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
        on_text: OnText | None = None,
        on_delta: OnDelta | None = None,
        on_compact: OnCompact | None = None,
        approve: Approve | None = None,
    ) -> None:
        self.name = name
        self.cfg = cfg
        self.provider = provider
        self.toolset = toolset
        self.system_prompt = system_prompt
        self.ctx = ToolContext(cfg=cfg)

        self.on_tool = on_tool
        self.on_result = on_result
        self.on_reasoning = on_reasoning
        self.on_text = on_text
        self.on_delta = on_delta
        self.on_compact = on_compact
        self.approve = approve

        # La memoria è condivisa: chi passa la stessa Conversation ai due agenti li fa
        # ragionare sulla stessa storia. Il system prompt è anteposto alla chiamata.
        self.convo = conversation if conversation is not None else Conversation()

    @property
    def messages(self) -> list[dict]:
        """La conversazione COSÌ COME viene inviata al modello: system prompt (di
        QUESTO agente) + storia condivisa. Vista di sola lettura."""
        return [{"role": "system", "content": self.system_prompt}, *self.convo.messages]

    def reset(self) -> None:
        self.convo.reset()

    # ── compaction / contesto ───────────────────────────────────────────────

    def compact(self) -> bool:
        """Compatta su richiesta esplicita (REPL /compact)."""
        return self._compact()

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

    def run(self, task: str, think: bool = False) -> AgentResult:
        self.convo.messages.append({"role": "user", "content": task})
        schemas = self.toolset.schemas()
        recent: dict[str, int] = {}
        step = 0
        turn_usage = Usage()
        resp: LLMResponse | None = None

        try:
            while step < self.cfg.max_steps:
                resp = self._complete(tools=schemas, think=think and step == 0)
                turn_usage = turn_usage + resp.usage

                if resp.reasoning and self.on_reasoning and not self._streaming():
                    self.on_reasoning(resp.reasoning)

                if not resp.has_tool_calls:
                    self.convo.messages.append({"role": "assistant", "content": resp.content})
                    return AgentResult(resp.content, turn_usage, step, "done")

                if resp.content and self.on_text and not self._streaming():
                    self.on_text(resp.content)

                step += 1
                self.convo.messages.append(self._assistant_msg(resp))
                try:
                    for tc in resp.tool_calls:
                        output, _ok = self._run_tool(tc, recent)
                        self.convo.messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                except StoppedByUser:
                    self._answer_unanswered(resp)
                    return AgentResult("", turn_usage, step, "stopped")

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
            return AgentResult("", turn_usage, step, "stopped")

        content, delta = self._force_final()
        return AgentResult(content, turn_usage + delta, step, "max_steps")

    # ── chiamata al modello (con compaction e gestione overflow) ──────────────

    def _streaming(self) -> bool:
        return bool(self.cfg.stream and self.on_delta)

    def _complete(self, tools, think) -> LLMResponse:
        self._maybe_compact()
        try:
            resp = self._raw_complete(tools, think)
        except Exception as exc:  # noqa: BLE001
            if is_context_overflow(exc) and self._compact(aggressive=True):
                log.warning("Overflow di contesto: compattato e ritento.")
                resp = self._raw_complete(tools, think)
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
        if self._ctx_estimate() > self.cfg.compact_threshold:
            self._compact()

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

    def _run_tool(self, tc: ToolCall, recent: dict[str, int]) -> tuple[str, bool]:
        name, args = tc.name, tc.arguments
        if self.on_tool:
            self.on_tool(name, args)

        t = self.toolset.get(name)
        if t is None:
            out = f"❌ Tool sconosciuto: {name}"
            if self.on_result:
                self.on_result(name, out, False)
            return out, False

        sig = hashlib.md5(json.dumps([name, args], sort_keys=True, default=str).encode()).hexdigest()[:12]
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
