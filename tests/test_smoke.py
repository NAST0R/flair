"""Suite di test offline (nessuna rete).

Esercita: normalizzazione usage dei due provider, parsing robusto args,
rilevamento reasoning model, router euristico, ed entrambi gli agenti
end-to-end (con un provider fittizio) sui tool reali — coding sandboxato e
generico cross-platform.
"""

from __future__ import annotations

import io
import json as json_module
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import APITimeoutError, BadRequestError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flair.agents import coding as coding_agent
from flair.agents import general as general_agent
from flair.config import load_config
from flair.core import router
from flair.core.agent import Conversation
from flair.core.tool import ToolContext, ToolError
from flair.llm import LLMResponse, ToolCall, Usage, is_context_overflow, parse_tool_args
from flair.llm.base import OpenAICompatProvider
from flair.llm.deepseek import DeepSeekProvider
from flair.llm.openai import OpenAIProvider
from flair.tools import web as web_tools
from flair.tools.fs import apply_edit

PASS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"FALLITO: {name} — {detail}"
    PASS.append(name)
    print(f"✓ {name}")



def _fake_response(content="ok", tool_calls=None, reasoning=None):
    msg = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7,
                            prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=5,
                            completion_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class _Recorder:
    """Finta `chat.completions` che registra i kwargs o solleva/sequenzia risposte."""

    def __init__(self, behavior=None):
        self.kwargs = None
        self.calls = 0
        self._behavior = behavior  # None | Exception | list

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls += 1
        b = self._behavior
        if b is None:
            return _fake_response()
        if isinstance(b, list):
            item = b[min(self.calls - 1, len(b) - 1)]
            if isinstance(item, BaseException):
                raise item
            return item
        if isinstance(b, BaseException):
            raise b
        return b


def _wire(provider, behavior=None) -> _Recorder:
    rec = _Recorder(behavior)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=rec))
    return rec


class FakeProvider:
    """Provider fittizio che restituisce una sequenza pre-programmata.

    Ogni elemento dello script può essere una LLMResponse o un'eccezione (che
    verrà sollevata). Accetta i kwargs del protocollo reale (max_tokens, stream,
    on_delta) e, se in streaming, emette il contenuto a pezzi su on_delta.
    """

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.seen = []
        self.calls = []  # kwargs di ogni chiamata

    def complete(self, messages, tools=None, think=False, max_tokens=None, stream=False,
                 on_delta=None, on_reasoning=None):
        self.seen.append([dict(m) for m in messages])
        self.calls.append({"think": think, "max_tokens": max_tokens, "stream": stream})
        if self.i >= len(self.script):
            return LLMResponse(content="FINE", usage=Usage(total_tokens=1))
        r = self.script[self.i]
        self.i += 1
        if isinstance(r, BaseException):
            raise r
        if stream and on_reasoning and r.reasoning:
            on_reasoning(r.reasoning)  # come il provider reale: ragionamento prima del contenuto
        if stream and on_delta and r.content:
            for ch in r.content:  # simula lo streaming carattere per carattere
                on_delta(ch)
        return r

    # Stesso calcolo del provider reale (serve al controllo di budget dell'agente).
    estimate_cost = OpenAICompatProvider.estimate_cost


_TC = [0]

def tc(name, **args):
    _TC[0] += 1
    return ToolCall(id=f"c{_TC[0]}_{name}", name=name, arguments=args)


def cfg_for(root: Path):
    cfg = load_config()
    cfg.deepseek.api_key = "fake"
    cfg.openai.api_key = "fake"
    cfg.root = root
    cfg.auto_approve = True
    return cfg


# ── 1. parsing robusto degli argomenti ───────────────────────────────────────

def test_arg_parse():
    check("args: dict passthrough", parse_tool_args({"a": 1}) == {"a": 1})
    check("args: json normale", parse_tool_args('{"path": "a.py"}') == {"path": "a.py"})
    check("args: doppia codifica", parse_tool_args('"{\\"x\\": 1}"') == {"x": 1})
    win = parse_tool_args('{"path": "C:\\Users\\me\\file.py"}')
    check("args: path Windows recuperato", win.get("path", "").endswith("file.py"), str(win))


# ── 2. normalizzazione usage (DeepSeek e OpenAI) ──────────────────────────────

def test_usage_normalization():
    ds = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, total_tokens=1200,
                         prompt_cache_hit_tokens=800, prompt_cache_miss_tokens=200,
                         completion_tokens_details=None)
    u = OpenAICompatProvider._usage(ds)
    check("usage DeepSeek: cache hit", u.cache_hit_tokens == 800, str(u))
    check("usage DeepSeek: cache miss", u.cache_miss_tokens == 200, str(u))

    oa = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, total_tokens=1200,
                         prompt_tokens_details=SimpleNamespace(cached_tokens=300),
                         completion_tokens_details=SimpleNamespace(reasoning_tokens=150))
    u2 = OpenAICompatProvider._usage(oa)
    check("usage OpenAI: cached→hit", u2.cache_hit_tokens == 300, str(u2))
    check("usage OpenAI: miss calcolato", u2.cache_miss_tokens == 700, str(u2))
    check("usage OpenAI: reasoning tokens", u2.reasoning_tokens == 150, str(u2))


# ── 3. rilevamento reasoning model + parametro token ──────────────────────────

def test_reasoning_detection():
    cfg = cfg_for(Path("."))
    cfg.provider = "deepseek"
    ds = DeepSeekProvider(cfg)
    check("deepseek: reasoner è reasoning", ds.is_reasoning_model("deepseek-reasoner"))
    check("deepseek: chat non è reasoning", not ds.is_reasoning_model("deepseek-chat"))
    check("deepseek: token param", ds.token_param == "max_tokens")

    cfg.provider = "openai"
    oa = OpenAIProvider(cfg)
    check("openai: o3 è reasoning", oa.is_reasoning_model("o3"))
    check("openai: o4-mini è reasoning", oa.is_reasoning_model("o4-mini"))
    check("openai: gpt-5-mini è reasoning", oa.is_reasoning_model("gpt-5-mini"))
    check("openai: gpt-5.1 è reasoning", oa.is_reasoning_model("gpt-5.1"))
    check("openai: gpt-4.1-mini NON è reasoning", not oa.is_reasoning_model("gpt-4.1-mini"))
    check("openai: gpt-4o NON è reasoning", not oa.is_reasoning_model("gpt-4o"))
    check("openai: token param", oa.token_param == "max_completion_tokens")
    check("openai: supporta reasoning_effort", oa.supports_reasoning_effort)
    check("deepseek: NON supporta reasoning_effort", not ds.supports_reasoning_effort)


# ── 4. router euristico (senza LLM) ───────────────────────────────────────────

def test_router():
    check("router: codice", router.classify("rifattorizza la funzione parse in main.py", None) == "coding")
    check("router: generico (browser)", router.classify("aprimi il browser su youtube", None) == "general")
    check("router: generico (canzone)", router.classify("trovami una canzone sul PC", None) == "general")
    check("router: sticky su ambiguo", router.classify("e adesso?", None, last_agent="coding") == "coding")
    # Coniugazioni/clitici italiani devono essere riconosciuti come azioni desktop.
    check("router: 'aprire' coniugato", router.classify("voglio aprire un sito di news", None) == "general")
    check("router: 'riproducimi'", router.classify("riproducimi un po' di musica", None) == "general")
    # 'trova' da solo non deve rubare una ricerca di codice.
    check("router: 'trova' non ruba il coding",
          router.classify("trova tutti gli usi della classe Parser", None, last_agent="coding") == "coding")
    # Costruire software (anche un progetto NUOVO o in una sottocartella) = coding,
    # nonostante parole come 'sito'/'app' che da sole sembrerebbero generiche.
    check("router: 'programma un sito' = coding",
          router.classify("programmi un sito web retro nella cartella corrente, sottocartella retrosite", None) == "coding")
    check("router: 'crea uno script' = coding",
          router.classify("creami uno script python che ordina i file", None) == "coding")
    check("router: \"sviluppa un'app\" = coding",
          router.classify("sviluppa un'app per la lista della spesa", None) == "coding")
    # Ma 'aprire'/'far partire' un sito o un'app resta general (niente falsi positivi).
    check("router: 'apri il sito' = general", router.classify("apri il sito della BBC", None) == "general")
    check("router: 'fai partire la app' = general", router.classify("fai partire la app di posta", None) == "general")


# ── 5. agente CODING end-to-end ───────────────────────────────────────────────

def test_coding_agent():
    root = Path("/tmp/flair3_coding")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    cfg = cfg_for(root)
    fake = FakeProvider([
        LLMResponse(tool_calls=[tc("list_directory", path=".")], usage=Usage(prompt_tokens=10, total_tokens=10)),
        LLMResponse(tool_calls=[tc("grep", pattern="def ", glob_filter="*.py")], usage=Usage(prompt_tokens=10, total_tokens=10)),
        LLMResponse(tool_calls=[tc("read_file", path="app.py")], usage=Usage(prompt_tokens=10, total_tokens=10)),
        LLMResponse(tool_calls=[tc("edit_file", path="app.py", old_string="return a + b", new_string="return a + b  # somma")], usage=Usage(prompt_tokens=10, total_tokens=10)),
        LLMResponse(tool_calls=[tc("write_file", path="pkg/new.py", content="X=1\n")], usage=Usage(prompt_tokens=10, total_tokens=10)),
        LLMResponse(content="Fatto.", usage=Usage(prompt_tokens=10, total_tokens=10)),
    ])
    agent = coding_agent.build(cfg, fake)
    res = agent.run("esplora e modifica")

    check("coding: termina 'done'", res.stopped_reason == "done", res.stopped_reason)
    check("coding: edit applicato", "# somma" in (root / "app.py").read_text())
    check("coding: write applicato", (root / "pkg" / "new.py").exists())

    # append-only: ogni snapshot è prefisso del successivo
    ok = all(b[: len(a)] == a for a, b in zip(fake.seen, fake.seen[1:], strict=False))
    check("coding: messaggi append-only", ok)

    # sandbox
    fake2 = FakeProvider([
        LLMResponse(tool_calls=[tc("read_file", path="../../../etc/passwd")], usage=Usage(total_tokens=1)),
        LLMResponse(content="bloccato", usage=Usage(total_tokens=1)),
    ])
    a2 = coding_agent.build(cfg, fake2)
    a2.run("leggi passwd")
    tmsgs = [m for m in a2.messages if m["role"] == "tool"]
    check("coding: sandbox blocca uscita", any("fuori dalla radice" in m["content"] for m in tmsgs))

    # edit ambiguo
    (root / "amb.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    fake3 = FakeProvider([
        LLMResponse(tool_calls=[tc("edit_file", path="amb.py", old_string="x = 1", new_string="x = 2")], usage=Usage(total_tokens=1)),
        LLMResponse(content="ambiguo", usage=Usage(total_tokens=1)),
    ])
    a3 = coding_agent.build(cfg, fake3)
    a3.run("edit ambiguo")
    tmsg = [m for m in a3.messages if m["role"] == "tool"][0]["content"]
    check("coding: edit ambiguo rifiutato", "occurs 2 times" in tmsg and (root / "amb.py").read_text() == "x = 1\nx = 1\n")

    # anti-loop
    loop = [LLMResponse(tool_calls=[tc("read_file", path="app.py")], usage=Usage(total_tokens=1)) for _ in range(10)]
    a4 = coding_agent.build(cfg, FakeProvider(loop + [LLMResponse(content="x", usage=Usage(total_tokens=1))] * 5))
    r4 = a4.run("loop")
    check("coding: anti-loop ferma", r4.stopped_reason == "loop" and r4.steps < cfg.max_steps, f"{r4.stopped_reason}/{r4.steps}")


# ── 6. agente GENERICO end-to-end ─────────────────────────────────────────────

def test_general_agent():
    root = Path("/tmp/flair3_general")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "song_test.mp3").write_text("fake", encoding="utf-8")
    (root / "note.txt").write_text("ciao\nmondo\n", encoding="utf-8")

    cfg = cfg_for(root)
    fake = FakeProvider([
        LLMResponse(tool_calls=[tc("get_datetime")], usage=Usage(total_tokens=1)),
        LLMResponse(tool_calls=[tc("system_info")], usage=Usage(total_tokens=1)),
        LLMResponse(tool_calls=[tc("search_files", query="song", extensions=[".mp3"], locations=[str(root)])], usage=Usage(total_tokens=1)),
        LLMResponse(tool_calls=[tc("read_file", path=str(root / "note.txt"))], usage=Usage(total_tokens=1)),
        LLMResponse(tool_calls=[tc("open_url", url="about:blank")], usage=Usage(total_tokens=1)),
        LLMResponse(content="Ecco i risultati.", usage=Usage(total_tokens=1)),
    ])
    agent = general_agent.build(cfg, fake)
    res = agent.run("che ore sono, info di sistema, trova la canzone, leggi note")

    tmsgs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    check("general: termina 'done'", res.stopped_reason == "done", res.stopped_reason)
    check("general: get_datetime ok", any("20" in t for t in tmsgs))  # un anno 20xx
    check("general: system_info ok", any("OS:" in t for t in tmsgs))
    check("general: search_files trova mp3", any("song_test.mp3" in t for t in tmsgs))
    check("general: read_file legge note", any("mondo" in t for t in tmsgs))
    check("general: open_url non crasha", any(("browser" in t.lower() or "about:blank" in t) for t in tmsgs))


# ── 7. gate di approvazione ───────────────────────────────────────────────────

def test_approval_gate():
    root = Path("/tmp/flair3_approval")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "f.py").write_text("a=1\n", encoding="utf-8")

    cfg = cfg_for(root)
    cfg.auto_approve = False  # forza il gate
    fake = FakeProvider([
        LLMResponse(tool_calls=[tc("write_file", path="f.py", content="HACKED\n")], usage=Usage(total_tokens=1)),
        LLMResponse(content="ok, annullato", usage=Usage(total_tokens=1)),
    ])
    agent = general_agent.build(cfg, fake, approve=lambda name, args: False)  # nega sempre
    agent.run("scrivi")
    check("approval: edit distruttivo negato non scrive", (root / "f.py").read_text() == "a=1\n")
    tmsg = [m for m in agent.messages if m["role"] == "tool"][0]["content"]
    check("approval: messaggio di annullamento", "cancelled" in tmsg)


# ── 9. matcher resiliente di edit_file ────────────────────────────────────────

def test_apply_edit():
    # esatto
    r, k = apply_edit("def f():\n    return 1\n", "return 1", "return 2")
    check("apply_edit: esatto", k == "exact" and "return 2" in r)
    # line-ending tolerant (trailing space nel file)
    r, k = apply_edit("def f():\n    x = 1   \n    return x\n", "    x = 1\n    return x",
                      "    x = 2\n    return x")
    check("apply_edit: fine-riga", k == "line-ending tolerant" and "x = 2" in r, k)
    # indentation tolerant + re-indentazione corretta
    src = "class A:\n    def m(self):\n        a = 1\n        b = 2\n        return a + b\n"
    r, k = apply_edit(src, "  a = 1\n  b = 2\n  return a + b", "  a = 10\n  b = 20\n  return a * b")
    check("apply_edit: indentazione", k == "indentation tolerant", k)
    check("apply_edit: re-indenta a 8 spazi", "        a = 10" in r and "        return a * b\n" in r, r)
    # ambiguo (match esatto multiplo)
    try:
        apply_edit("x = 1\nx = 1\n", "x = 1", "x = 2")
        check("apply_edit: ambiguo solleva", False)
    except ToolError:
        check("apply_edit: ambiguo solleva", True)
    # replace_all
    r, _ = apply_edit("x = 1\nx = 1\n", "x = 1", "x = 2", replace_all=True)
    check("apply_edit: replace_all", r == "x = 2\nx = 2\n", r)
    # non trovato
    try:
        apply_edit("a\n", "zzz", "y")
        check("apply_edit: non trovato solleva", False)
    except ToolError:
        check("apply_edit: non trovato solleva", True)


# ── 10. percorso reale del provider (kwargs inviati all'API) ──────────────────

def test_provider_request_path():
    cfg = cfg_for(Path("."))
    msgs = [{"role": "user", "content": "ciao"}]

    cfg.provider = "deepseek"
    ds = DeepSeekProvider(cfg)
    rec = _wire(ds)
    ds.complete(msgs, think=False)  # deepseek-v4-flash (non-thinking)
    check("provider ds-fast: max_tokens", "max_tokens" in rec.kwargs)
    check("provider ds-fast: ha temperature", "temperature" in rec.kwargs)
    check("provider ds-fast: no max_completion_tokens", "max_completion_tokens" not in rec.kwargs)
    check("provider ds-fast: nessun thinking (non-thinking)", "extra_body" not in rec.kwargs, str(rec.kwargs))
    ds.complete(msgs, think=True)  # deepseek-v4-pro: thinking via parametro
    check("provider ds-think: modello v4-pro", rec.kwargs["model"] == "deepseek-v4-pro", str(rec.kwargs.get("model")))
    check("provider ds-think: thinking attivo via extra_body",
          rec.kwargs.get("extra_body") == {"thinking": {"type": "enabled"}}, str(rec.kwargs.get("extra_body")))
    check("provider ds-think: temperature presente (DeepSeek la accetta)", "temperature" in rec.kwargs)
    ds.complete(msgs, max_tokens=4)
    check("provider: max_tokens override", rec.kwargs["max_tokens"] == 4, str(rec.kwargs.get("max_tokens")))

    # Alias legacy (deepseek-reasoner): la modalità è nel nome, niente extra_body.
    cfg.deepseek.think_model = "deepseek-reasoner"
    rec = _wire(ds)
    ds.complete(msgs, think=True)
    check("provider ds-legacy: nessun extra_body (modalità nel nome)", "extra_body" not in rec.kwargs, str(rec.kwargs))
    cfg.deepseek.think_model = "deepseek-v4-pro"

    cfg.provider = "openai"
    oa = OpenAIProvider(cfg)
    rec = _wire(oa)
    oa.complete(msgs, think=False)  # gpt-4.1-mini (non-reasoning)
    check("provider oai-fast: max_completion_tokens", "max_completion_tokens" in rec.kwargs)
    check("provider oai-fast: ha temperature", "temperature" in rec.kwargs)
    check("provider oai-fast: no max_tokens", "max_tokens" not in rec.kwargs)
    # think model = gpt-5-mini (reasoning): niente temperature, reasoning_effort auto 'medium'
    oa.complete(msgs, think=True)
    check("provider oai-think: no temperature", "temperature" not in rec.kwargs)
    check("provider oai-think: reasoning_effort auto medium", rec.kwargs.get("reasoning_effort") == "medium", str(rec.kwargs))
    # override esplicito di reasoning_effort
    cfg.openai.reasoning_effort = "high"
    oa.complete(msgs, think=True)
    check("provider oai-think: reasoning_effort override", rec.kwargs.get("reasoning_effort") == "high")
    cfg.openai.reasoning_effort = None
    # fast model reasoning (think=False) NON forza reasoning_effort
    oa.complete(msgs, think=False)
    check("provider oai-fast: niente reasoning_effort forzato", "reasoning_effort" not in rec.kwargs)

    # DeepSeek: MAI reasoning_effort (parametro OpenAI), MA sempre temperature.
    cfg.provider = "deepseek"
    rec = _wire(ds)
    ds.complete(msgs, think=True)  # deepseek-v4-pro
    check("provider ds: niente reasoning_effort", "reasoning_effort" not in rec.kwargs, str(rec.kwargs))
    check("provider ds: temperature presente", "temperature" in rec.kwargs)

    # retry: transitorio → ritenta; non-transitorio → propaga subito
    import flair.llm.base as base
    orig_sleep = base.time.sleep
    base.time.sleep = lambda *_: None
    try:
        rec = _wire(ds, [APITimeoutError(request=httpx.Request("POST", "https://x")), _fake_response("ok")])
        ds.complete(msgs)
        check("provider: retry su transitorio", rec.calls == 2, f"calls={rec.calls}")
        rec = _wire(ds, RuntimeError("boom"))
        try:
            ds.complete(msgs)
            check("provider: non-transitorio propaga", False)
        except RuntimeError:
            check("provider: non-transitorio propaga", rec.calls == 1, f"calls={rec.calls}")
    finally:
        base.time.sleep = orig_sleep

    # rilevatore di overflow del contesto
    err = BadRequestError("This model's maximum context length is 1000 tokens",
                          response=httpx.Response(400, request=httpx.Request("POST", "https://x")), body=None)
    check("overflow: rilevato", is_context_overflow(err))
    check("overflow: errore generico no", not is_context_overflow(RuntimeError("x")))


# ── 11. assemblaggio dello streaming ──────────────────────────────────────────

def test_streaming_assembly():
    cfg = cfg_for(Path("."))
    cfg.provider = "deepseek"
    ds = DeepSeekProvider(cfg)

    def chunk(content=None, tcs=None, usage=None):
        delta = SimpleNamespace(content=content, reasoning_content=None, tool_calls=tcs)
        choices = [] if (content is None and tcs is None and usage is not None) else [SimpleNamespace(delta=delta)]
        return SimpleNamespace(choices=choices, usage=usage)

    def tcd(idx, _id=None, name=None, args=None):
        fn = SimpleNamespace(name=name, arguments=args)
        return SimpleNamespace(index=idx, id=_id, function=fn)

    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7,
                            prompt_cache_hit_tokens=1, prompt_cache_miss_tokens=2,
                            completion_tokens_details=None)
    chunks = [
        chunk(content="Ciao "),
        chunk(content="mondo"),
        chunk(tcs=[tcd(0, "call_1", "read_file", '{"path"')]),
        chunk(tcs=[tcd(0, None, None, ':"a.py"}')]),
        chunk(usage=usage),
    ]
    rec = _Recorder()
    rec.create = lambda **kw: iter(chunks)  # type: ignore
    ds._client = SimpleNamespace(chat=SimpleNamespace(completions=rec))

    got = []
    resp = ds.complete([{"role": "user", "content": "x"}], stream=True, on_delta=got.append)
    check("stream: contenuto assemblato", resp.content == "Ciao mondo", resp.content)
    check("stream: on_delta ricevuto", "".join(got) == "Ciao mondo")
    check("stream: tool call assemblata", len(resp.tool_calls) == 1 and resp.tool_calls[0].name == "read_file")
    check("stream: argomenti tool parsati", resp.tool_calls[0].arguments == {"path": "a.py"}, str(resp.tool_calls[0].arguments))
    check("stream: usage dal chunk finale", resp.usage.prompt_tokens == 3 and resp.usage.cache_hit_tokens == 1)


# ── 12. compaction del contesto ───────────────────────────────────────────────

def test_compaction():
    cfg = cfg_for(Path("."))
    cfg.context_window = 1000      # soglia = 750
    cfg.compact_keep_recent = 1
    compacted = {}

    fake = FakeProvider([
        LLMResponse(tool_calls=[tc("read_file", path="app.py")], usage=Usage(prompt_tokens=5000, total_tokens=5000)),
        LLMResponse(content="RIASSUNTO DEL LAVORO", usage=Usage(prompt_tokens=10, total_tokens=10)),  # chiamata di sintesi
        LLMResponse(content="completato", usage=Usage(prompt_tokens=10, total_tokens=10)),
    ])
    agent = general_agent.build(cfg, fake, on_compact=lambda b, a: compacted.update(before=b, after=a))
    res = agent.run("task lungo")

    check("compaction: termina done", res.stopped_reason == "done", res.stopped_reason)
    check("compaction: callback invocato", "before" in compacted and compacted["after"] < compacted["before"], str(compacted))
    joined = "\n".join(m.get("content") or "" for m in agent.messages)
    check("compaction: riassunto inserito", "[Riassunto del lavoro svolto finora]" in joined and "RIASSUNTO DEL LAVORO" in joined)
    check("compaction: nessun 'tool' orfano in testa", agent.messages[1]["role"] != "tool")


# ── 13. _safe_split non lascia mai un 'tool' orfano ───────────────────────────

def test_safe_split():
    cfg = cfg_for(Path("."))
    agent = general_agent.build(cfg, FakeProvider([]))
    # La storia condivisa NON contiene il system prompt (lo antepone l'agente).
    agent.convo.messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "r1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "r2"},
    ]
    for keep in range(0, 7):
        split = agent._safe_split(keep)
        ok = split >= len(agent.convo.messages) or agent.convo.messages[split]["role"] != "tool"
        check(f"safe_split: keep={keep} non su tool", ok, f"split={split}")


# ── 14. overflow → compatta e ritenta ─────────────────────────────────────────

def test_overflow_retry():
    cfg = cfg_for(Path("."))
    overflow = BadRequestError("maximum context length exceeded",
                               response=httpx.Response(400, request=httpx.Request("POST", "https://x")), body=None)
    fake = FakeProvider([
        overflow,                                                   # 1) chiamata principale → overflow
        LLMResponse(content="riassunto", usage=Usage(total_tokens=5)),  # 2) sintesi della compaction
        LLMResponse(content="ripreso dopo compaction", usage=Usage(total_tokens=5)),  # 3) retry
    ])
    agent = general_agent.build(cfg, fake)
    # conversazione pregressa così che la compaction abbia qualcosa da comprimere
    agent.convo.messages += [
        {"role": "user", "content": "prima"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "contenuto vecchio molto lungo " * 5},
        {"role": "assistant", "content": "risposta precedente"},
    ]
    res = agent.run("nuovo task")
    check("overflow: recuperato senza crash", res.content == "ripreso dopo compaction", res.content)
    joined = "\n".join(m.get("content") or "" for m in agent.messages)
    check("overflow: ha compattato", "[Riassunto del lavoro svolto finora]" in joined)


# ── 15. ricerca web (cascata di backend, parser e gestione errori, offline) ───

def test_web_search():
    import urllib.request

    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    orig = urllib.request.urlopen
    orig_ddgs = web_tools.DDGS
    cfg = cfg_for(Path("."))
    cfg.tavily_api_key = None
    ctx = ToolContext(cfg=cfg)

    # scraping — markup endpoint html
    html_sample = (
        '<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Titolo &amp; A</a>'
        '<a class="result__snippet">Estratto <b>A</b>.</a>'
        '<a class="result__a" href="//example.org/b">Titolo B</a>'
        '<a class="result__snippet">Estratto B.</a>'
    )
    web_tools.DDGS = None  # forza il percorso scraping
    urllib.request.urlopen = lambda *a, **k: FakeResp(html_sample.encode())
    try:
        out = web_tools.web_search(ctx, query="qualcosa")
    finally:
        urllib.request.urlopen = orig
    check("web scrape html: titolo+url", "Titolo & A" in out and "https://example.com/a" in out, out)
    check("web scrape html: url protocollo-relativo", "https://example.org/b" in out, out)

    # scraping — markup endpoint lite
    lite_sample = (
        "<a rel='nofollow' class='result-link' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fsite.it%2Fx'>Notizia X</a>"
        "<td class='result-snippet'>Estratto X.</td>"
    )
    urllib.request.urlopen = lambda *a, **k: FakeResp(lite_sample.encode())
    try:
        out = web_tools.web_search(ctx, query="x")
    finally:
        urllib.request.urlopen = orig
    check("web scrape lite: parsing", "Notizia X" in out and "https://site.it/x" in out, out)

    # backend ddgs (fittizio) ha priorità sullo scraping
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, q, **kw):
            return [{"title": "DA DDGS", "href": "https://d.dg/1", "body": "corpo"}]
    web_tools.DDGS = FakeDDGS
    try:
        out = web_tools.web_search(ctx, query="y")
    finally:
        web_tools.DDGS = orig_ddgs
    check("web ddgs: usato e mappato", "DA DDGS" in out and "ddgs" in out, out)

    # nessun backend funziona → errore onesto, niente eccezioni
    web_tools.DDGS = None
    urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(OSError("rete"))
    try:
        out = web_tools.web_search(ctx, query="z")
    finally:
        urllib.request.urlopen = orig
        web_tools.DDGS = orig_ddgs
    check("web fail: errore onesto", out.startswith("❌ No results") and "TAVILY_API_KEY" in out, out)
    check("web fail: suggerisce ddgs", "pip install ddgs" in out, out)


# ── 16. persistenza sessioni, nuovi tool, switch runtime (offline) ────────────

def test_session_persistence():
    from flair.session_store import SessionStore
    root = Path("/tmp/flair_sess_test")
    shutil.rmtree(root, ignore_errors=True)
    cfg = cfg_for(root)
    cfg.session_dir = root / "sessions"
    prov = SimpleNamespace()
    agent = coding_agent.build(cfg, prov)
    agent.convo.messages.append({"role": "user", "content": "ciao"})
    agent.convo.messages.append({"role": "assistant", "content": "ciao a te"})
    agent.convo.total_usage = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                                    cache_hit_tokens=50, cache_miss_tokens=50, reasoning_tokens=5)

    store = SessionStore(cfg.session_dir)
    path = store.save("lavoro", {"last_agent": "coding", "conversation": agent.convo.dump()})
    check("sessione: salvataggio crea file", path is not None and path.exists())
    check("sessione: exists()", store.exists("lavoro"))
    check("sessione: latest()", store.latest() == "lavoro")
    # Scrittura atomica: il file finale è JSON valido e nessun temporaneo (.tmp) resta.
    import json as _json
    _json.loads(path.read_text(encoding="utf-8"))
    leftovers = [p.name for p in cfg.session_dir.iterdir() if p.suffix == ".tmp"]
    check("sessione: scrittura atomica senza .tmp residui", not leftovers, str(leftovers))
    store.save("lavoro", {"last_agent": "general", "conversation": agent.convo.dump()})  # sovrascrittura
    check("sessione: sovrascrittura atomica ancora valida",
          _json.loads(path.read_text(encoding="utf-8")).get("last_agent") == "general")

    agent2 = coding_agent.build(cfg, prov)
    state = store.load("lavoro")
    agent2.convo.load(state["conversation"])
    check("sessione: messaggi ripristinati",
          [m.get("content") for m in agent2.messages[-2:]] == ["ciao", "ciao a te"])
    check("sessione: uso ripristinato",
          agent2.convo.total_usage.total_tokens == 120 and agent2.convo.total_usage.cache_hit_tokens == 50)
    check("sessione: caricamento mancante → None", store.load("inesistente") is None)


def test_cli_session_roundtrip():
    from flair.cli import CLI
    root = Path("/tmp/flair_cli_sess")
    shutil.rmtree(root, ignore_errors=True)
    cfg = cfg_for(root)
    cfg.session_dir = root / "sessions"
    cli = CLI(cfg)
    cli.convo.messages.append({"role": "user", "content": "ricordami"})
    cli.last_agent = "general"
    cli.session_name = "s1"
    cli._save_session()

    cli2 = CLI(cfg)
    check("cli sessione: load ok", cli2._load_session("s1"))
    check("cli sessione: messaggio presente",
          any(m.get("content") == "ricordami" for m in cli2.convo.messages))
    check("cli sessione: last_agent ripreso", cli2.last_agent == "general")


def test_multi_edit():
    from flair.tools import coding as coding_tools
    root = Path("/tmp/flair_me")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    ctx = ToolContext(cfg=cfg_for(root))

    out = coding_tools.multi_edit(ctx, path="a.py", edits=[
        {"old_string": "x = 1", "new_string": "x = 10"},
        {"old_string": "z = 3", "new_string": "z = 30"},
    ])
    check("multi_edit: due modifiche applicate", out.startswith("✓") and "2 edits" in out)
    check("multi_edit: contenuto corretto", (root / "a.py").read_text() == "x = 10\ny = 2\nz = 30\n")

    before = (root / "a.py").read_text()
    out = coding_tools.multi_edit(ctx, path="a.py", edits=[
        {"old_string": "x = 10", "new_string": "x = 99"},
        {"old_string": "NON_ESISTE", "new_string": "!"},
    ])
    check("multi_edit: fallimento indica la modifica", out.startswith("❌ Edit #2"))
    check("multi_edit: atomico (nessuna scrittura su errore)", (root / "a.py").read_text() == before)


def test_web_fetch():
    import urllib.request

    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    ctx = ToolContext(cfg=cfg_for(Path(".")))
    sample = b"<body><h1>Titolo</h1><p>Primo &amp; secondo.</p><script>bad()</script></body>"
    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: FakeResp(sample)
    try:
        out = web_tools.web_fetch(ctx, url="example.com")
    finally:
        urllib.request.urlopen = orig
    check("web_fetch: estrae testo", "Titolo" in out and "Primo & secondo." in out)
    check("web_fetch: rimuove script", "bad()" not in out)

    urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(OSError("giù"))
    try:
        out = web_tools.web_fetch(ctx, url="https://x")
    finally:
        urllib.request.urlopen = orig
    check("web_fetch: errore pulito", out.startswith("❌ Could not fetch"))


def test_runtime_switch_and_context():
    from flair.llm import create_provider
    cfg = cfg_for(Path("."))
    cfg.provider = "deepseek"
    check("switch: factory deepseek", isinstance(create_provider(cfg), DeepSeekProvider) and cfg.active is cfg.deepseek)
    cfg.provider = "openai"
    check("switch: factory openai", isinstance(create_provider(cfg), OpenAIProvider) and cfg.active is cfg.openai)

    prov = SimpleNamespace()
    agent = coding_agent.build(cfg, prov)
    agent.convo.messages.append({"role": "user", "content": "x" * 4000})
    tokens, frac = agent.context_fill()
    check("contesto: tokens>0", tokens > 0)
    check("contesto: frazione 0..1", 0.0 <= frac <= 1.0)


def test_streaming_reasoning_order():
    cfg = cfg_for(Path("."))
    cfg.provider = "deepseek"
    ds = DeepSeekProvider(cfg)

    def rchunk(reasoning=None, content=None, tcs=None, usage=None):
        delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tcs)
        empty = content is None and reasoning is None and tcs is None and usage is not None
        choices = [] if empty else [SimpleNamespace(delta=delta)]
        return SimpleNamespace(choices=choices, usage=usage)

    def tcd(idx, _id=None, name=None, args=None):
        return SimpleNamespace(index=idx, id=_id, function=SimpleNamespace(name=name, arguments=args))

    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7,
                            prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=3, completion_tokens_details=None)

    # caso 1: ragionamento (arriva prima) → poi testo. on_reasoning DEVE precedere on_delta.
    chunks = [rchunk(reasoning="penso… "), rchunk(reasoning="ancora"),
              rchunk(content="Ecco "), rchunk(content="la risposta"), rchunk(usage=usage)]
    rec = _Recorder()
    rec.create = lambda **kw: iter(chunks)  # type: ignore
    ds._client = SimpleNamespace(chat=SimpleNamespace(completions=rec))
    events = []
    resp = ds.complete([{"role": "user", "content": "x"}], stream=True,
                       on_delta=lambda p: events.append(("delta", p)),
                       on_reasoning=lambda r: events.append(("reason", r)))
    check("stream: ragionamento PRIMA del contenuto", events[0] == ("reason", "penso… ancora"), str(events))
    check("stream: poi il contenuto", events[1][0] == "delta" and resp.content == "Ecco la risposta", str(events))
    check("stream: ragionamento emesso una sola volta", sum(1 for e in events if e[0] == "reason") == 1, str(events))

    # caso 2: ragionamento + tool, nessun testo → ragionamento emesso comunque, prima del tool.
    chunks2 = [rchunk(reasoning="rifletto sul tool"),
               rchunk(tcs=[tcd(0, "c1", "read_file", '{"path":"a.py"}')]), rchunk(usage=usage)]
    rec2 = _Recorder()
    rec2.create = lambda **kw: iter(chunks2)  # type: ignore
    ds._client = SimpleNamespace(chat=SimpleNamespace(completions=rec2))
    ev2 = []
    resp2 = ds.complete([{"role": "user", "content": "x"}], stream=True,
                        on_delta=lambda p: ev2.append(("delta", p)),
                        on_reasoning=lambda r: ev2.append(("reason", r)))
    check("stream tool-only: ragionamento emesso", ("reason", "rifletto sul tool") in ev2)
    check("stream tool-only: tool assemblato", bool(resp2.tool_calls) and resp2.tool_calls[0].name == "read_file")

    # caso 3: provider OpenAI, campo `reasoning` (convenzione OpenAI/GPT-OSS) al posto di reasoning_content.
    cfg.provider = "openai"
    oa = OpenAIProvider(cfg)

    def ochunk(reasoning=None, content=None, usage=None):
        delta = SimpleNamespace(content=content, reasoning=reasoning, tool_calls=None)  # niente reasoning_content
        empty = content is None and reasoning is None and usage is not None
        return SimpleNamespace(choices=[] if empty else [SimpleNamespace(delta=delta)], usage=usage)

    chunks3 = [ochunk(reasoning="penso (oa) "), ochunk(reasoning="ancora"),
               ochunk(content="Risposta OA"), ochunk(usage=usage)]
    rec3 = _Recorder()
    rec3.create = lambda **kw: iter(chunks3)  # type: ignore
    oa._client = SimpleNamespace(chat=SimpleNamespace(completions=rec3))
    ev3 = []
    resp3 = oa.complete([{"role": "user", "content": "x"}], stream=True,
                        on_delta=lambda p: ev3.append(("delta", p)),
                        on_reasoning=lambda r: ev3.append(("reason", r)))
    check("stream OpenAI: campo `reasoning` PRIMA del contenuto", bool(ev3) and ev3[0] == ("reason", "penso (oa) ancora"), str(ev3))
    check("stream OpenAI: reasoning nel resp", resp3.reasoning == "penso (oa) ancora" and resp3.content == "Risposta OA")


def test_system_write_edit():
    from flair.tools import system as st
    root = Path("/tmp/flair_sys_we")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    ctx = ToolContext(cfg=cfg_for(root))
    target = root / "REPORT.md"

    out = st.write_file(ctx, path=str(target), content="# Titolo\n\nCorpo.\n")
    check("system write_file: crea", target.exists() and "Created" in out)
    check("system write_file: contenuto", target.read_text() == "# Titolo\n\nCorpo.\n")
    out = st.write_file(ctx, path=str(target), content="nuovo")
    check("system write_file: sovrascrive", "Overwrote" in out and target.read_text() == "nuovo")

    nested = root / "a" / "b" / "c.txt"
    st.write_file(ctx, path=str(nested), content="ok")
    check("system write_file: crea cartelle mancanti", nested.exists())

    target.write_text("alpha\nbeta\n")
    out = st.edit_file(ctx, path=str(target), old_string="beta", new_string="gamma")
    check("system edit_file: modifica", target.read_text() == "alpha\ngamma\n" and out.startswith("✓"))
    out = st.edit_file(ctx, path=str(root / "nope.md"), old_string="x", new_string="y")
    check("system edit_file: file inesistente", out.startswith("❌"))


def test_cli_always_per_tool():
    import io as _io

    from rich.console import Console

    from flair.cli import CLI
    cfg = cfg_for(Path("."))
    cfg.auto_approve = False
    cli = CLI(cfg)
    cli.console = Console(file=_io.StringIO())  # silenzia l'output del pannello
    calls = {"n": 0}

    def fake_input(_prompt):
        calls["n"] += 1
        return "a"  # always
    cli.console.input = fake_input  # type: ignore

    ok1 = cli._approve("run_command", {"command": "echo 1"})
    ok2 = cli._approve("run_command", {"command": "echo 2"})  # comando diverso, stesso tool
    check("always per-tool: prima approvata (chiede una volta)", ok1 is True and calls["n"] == 1)
    check("always per-tool: 2ª auto-approvata SENZA richiesta", ok2 is True and calls["n"] == 1)
    cli._approve("write_file", {"path": "x", "content": ""})  # tool diverso → richiede di nuovo
    check("always per-tool: tool diverso richiede", calls["n"] == 2)


def test_approval_prompt_brackets():
    import io as _io

    from rich.console import Console
    c = Console(file=_io.StringIO())
    c.print(r"proceed? \[y]es / \[n]o / \[a]lways")  # parentesi escape-ate
    out = c.file.getvalue()
    check("prompt: mostra [y]es/[n]o/[a]lways letterali", "[y]es" in out and "[n]o" in out and "[a]lways" in out, out)


def test_help_renders():
    import io as _io

    from rich.console import Console

    from flair.cli import CLI
    cli = CLI(cfg_for(Path(".")))
    cli.console = Console(file=_io.StringIO(), width=100)
    cli._print_help()
    out = cli.console.file.getvalue()
    for token in ("/code", "/do", "/provider", "/model", "/think-model", "/compact",
                  "/save", "/load", "/sessions", "/reset", "/root", "exit"):
        check(f"help: contiene {token}", token in out, out[:200])
    check("help: argomenti opzionali [nome] mostrati", "[name]" in out and "<task>" in out, out[:400])


def test_stop_flow():
    root = Path("/tmp/flair_stop")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / "a.py").write_text("x\n")
    cfg = cfg_for(root)
    cfg.auto_approve = False
    # un turno con DUE tool call distruttive; l'utente ferma alla prima
    fake = FakeProvider([
        LLMResponse(content="procedo", tool_calls=[
            tc("write_file", path="a.py", content="uno"),
            tc("write_file", path="b.py", content="due"),
        ], usage=Usage(total_tokens=1)),
        LLMResponse(content="non dovrei arrivare qui", usage=Usage(total_tokens=1)),
    ])
    agent = coding_agent.build(cfg, fake, approve=lambda name, args: "stop")
    res = agent.run("scrivi due file")

    check("stop: stopped_reason = stopped", res.stopped_reason == "stopped", res.stopped_reason)
    check("stop: nessun file scritto", (root / "a.py").read_text() == "x\n" and not (root / "b.py").exists())
    check("stop: il modello non è richiamato dopo lo stop", fake.i == 1, fake.i)
    # ogni tool_call dell'assistente DEVE avere una risposta 'tool' (conversazione valida per l'API)
    asst = [m for m in agent.messages if m["role"] == "assistant" and m.get("tool_calls")][-1]
    tool_ids = {c["id"] for c in asst["tool_calls"]}
    answered = {m["tool_call_id"] for m in agent.messages if m["role"] == "tool"}
    check("stop: ogni tool_call risposta (conversazione valida)", tool_ids <= answered, f"{tool_ids} vs {answered}")
    tmsgs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    check("stop: l'interruzione diventa informazione", any("Stopped by the user" in t for t in tmsgs))


def test_keyboard_interrupt_during_model_call():
    # Ctrl-C DURANTE la chiamata al modello (lo scenario dello stallo di rete): l'agente
    # non deve crashare, deve chiudere il turno come "stopped" lasciando la conversazione
    # valida. Nessun tool_call è stato emesso, quindi niente da "richiudere".
    cfg = cfg_for(Path("."))
    fake = FakeProvider([KeyboardInterrupt()])
    agent = coding_agent.build(cfg, fake)
    res = agent.run("fai un check")
    check("ctrl-c (modello): stopped_reason = stopped", res.stopped_reason == "stopped", res.stopped_reason)
    check("ctrl-c (modello): chiamata tentata", fake.i == 1, fake.i)
    roles = [m["role"] for m in agent.messages]
    check("ctrl-c (modello): nessun tool_call pendente", "tool" not in roles, roles)
    check("ctrl-c (modello): ultimo messaggio è dell'utente", roles[-1] == "user", roles)


def test_keyboard_interrupt_mid_tools():
    # Ctrl-C MENTRE un tool è in approvazione/esecuzione: l'assistant con i tool_call è
    # già in cronologia, quindi ogni tool_call DEVE ricevere una risposta 'tool', altrimenti
    # la prossima chiamata API fallirebbe. La conversazione resta valida.
    root = Path("/tmp/flair_kbi")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / "a.py").write_text("x\n")
    cfg = cfg_for(root)
    cfg.auto_approve = False
    fake = FakeProvider([
        LLMResponse(content="procedo", tool_calls=[
            tc("write_file", path="a.py", content="uno"),
            tc("write_file", path="b.py", content="due"),
        ], usage=Usage(total_tokens=1)),
        LLMResponse(content="non dovrei arrivare qui", usage=Usage(total_tokens=1)),
    ])

    def _ctrl_c(_name, _args):
        raise KeyboardInterrupt
    agent = coding_agent.build(cfg, fake, approve=_ctrl_c)
    res = agent.run("scrivi due file")
    check("ctrl-c (tool): stopped_reason = stopped", res.stopped_reason == "stopped", res.stopped_reason)
    check("ctrl-c (tool): nessun file scritto", (root / "a.py").read_text() == "x\n" and not (root / "b.py").exists())
    check("ctrl-c (tool): modello non richiamato dopo", fake.i == 1, fake.i)
    asst = [m for m in agent.messages if m["role"] == "assistant" and m.get("tool_calls")][-1]
    tool_ids = {c["id"] for c in asst["tool_calls"]}
    answered = {m["tool_call_id"] for m in agent.messages if m["role"] == "tool"}
    check("ctrl-c (tool): ogni tool_call risposta (conversazione valida)", tool_ids <= answered, f"{tool_ids} vs {answered}")
    tmsgs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    check("ctrl-c (tool): interruzione registrata", any("Stopped by the user" in t for t in tmsgs))


def test_repl_survives_turn_error():
    # La REPL non deve MAI crashare per un errore nel turno (es. timeout di rete del modello
    # esaurita la coda di retry/fallback) né per Ctrl-C: _safe_run_task li assorbe.
    import io as _io

    from rich.console import Console

    from flair.cli import CLI
    cli = CLI(cfg_for(Path(".")))
    cli.console = Console(file=_io.StringIO())

    def _boom(*_a, **_k):
        raise RuntimeError("read timeout simulato")
    cli.run_task = _boom  # type: ignore
    propagated = False
    try:
        cli._safe_run_task("ciao")
    except Exception:
        propagated = True
    check("repl: errore nel turno non propaga", not propagated)

    def _kb(*_a, **_k):
        raise KeyboardInterrupt
    cli.run_task = _kb  # type: ignore
    propagated_kb = False
    try:
        cli._safe_run_task("ciao")
    except BaseException:  # noqa: BLE001
        propagated_kb = True
    check("repl: Ctrl-C nel turno non propaga", not propagated_kb)


def test_streaming_fallback_no_duplication():
    # A1: uno stallo di rete a metà stream non deve sdoppiare l'output.
    import httpx

    cfg = cfg_for(Path("."))
    cfg.provider = "deepseek"

    def chunk(content):
        delta = SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)

    # Caso 1: stallo PRIMA di emettere contenuto → fallback non-streaming pulito.
    ds = DeepSeekProvider(cfg)

    def gen_no_content():
        if False:        # rende la funzione un generatore senza codice irraggiungibile
            yield
        raise httpx.ReadTimeout("stallo")

    def create_a(**kw):
        if kw.get("stream"):
            return gen_no_content()
        return _fake_response(content="risposta completa")
    ds._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_a)))
    got_a: list[str] = []
    resp_a = ds.complete([{"role": "user", "content": "x"}], stream=True, on_delta=got_a.append)
    check("A1: stallo pre-contenuto → fallback non-streaming", resp_a.content == "risposta completa", resp_a.content)
    check("A1: nulla emesso a video nel caso pulito", got_a == [], got_a)

    # Caso 2: stallo DOPO aver emesso testo → niente fallback, propaga (no sdoppiamento).
    ds2 = DeepSeekProvider(cfg)

    def gen_with_content():
        yield chunk("parz")
        raise httpx.ReadTimeout("stallo")

    def create_b(**kw):
        if kw.get("stream"):
            return gen_with_content()
        return _fake_response(content="NON dovrei rigenerare")
    ds2._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_b)))
    got_b: list[str] = []
    raised = False
    try:
        ds2.complete([{"role": "user", "content": "x"}], stream=True, on_delta=got_b.append)
    except httpx.ReadTimeout:
        raised = True
    check("A1: contenuto già emesso → propaga errore originale", raised)
    check("A1: nessuna rigenerazione (no output sdoppiato)", "".join(got_b) == "parz", "".join(got_b))


def test_cost_estimate_cache():
    # A2: un prompt interamente in cache non deve essere addebitato due volte.
    cfg = cfg_for(Path("."))
    cfg.price_cache_hit, cfg.price_cache_miss, cfg.price_output = 1.0, 10.0, 20.0  # USD/1M
    ds = DeepSeekProvider(cfg)

    # Tutto in cache (miss riportato = 0): costo solo a prezzo hit.
    full_hit = Usage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000,
                     cache_hit_tokens=1_000_000, cache_miss_tokens=0)
    check("A2: full cache hit non raddoppia", abs(ds.estimate_cost(full_hit, cfg) - 1.0) < 1e-9,
          ds.estimate_cost(full_hit, cfg))

    # Suddivisione normale.
    mixed = Usage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000,
                  cache_hit_tokens=800_000, cache_miss_tokens=200_000)
    check("A2: hit+miss sommati correttamente", abs(ds.estimate_cost(mixed, cfg) - (0.8 + 2.0)) < 1e-9,
          ds.estimate_cost(mixed, cfg))

    # Provider che NON riporta la cache (entrambi 0): ricade su prompt_tokens a prezzo miss.
    nocache = Usage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000,
                    cache_hit_tokens=0, cache_miss_tokens=0)
    check("A2: senza dati cache → tutto a prezzo miss", abs(ds.estimate_cost(nocache, cfg) - 10.0) < 1e-9,
          ds.estimate_cost(nocache, cfg))


def test_run_once_exit_codes():
    # A3: la modalità one-shot non crasha; ritorna un exit code che riflette l'esito.
    import io as _io

    from rich.console import Console

    from flair.cli import CLI
    from flair.core.agent import AgentResult
    cli = CLI(cfg_for(Path(".")))
    cli.console = Console(file=_io.StringIO())

    cli.run_task = lambda *a, **k: AgentResult("ok", stopped_reason="done")   # type: ignore
    check("A3: run_once done → 0", cli.run_once("ciao") == 0)

    cli.run_task = lambda *a, **k: AgentResult("", stopped_reason="max_steps")  # type: ignore
    check("A3: run_once max_steps → 2", cli.run_once("ciao") == 2)

    cli.run_task = lambda *a, **k: AgentResult("", stopped_reason="loop")       # type: ignore
    check("A3: run_once loop → 3", cli.run_once("ciao") == 3)

    cli.run_task = lambda *a, **k: AgentResult("", stopped_reason="stopped")    # type: ignore
    check("A3: run_once stopped → 4", cli.run_once("ciao") == 4)

    cli.run_task = lambda *a, **k: AgentResult("", stopped_reason="budget")     # type: ignore
    check("A3: run_once budget → 5", cli.run_once("ciao") == 5)

    def _boom(*_a, **_k):
        raise RuntimeError("timeout simulato")
    cli.run_task = _boom                          # type: ignore
    check("A3: run_once errore → 1", cli.run_once("ciao") == 1)

    def _kb(*_a, **_k):
        raise KeyboardInterrupt
    cli.run_task = _kb                            # type: ignore
    check("A3: run_once Ctrl-C → 130", cli.run_once("ciao") == 130)


def test_tools_command():
    # D1: /tools elenca i tool dell'agente attivo senza errori.
    import io as _io

    from rich.console import Console

    from flair.cli import CLI
    cli = CLI(cfg_for(Path(".")))
    cli.console = Console(file=_io.StringIO())

    names = [n for n, _ in cli.agents["general"].toolset.catalog()]
    check("D1: catalog espone i nomi dei tool", "search_files" in names and "web_search" in names, names)
    coding_names = [n for n, _ in cli.agents["coding"].toolset.catalog()]
    check("D1: catalog coding ha codice + web, non i tool desktop",
          "grep" in coding_names and "web_search" in coding_names and "search_files" not in coding_names,
          coding_names)

    cli.last_agent = "coding"
    cli._print_tools()  # non deve sollevare
    out = cli.console.file.getvalue()
    check("D1: _print_tools mostra un tool del coding", "grep" in out, out[:200])


def test_cli_approve_stop_and_yes():
    import io as _io

    from rich.console import Console

    from flair.cli import CLI
    cfg = cfg_for(Path("."))
    cfg.auto_approve = False
    cli = CLI(cfg)
    cli.console = Console(file=_io.StringIO())

    cli.console.input = lambda _p: "s"        # type: ignore
    check("approve: 's' = stop", cli._approve("run_command", {"command": "x"}) == "stop")
    cli.console.input = lambda _p: "stop"     # type: ignore
    check("approve: 'stop' = stop", cli._approve("run_command", {"command": "x"}) == "stop")
    cli.console.input = lambda _p: "si"       # type: ignore
    check("approve: 'si' = sì (yes)", cli._approve("run_command", {"command": "x"}) is True)
    cli.console.input = lambda _p: "sì"       # type: ignore
    check("approve: 'sì' = yes", cli._approve("run_command", {"command": "x"}) is True)
    cli.console.input = lambda _p: "n"        # type: ignore
    check("approve: 'n' = no", cli._approve("run_command", {"command": "x"}) is False)

    def _boom(_p):
        raise KeyboardInterrupt
    cli.console.input = _boom                 # type: ignore
    check("approve: Ctrl-C = stop", cli._approve("run_command", {"command": "x"}) == "stop")


def test_shell_multiline_routing():
    from flair.tools import shell

    # riga singola → sempre None (si usa la shell normale)
    check("shell: riga singola non instradata", shell.powershell_script_for("echo ciao", True) is None)
    # multi-riga ma NON Windows → None (sh gestisce il multi-riga)
    check("shell: multi-riga non-Windows usa la shell", shell.powershell_script_for("a\nb", False) is None)
    # multi-riga su Windows, wrapper powershell → estrae il corpo dello script
    cmd = 'powershell -NoProfile -Command "Add-Type @\'\nusing System;\n\'@\nWrite-Host hi"'
    body = shell.powershell_script_for(cmd, True)
    check("shell: wrapper powershell → corpo estratto",
          body is not None and body.startswith("Add-Type") and "powershell" not in body, repr(body))
    # multi-riga su Windows senza wrapper → si esegue l'intero comando come script
    bare = "Add-Type @'\nusing System;\n'@\nWrite-Host hi"
    check("shell: multi-riga senza wrapper → script intero", shell.powershell_script_for(bare, True) == bare)
    # pwsh -c maiuscole/minuscole e .exe
    check("shell: pwsh -c riconosciuto",
          shell.powershell_script_for('pwsh.exe -c "Get-Date\nGet-Host"', True) == "Get-Date\nGet-Host")


def test_shell_decoding_robust():
    from flair.tools import shell
    # output con byte non-UTF8: senza errors="replace" run() alzerebbe UnicodeDecodeError
    cmd = r'''python3 -c 'import sys; sys.stdout.buffer.write(b"\xff\xfe ok")' '''
    proc = shell.run_shell(cmd.strip(), timeout=10)
    check("shell: output non decodificabile non rompe",
          proc.returncode == 0 and isinstance(proc.stdout, str) and "ok" in proc.stdout, repr(proc.stdout))


def test_search_files_coercion():
    from flair.tools import system as st
    root = Path("/tmp/flair_search")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / "song.mp3").write_text("x")
    (root / "note.txt").write_text("y")
    ctx = ToolContext(cfg=cfg_for(root))
    # extensions e locations come STRINGA (errore comune del modello) → trattati come liste
    out = st.search_files(ctx, query="", extensions="mp3", locations=str(root))
    check("search_files: extensions/locations stringa coerced", "song.mp3" in out and "note.txt" not in out, out)
    # le liste regolari continuano a funzionare
    out2 = st.search_files(ctx, query="", extensions=["txt"], locations=[str(root)])
    check("search_files: liste regolari ok", "note.txt" in out2 and "song.mp3" not in out2, out2)


def test_bool_coercion():
    from flair.core.tool import ToolError
    from flair.tools.fs import apply_edit, as_bool

    check("as_bool: 'false' → False", as_bool("false") is False and as_bool("0") is False and as_bool("no") is False)
    check("as_bool: 'true' → True", as_bool("true") is True and as_bool("1") is True and as_bool("sì") is True)
    check("as_bool: bool/None invariati", as_bool(True) is True and as_bool(False) is False and as_bool(None) is False)
    check("as_bool: stringa vuota → False", as_bool("") is False)

    # replace_all="false" (stringa) NON deve diventare truthy: con 2 match → ambiguo → ToolError
    raised = False
    try:
        apply_edit("a a a", "a", "b", "false")
    except ToolError:
        raised = True
    check("bool: replace_all='false' trattato come False (match ambiguo)", raised)
    # replace_all="true" (stringa) → sostituisce tutto
    out, _strat = apply_edit("a a a", "a", "b", "true")
    check("bool: replace_all='true' sostituisce tutto", out == "b b b", out)


def test_powershell_temp_cleanup():
    import os
    import subprocess as _sp

    from flair.tools import shell
    real_run = shell.subprocess.run
    cap: dict = {}

    # 1) Successo: il file esiste DURANTE l'esecuzione e sparisce DOPO.
    def fake_ok(args, **kw):
        path = args[-1]
        cap["path"] = path
        cap["existed_during"] = os.path.exists(path)
        cap["content"] = open(path, encoding="utf-8-sig").read()
        return _sp.CompletedProcess(args, 0, stdout="ciao", stderr="")
    shell.subprocess.run = fake_ok
    try:
        proc = shell.run_powershell_script("Write-Output 'x'", timeout=5)
    finally:
        shell.subprocess.run = real_run
    check("ps cleanup: file presente durante l'esecuzione", cap["existed_during"] is True)
    check("ps cleanup: script scritto nel .ps1", "Write-Output" in cap["content"])
    check("ps cleanup: file CANCELLATO dopo (successo)", not os.path.exists(cap["path"]))
    check("ps cleanup: output catturato", proc.stdout == "ciao")

    # 2) Timeout: il file deve sparire COMUNQUE (uscita per eccezione).
    def fake_timeout(args, **kw):
        cap["path_to"] = args[-1]
        cap["existed_to"] = os.path.exists(args[-1])
        raise _sp.TimeoutExpired(args, kw.get("timeout", 0))
    shell.subprocess.run = fake_timeout
    raised = False
    try:
        shell.run_powershell_script("Start-Sleep 999", timeout=1)
    except _sp.TimeoutExpired:
        raised = True
    finally:
        shell.subprocess.run = real_run
    check("ps cleanup: TimeoutExpired propagato", raised and cap["existed_to"] is True)
    check("ps cleanup: file CANCELLATO anche dopo timeout", not os.path.exists(cap["path_to"]))

    # 3) PowerShell assente: il tool risponde con errore pulito, niente crash. Forziamo
    #    l'assenza (FileNotFoundError) così il test è deterministico su OGNI piattaforma —
    #    anche dove PowerShell/pwsh È installato, es. i runner CI Linux.
    from flair.tools import system as st

    def fake_missing(args, **kw):
        raise FileNotFoundError(2, "No such file or directory", args[0])
    shell.subprocess.run = fake_missing
    try:
        out = st.run_powershell(ToolContext(cfg=cfg_for(Path("."))), script="Write-Output hi", timeout=5)
    finally:
        shell.subprocess.run = real_run
    check("ps cleanup: errore pulito se PowerShell assente", out.startswith("❌"), out)


# ── 17. schemi tool senza drift ───────────────────────────────────────────────

def test_tool_schemas():
    from flair.core.tool import Toolset
    from flair.tools import coding as ct
    from flair.tools import system as st
    from flair.tools import web as wt
    for mod in (ct, st, wt):
        ts = Toolset(mod.TOOLS)
        for s in ts.schemas():
            fn = s["function"]
            check(f"schema {mod.__name__}:{fn['name']}",
                  "name" in fn and "description" in fn and "parameters" in fn and ts.get(fn["name"]) is not None)


# ── Memoria condivisa tra i due agenti ────────────────────────────────────────

def test_shared_memory():
    cfg = cfg_for(Path("."))
    convo = Conversation()
    coding = coding_agent.build(
        cfg, FakeProvider([LLMResponse(content="fatto", usage=Usage(total_tokens=3))]),
        conversation=convo)
    general = general_agent.build(
        cfg, FakeProvider([LLMResponse(content="ok", usage=Usage(total_tokens=3))]),
        conversation=convo)

    # un turno svolto sul coding deve essere visibile al general: stessa memoria
    coding.run("ricorda: il progetto si chiama Zeta")
    check("memoria condivisa: stesso oggetto Conversation", general.convo is coding.convo)
    check("memoria condivisa: general vede il turno di coding",
          any("Zeta" in (m.get("content") or "") for m in general.convo.messages))

    # ogni agente antepone il PROPRIO system prompt, ma la coda è identica
    check("memoria condivisa: system prompt per-agente",
          coding.messages[0]["role"] == "system"
          and coding.messages[0]["content"] != general.messages[0]["content"])
    check("memoria condivisa: coda identica", coding.messages[1:] == general.messages[1:])

    # reset da un agente azzera la conversazione condivisa
    general.reset()
    check("memoria condivisa: reset comune", coding.convo.messages == [])


# ── Router via LLM (decisione primaria) + fallback euristico ──────────────────

def test_router_llm():
    # L'LLM è il decisore: vince anche quando l'euristica direbbe altro.
    p_cod = FakeProvider([LLMResponse(content="coding", usage=Usage(total_tokens=1))])
    check("router LLM: sceglie coding", router.classify("apri il browser", p_cod, last_agent="general") == "coding")
    check("router LLM: una sola chiamata economica",
          len(p_cod.calls) == 1 and p_cod.calls[0]["max_tokens"] == 2 and p_cod.calls[0]["think"] is False)

    p_gen = FakeProvider([LLMResponse(content="general", usage=Usage(total_tokens=1))])
    check("router LLM: sceglie general", router.classify("rifattorizza main.py", p_gen) == "general")

    # Risposta inattesa dell'LLM → ripiega sull'euristica (non sull'azzardo).
    p_junk = FakeProvider([LLMResponse(content="boh", usage=Usage(total_tokens=1))])
    check("router LLM: risposta inattesa → euristica",
          router.classify("rifattorizza la funzione in main.py", p_junk) == "coding")

    # Provider che esplode (offline) → fallback euristico, nessun crash.
    class _Boom:
        def complete(self, *a, **k):
            raise RuntimeError("offline")
    check("router LLM: errore di rete → euristica",
          router.classify("aprimi youtube nel browser", _Boom()) == "general")


# ── Compaction: pairing tool_call/tool sempre valido (anche su due passaggi) ──

def _conversation_valid(msgs) -> bool:
    """Approssima i vincoli dell'API: ogni messaggio 'tool' deve rispondere a una
    tool_call dichiarata dall'assistant immediatamente precedente."""
    open_ids: set = set()
    for m in msgs:
        role = m["role"]
        if role == "assistant":
            open_ids = {tc["id"] for tc in (m.get("tool_calls") or [])}
        elif role == "tool":
            if m.get("tool_call_id") not in open_ids:
                return False
            open_ids.discard(m["tool_call_id"])
        else:  # system / user
            open_ids = set()
    return True


def test_compaction_valid_pairing():
    cfg = cfg_for(Path("."))
    fake = FakeProvider([
        LLMResponse(content="RIASSUNTO-1", usage=Usage(total_tokens=5)),
        LLMResponse(content="RIASSUNTO-2", usage=Usage(total_tokens=5)),
    ])
    agent = coding_agent.build(cfg, fake)
    agent.convo.messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "r1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "r2"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "glob", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "r3"},
        {"role": "assistant", "content": "quasi finito"},
    ]
    # keep=2: lo split naive (indice 6) cadrebbe su un 'tool'; _safe_split lo scavalca.
    ok = agent._compact(aggressive=True)
    check("compaction valida: ha compattato", ok)
    check("compaction valida: riassunto in testa",
          agent.convo.messages[0]["content"].startswith("[Riassunto"))
    check("compaction valida: nessun 'tool' orfano in testa", agent.convo.messages[0]["role"] != "tool")
    check("compaction valida: conversazione API-valida (1)", _conversation_valid(agent.messages))

    # Seconda compattazione di fila: deve restare valida e reinserire un riassunto.
    agent.convo.messages += [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "d", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "d", "content": "r4"},
        {"role": "assistant", "content": "fine"},
    ]
    ok2 = agent._compact(aggressive=True)
    check("compaction valida: seconda compattazione", ok2)
    check("compaction valida: conversazione API-valida (2)", _conversation_valid(agent.messages))
    check("compaction valida: riassunto presente",
          any("[Riassunto" in (m.get("content") or "") for m in agent.convo.messages))


# ── Robustezza dei tool: path mancanti, grep su file, kwarg sconosciuti ───────

def test_tool_robustness():
    import tempfile

    from flair.tools import coding

    root = Path(tempfile.mkdtemp(prefix="flair_toolrob_"))
    (root / "alpha.py").write_text("def foo():\n    return 1\n\nclass Bar:\n    pass\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "beta.txt").write_text("hello foo world\n", encoding="utf-8")
    cfg = cfg_for(root)
    ctx = ToolContext(cfg=cfg)

    # 1) Path inesistente → errore chiaro (non un "nessuna corrispondenza" fuorviante).
    r = coding.grep(ctx, pattern="foo", path="non_esiste")
    check("grep: path inesistente → ❌", r.startswith("❌") and "does not exist" in r, r)
    g = coding.glob(ctx, pattern="*.py", path="non_esiste")
    check("glob: path inesistente → ❌", g.startswith("❌"), g)

    # 2) grep puntato su un FILE → cerca in quel file (prima tornava vuoto in silenzio).
    r2 = coding.grep(ctx, pattern="class Bar", path="alpha.py")
    check("grep: su un singolo file trova le corrispondenze",
          "alpha.py" in r2 and "No files match" not in r2 and not r2.startswith("❌"), r2)

    # Caso normale invariato: ricorsivo su cartella, attraversa le sottocartelle.
    r3 = coding.grep(ctx, pattern="foo")
    check("grep: ricorsivo su cartella (invariato)", "alpha.py" in r3 and "beta.txt" in r3, r3)

    # 3) Argomento sconosciuto: il tool gira lo stesso, con nota; nessuna eccezione.
    r4 = coding.read_file(ctx, path="alpha.py", raw=True)
    check("dispatch: kwarg sconosciuto ignorato con nota",
          r4.startswith("ℹ️ Ignored arguments") and "raw" in r4.splitlines()[0], r4.splitlines()[0])
    r5 = coding.read_file(ctx, path="alpha.py", limit=1)
    check("dispatch: kwarg validi → nessuna nota", not r5.startswith("ℹ️"), r5.splitlines()[0])

    # Un argomento OBBLIGATORIO mancante dà ora un errore AZIONABILE (ToolError) che
    # nomina cosa manca, invece di un TypeError grezzo.
    raised = False
    try:
        coding.grep(ctx, path="alpha.py")  # manca 'pattern'
    except ToolError as exc:
        raised = "pattern" in str(exc)
    check("dispatch: obbligatorio mancante → ToolError che nomina l'argomento", raised)

    # Argomento mancante perché inviato col nome SBAGLIATO (filename invece di path):
    # l'errore nomina la chiave ignorata e suggerisce quella giusta.
    alias_msg = ""
    try:
        coding.read_file(ctx, filename="alpha.py")  # 'filename' al posto di 'path'
    except ToolError as exc:
        alias_msg = str(exc)
    check("dispatch: alias suggerito (filename→path)",
          "path" in alias_msg and "filename" in alias_msg, alias_msg)

    # DRY: write_file condiviso → anche coding ora ha il guard sulle directory.
    rd = coding.write_file(ctx, path="sub", content="x")
    check("DRY: coding write_file su directory → ❌", rd.startswith("❌") and "directory" in rd, rd)

    # DRY: run_command condiviso → output formattato (echo della riga + exit code).
    from flair.tools import shell
    rc = shell.run_command_impl("echo flairtest", 15, cwd=str(root), max_chars=8000)
    check("DRY: run_command_impl esegue e formatta",
          "flairtest" in rc and "exit code 0" in rc and rc.startswith("$ echo flairtest"), rc)

    shutil.rmtree(root, ignore_errors=True)


# ── /root allinea la directory di processo (vale anche per general) ───────────

def test_root_chdir():
    import os
    import tempfile

    from flair.cli import CLI

    old_cwd = os.getcwd()
    target = Path(tempfile.mkdtemp(prefix="flair_root_")).resolve()
    try:
        cfg = cfg_for(Path(".").resolve())
        cfg.session_dir = target / "sessions"   # assoluta e isolata
        cli = CLI(cfg)
        before_convo = cli.convo
        cli.convo.messages.append({"role": "user", "content": "memoria"})

        cli._apply_root(target)
        check("root: la CWD del processo segue la root", Path(os.getcwd()).resolve() == target, os.getcwd())
        check("root: cfg.root aggiornata", cli.cfg.root == target)
        check("root: coding ricostruito ma memoria condivisa preservata",
              cli.agents["coding"].convo is before_convo
              and any(m.get("content") == "memoria" for m in cli.agents["coding"].convo.messages))
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(target, ignore_errors=True)


# ── Affidabilità invocazione tool: coercizione tipi, troncamento, append ──────

def test_arg_coercion():
    import tempfile

    from flair.core.tool import _coerce
    from flair.tools import coding

    # Coercer: stringhe → tipo dichiarato, valori validi intatti, string non toccata.
    check("coerce: int da stringa", _coerce("2", "integer") == 2 and _coerce(2, "integer") == 2)
    check("coerce: int non-numerico resta", _coerce("abc", "integer") == "abc")
    check("coerce: bool da stringa", _coerce("true", "boolean") is True and _coerce("false", "boolean") is False)
    check("coerce: array da singolo", _coerce("mp3", "array") == ["mp3"])
    check("coerce: array da JSON", _coerce('["a","b"]', "array") == ["a", "b"])
    check("coerce: string non toccata", _coerce("x", "string") == "x")

    # Al confine del tool: offset/limit/ignore_case come STRINGHE ora funzionano
    # (prima → TypeError → ❌). File noto in cartella temporanea (CWD-indipendente).
    root = Path(tempfile.mkdtemp(prefix="flair_coerce_")).resolve()
    try:
        cfg = cfg_for(root)
        ctx = ToolContext(cfg=cfg)
        (root / "f.txt").write_text("riga1\nriga2\nriga3\n", encoding="utf-8")
        out = coding.read_file(ctx, path="f.txt", offset="2", limit="1")
        check("coerce: read_file offset/limit stringa",
              not out.startswith("❌") and "riga2" in out and "lines 2-2" in out, out.splitlines()[0])
        (root / "code.py").write_text("def GREP_ME():\n    pass\n", encoding="utf-8")
        g = coding.grep(ctx, pattern="grep_me", path="code.py", ignore_case="true")
        check("coerce: grep ignore_case stringa", not g.startswith("❌") and "Nessuna" not in g, g[:60])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_truncated_args_guidance():
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="flair_raw_")).resolve()
    try:
        agent = general_agent.build(cfg_for(root), FakeProvider([]))
        # parse_tool_args ripiega su {"_raw": ...} quando gli argomenti sono troncati.
        out, ok = agent._run_tool(ToolCall(id="x", name="write_file", arguments={"_raw": '{"path":"a"'}), {})
        check("troncamento: errore azionabile, non eseguito", ok is False and out.startswith("❌"))
        check("troncamento: suggerisce append/parti", "append=true" in out and "truncat" in out.lower(), out)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_finish_reason_truncation_note():
    import io as _io
    import os as _os
    import tempfile

    from rich.console import Console

    from flair.cli import CLI

    root = Path(tempfile.mkdtemp(prefix="flair_fr_")).resolve()
    cwd = _os.getcwd()
    try:
        # Livello agente: il troncamento è un FLAG; il contenuto resta pulito.
        agent = general_agent.build(cfg_for(root), FakeProvider([
            LLMResponse(content="risposta a metà", finish_reason="length", usage=Usage(total_tokens=5)),
        ]))
        res = agent.run("scrivimi qualcosa di lungo")
        check("finish_reason: flag truncated", res.truncated is True)
        check("finish_reason: contenuto NON sporcato", res.content == "risposta a metà", res.content)
        # In cronologia (lato modello) c'è il marcatore di continuazione, così a un
        # "continua" riprende dal punto esatto invece di ricominciare.
        stored = agent.convo.messages[-1]["content"]
        check("finish_reason: marcatore di continuazione in cronologia",
              "RESUME" in stored and "risposta a metà" in stored, stored[-120:])

        # Livello CLI in STREAMING (il caso del bug): la nota va comunque mostrata.
        cli = CLI(cfg_for(root))
        cli.console = Console(file=_io.StringIO())
        cli.cfg.stream = True
        cli.agents["general"] = general_agent.build(cli.cfg, FakeProvider([
            LLMResponse(content="risposta a metà", finish_reason="length", usage=Usage(total_tokens=5)),
        ]))
        saved_stdout = sys.stdout
        sys.stdout = _io.StringIO()   # in streaming i delta vanno su stdout: zittiscili
        try:
            cli.run_task("dimmi tutto", agent_key="general")
        finally:
            sys.stdout = saved_stdout
        out = cli.console.file.getvalue()
        check("finish_reason: la CLI mostra la nota anche in streaming", "truncated" in out, out[-160:])

        # Caso ragionamento-a-vuoto: troncato MA contenuto vuoto (tutto il budget nel
        # reasoning). Niente marcatore "RIPRENDI" (inutile e si accumulerebbe), e la CLI
        # dà la guida giusta (alza FLAIR_MAX_TOKENS / niente --think), non "continua".
        agent2 = general_agent.build(cfg_for(root), FakeProvider([
            LLMResponse(content="", finish_reason="length", usage=Usage(total_tokens=5, reasoning_tokens=5)),
        ]))
        res2 = agent2.run("ragiona tantissimo")
        check("finish_reason: ragionamento-a-vuoto è truncated", res2.truncated is True)
        check("finish_reason: nessun marcatore su contenuto vuoto",
              "RIPRENDI" not in (agent2.convo.messages[-1]["content"] or ""))
        cli2 = CLI(cfg_for(root))
        cli2.console = Console(file=_io.StringIO())
        cli2.cfg.stream = False
        cli2.agents["general"] = general_agent.build(cli2.cfg, FakeProvider([
            LLMResponse(content="", finish_reason="length", usage=Usage(total_tokens=5, reasoning_tokens=5)),
        ]))
        cli2.run_task("ragiona tantissimo", agent_key="general")
        out2 = cli2.console.file.getvalue()
        check("finish_reason: guida diversa per ragionamento-a-vuoto",
              "FLAIR_MAX_TOKENS" in out2 and "No answer" in out2, out2[-200:])
    finally:
        try:
            _os.chdir(cwd)
        except OSError:
            pass
        shutil.rmtree(root, ignore_errors=True)


def test_write_file_append():
    import tempfile

    from flair.tools import coding
    root = Path(tempfile.mkdtemp(prefix="flair_append_")).resolve()
    try:
        cfg = cfg_for(root)
        ctx = ToolContext(cfg=cfg)
        coding.write_file(ctx, path="big.txt", content="parte1\n")
        out = coding.write_file(ctx, path="big.txt", content="parte2\n", append="true")  # stringa → bool
        check("append: messaggio 'aggiunto in coda'", "Appended to" in out, out)
        check("append: contenuto concatenato", (root / "big.txt").read_text() == "parte1\nparte2\n")
        # append su file inesistente = crea
        out2 = coding.write_file(ctx, path="nuovo.txt", content="x\n", append=True)
        check("append: su file nuovo crea", (root / "nuovo.txt").read_text() == "x\n" and "Created" in out2)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_repo_map():
    import tempfile

    from flair.tools import coding

    root = Path(tempfile.mkdtemp(prefix="flair_map_")).resolve()
    try:
        (root / "app.py").write_text(
            "import os\n\nGLOBAL = 1\n\ndef alpha(a, b=2):\n    return a\n\n"
            "class Beta:\n    def m1(self): pass\n    def m2(self, x): pass\n",
            encoding="utf-8")
        sub = root / "web"
        sub.mkdir()
        (sub / "ui.js").write_text(
            "export function render(state) {}\nclass Widget {}\nconst helper = (x) => x;\n",
            encoding="utf-8")
        (root / "README.md").write_text("# titolo\nsolo testo\n", encoding="utf-8")  # niente simboli
        nm = root / "node_modules" / "dep"
        nm.mkdir(parents=True)
        (nm / "lib.js").write_text("function shouldBeSkipped(){}\n", encoding="utf-8")

        cfg = cfg_for(root)
        ctx = ToolContext(cfg=cfg)
        out = coding.repo_map(ctx, path=".")
        check("repo_map: funzione py con args", "def alpha(a, b=…)" in out, out)
        check("repo_map: classe py coi metodi", "class Beta: m1, m2" in out, out)
        check("repo_map: funzione js", "function render(state)" in out, out)
        check("repo_map: classe js + arrow const", "class Widget" in out and "const helper(x)" in out, out)
        check("repo_map: node_modules saltato", "shouldBeSkipped" not in out and "node_modules" not in out)
        check("repo_map: file senza simboli escluso", "README.md" not in out)
        # path inesistente → errore pulito
        check("repo_map: path inesistente → errore", coding.repo_map(ctx, path="non_esiste").startswith("❌"))
        # cap rispettato
        cfg.repomap_max_chars = 50
        check("repo_map: output limitato dal cap", len(coding.repo_map(ctx, path=".")) < 400)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_repo_map_languages():
    from flair.tools import repomap

    # Matrice (estensione → sorgente realistico, simboli attesi). Guarda contro
    # regressioni la copertura multi-linguaggio dell'estrazione (non il filesystem).
    cases = {
        ".ts": ("export class Foo {}\nexport function bar(a, b) {}\nexport const baz = (x) => x;\n"
                "interface Iface {}\ntype Alias = number;\nenum Color { Red }",
                ["class Foo", "function bar(", "const baz(", "interface Iface", "type Alias", "enum Color"]),
        ".go": ("package main\nfunc Hello(name string) string { return name }\n"
                "func (s *Server) Start(p int) {}\ntype Server struct {}",
                ["func Hello(", "func Start(", "type Server"]),
        ".rs": ("pub fn add(a: i32) -> i32 { a }\nasync fn run() {}\nstruct Point { x: i32 }\n"
                "enum Dir { N }\ntrait Draw {}\nimpl Point {\n    fn new() -> Self {}\n}",
                ["fn add(", "fn run(", "struct Point", "enum Dir", "trait Draw", "fn new("]),
        ".java": ("public class Main {\n  public static void main(String[] args) {}\n"
                  "  private int helper(int x) { return x; }\n}\ninterface Svc {}",
                  ["class Main", "main(", "helper(", "interface Svc"]),
        ".cs": ("namespace App {\n  public class Worker {\n    public void Run(int n) {}\n  }\n"
                "  public interface IFoo {}\n}",
                ["namespace App", "class Worker", "Run(", "interface IFoo"]),
        ".c": ("int add(int a, int b) { return a+b; }\nstatic void helper(void) {}\n"
               "int main(void) {\n  if (x) {}\n  for (;;) {}\n}\nstruct Point { int x; };",
               ["add(", "helper(", "main(", "struct Point"]),
        ".cpp": ("class Widget {\npublic:\n  void draw();\n};\nvoid Widget::draw() {}\n"
                 "int compute(int n) { return n; }",
                 ["class Widget", "compute("]),
        ".rb": ("class Dog\n  def bark(loud)\n  end\n  def self.make\n  end\nend\nmodule Pets\nend",
                ["class Dog", "def bark", "def self.make", "module Pets"]),
        ".php": ("<?php\nclass User {\n  public function getName() {}\n}\ninterface Repo {}\n"
                 "function freestanding($a) {}",
                 ["class User", "function getName(", "interface Repo", "function freestanding("]),
        ".swift": ("class VM {\n  func load(id: Int) {}\n}\nstruct Pt {}\nenum E {}\nprotocol P {}\nfunc global() {}",
                   ["class VM", "func load(", "struct Pt", "enum E", "protocol P", "func global("]),
        ".kt": ("class Repo {\n  fun fetch(id: Int): String { return \"\" }\n}\nobject Singleton {}\n"
                "interface Api {}\nfun topLevel(x: Int) {}",
                ["class Repo", "fun fetch(", "object Singleton", "interface Api", "fun topLevel("]),
        ".scala": ("class Service {\n  def run(n: Int): Unit = {}\n}\nobject App {}\ntrait Base {}",
                   ["class Service", "def run", "object App", "trait Base"]),
        ".sh": ("#!/bin/bash\nfunction deploy {\n  echo hi\n}\nbuild() {\n  echo build\n}",
                ["function deploy", "build"]),
        ".lua": ("function M.greet(name)\nend\nlocal function helper(x)\nend",
                 ["function M.greet(", "function helper("]),
        ".dart": ("class Widget {\n  void build() {}\n}\nmixin Logger {}\nenum Status { ok }",
                  ["class Widget", "mixin Logger", "enum Status"]),
        ".ex": ("defmodule MyApp do\n  def hello(name) do\n  end\n  defp secret() do\n  end\nend",
                ["defmodule MyApp", "def hello", "defp secret"]),
        ".pl": ("package Foo;\nsub greet {\n  my $x = shift;\n}", ["package Foo", "sub greet"]),
        ".r": ("add <- function(a, b) {\n  a + b\n}\nmul = function(x) x", ["add(", "mul("]),
        ".jl": ("function area(r)\n  pi*r^2\nend\nstruct Point end\nsquare(x) = x*x",
                ["function area", "struct Point", "square("]),
        ".zig": ("pub fn main() void {}\nconst Point = struct { x: i32 };", ["fn main(", "const Point"]),
        ".nim": ("proc greet(name: string) =\n  echo name\ntype Animal = object", ["proc greet(", "Animal"]),
        ".clj": ("(defn add [a b] (+ a b))\n(def x 1)\n(defmacro unless [t b] )",
                 ["defn add", "def x", "defmacro unless"]),
        ".hs": ("module Main where\nadd :: Int -> Int -> Int\nadd a b = a + b\ndata Tree = Leaf",
                ["add", "data Tree"]),
        ".sql": ("CREATE TABLE users (id INT);\ncreate view v as select 1;", ["TABLE users", "view v"]),
    }
    for ext, (src, expected) in cases.items():
        joined = " | ".join(repomap._symbols_for(Path("t" + ext), src))
        missing = [e for e in expected if e not in joined]
        check(f"repo_map lang {ext}: simboli estratti", not missing, f"mancanti {missing} in: {joined[:120]}")
    # I commenti/usi a metà riga non devono generare falsi positivi di rilievo.
    noise = " | ".join(repomap._symbols_for(Path("t.c"), "    result = compute(x);\n    return foo(y);\n"))
    check("repo_map: niente falsi positivi da chiamate", "compute(" not in noise and "foo(" not in noise, noise)


def test_explore_subagent():
    from flair.agents import explorer
    from flair.core.agent import Conversation
    from flair.tools import subagent

    cfg = cfg_for(Path("."))

    # 1) Il sub-agente esploratore è di SOLA LETTURA e non ricorsivo.
    exp = explorer.build(cfg, FakeProvider([]), conversation=Conversation())
    names = {s["function"]["name"] for s in exp.toolset.schemas()}
    check("explore: sub-agente di sola lettura (niente edit/scrittura/comandi)",
          {"edit_file", "write_file", "multi_edit", "run_command", "run_powershell"}.isdisjoint(names), str(names))
    check("explore: niente ricorsione (explore non nel toolset del sub-agente)", "explore" not in names)
    check("explore: ha i tool di lettura", {"read_file", "grep", "repo_map", "glob", "list_directory"} <= names, str(names))

    # 2) Meccanica: il tool costruisce un sub-agente con conversazione ISOLATA, ne
    #    restituisce la sintesi e RIPORTA i suoi token via ctx.delegated_usage.
    prov = FakeProvider([LLMResponse(content="X è in foo.py:10",
                                     usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120))])
    ctx = ToolContext(cfg=cfg, provider=prov)
    out = subagent.explore(ctx, task="dove è definito X?")
    check("explore: restituisce la sintesi del sub-agente", "X è in foo.py:10" in out, out)
    check("explore: footer del sub-agente", "🔭 esplorato" in out, out)
    check("explore: usage del sub-agente riportato (delegated_usage)",
          ctx.delegated_usage is not None and ctx.delegated_usage.total_tokens == 120, str(ctx.delegated_usage))
    check("explore: il sub-agente è stato eseguito (1 chiamata)", len(prov.calls) == 1, str(prov.calls))

    # 3) Senza provider nel contesto → errore pulito, niente eccezioni.
    no_prov = ToolContext(cfg=cfg, provider=None)
    check("explore: errore pulito senza provider", subagent.explore(no_prov, task="x").startswith("❌"))


def test_explore_usage_accounting():
    """End-to-end: i token del sub-agente confluiscono SIA nel turno SIA nella
    sessione, una sola volta, e le sue letture NON entrano nel contesto del genitore."""
    from flair.core.agent import Conversation

    cfg = cfg_for(Path("."))
    convo = Conversation()

    def U(t, h, m):
        return Usage(prompt_tokens=t, total_tokens=t, cache_hit_tokens=h, cache_miss_tokens=m)

    # Stesso provider per genitore e sub-agente (come nel reale). Ordine delle chiamate:
    # genitore→explore, sub→list_directory, sub→sintesi, genitore→risposta finale.
    prov = FakeProvider([
        LLMResponse(tool_calls=[tc("explore", task="dove è X?")], usage=U(100, 80, 20)),
        LLMResponse(tool_calls=[tc("list_directory", path=".")], usage=U(300, 250, 50)),
        LLMResponse(content="X è in foo.py:10", usage=U(900, 850, 50)),
        LLMResponse(content="Trovato: foo.py:10", usage=U(260, 230, 30)),
    ])
    agent = coding_agent.build(cfg, prov, conversation=convo)
    result = agent.run("trova X")

    check("explore accounting: il turno include i token del sub-agente",
          result.usage.total_tokens == 100 + 300 + 900 + 260, str(result.usage))
    check("explore accounting: sessione = turno (un solo turno)",
          convo.total_usage.total_tokens == result.usage.total_tokens, str(convo.total_usage))
    check("explore accounting: cache-hit sommati correttamente",
          result.usage.cache_hit_tokens == 80 + 250 + 850 + 230, str(result.usage))
    check("explore accounting: cache-miss sommati correttamente",
          result.usage.cache_miss_tokens == 20 + 50 + 50 + 30, str(result.usage))
    check("explore accounting: niente doppio conteggio (delegated_usage azzerato)",
          agent.ctx.delegated_usage.total_tokens == 0, str(agent.ctx.delegated_usage))
    joined = " ".join((m.get("content") or "") for m in convo.messages)
    check("explore accounting: le letture del sub-agente NON sono nel contesto del genitore",
          "list_directory" not in joined, joined[:160])


def test_evals_harness():
    import importlib.util

    tasks_path = Path(__file__).resolve().parent / "evals" / "tasks.py"
    check("evals: tasks.py presente", tasks_path.exists())
    spec = importlib.util.spec_from_file_location("_eval_tasks_probe", tasks_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_eval_tasks_probe"] = mod  # le @dataclass risolvono il modulo da sys.modules
    spec.loader.exec_module(mod)
    check("evals: almeno un task definito", len(mod.TASKS) >= 1)
    names = set()
    for t in mod.TASKS:
        check(f"evals: task '{t.name}' ben formato",
              callable(t.setup) and callable(t.check) and bool(t.prompt) and t.agent in ("coding", "general"))
        names.add(t.name)
    check("evals: nomi dei task univoci", len(names) == len(mod.TASKS))


def test_explore_usage_on_stop():
    """Regressione: se l'utente ferma il flusso DOPO un explore nello stesso batch,
    i token delegati non devono perdersi (fold su ogni uscita, anche 'stopped')."""
    from flair.core.agent import Conversation

    cfg = cfg_for(Path("."))
    cfg.auto_approve = False   # serve il gate di approvazione per simulare lo stop

    def U(t):
        return Usage(prompt_tokens=t, total_tokens=t)

    # Batch: explore (gira, delega 900 token) poi write_file (l'utente risponde "stop").
    prov = FakeProvider([
        LLMResponse(tool_calls=[tc("explore", task="dove è X?"),
                                tc("write_file", path="x.txt", content="x")], usage=U(100)),
        LLMResponse(content="X è in foo.py:10", usage=U(900)),   # risposta del sub-agente
    ])
    convo = Conversation()
    agent = coding_agent.build(cfg, prov, conversation=convo,
                               approve=lambda name, args: "stop")
    result = agent.run("trova X e scrivilo in x.txt")

    check("fold su stop: turno interrotto", result.stopped_reason == "stopped", result.stopped_reason)
    check("fold su stop: il turno include i token delegati",
          result.usage.total_tokens == 100 + 900, str(result.usage))
    check("fold su stop: sessione allineata al turno",
          convo.total_usage.total_tokens == result.usage.total_tokens, str(convo.total_usage))
    check("fold su stop: delegated_usage azzerato (niente doppi conteggi dopo)",
          agent.ctx.delegated_usage.total_tokens == 0, str(agent.ctx.delegated_usage))


def test_router_usage_accounting():
    """La chiamata del router è costo reale: con una Conversation va sommata al totale."""
    from flair.core import router
    from flair.core.agent import Conversation

    convo = Conversation()
    prov = FakeProvider([LLMResponse(content="coding", usage=Usage(prompt_tokens=160, total_tokens=162))])
    key = router.classify("sistemami questo bug nel modulo auth", prov, None, convo=convo)
    check("router accounting: classificazione corretta", key == "coding", key)
    check("router accounting: usage sommato alla sessione",
          convo.total_usage.total_tokens == 162, str(convo.total_usage))
    # Senza convo: comportamento invariato (nessun errore, nessun accounting).
    prov2 = FakeProvider([LLMResponse(content="general", usage=Usage(total_tokens=5))])
    check("router accounting: senza convo funziona come prima",
          router.classify("che ore sono?", prov2, None) == "general")


def test_plan_tool():
    from flair.tools import plan as plan_mod

    cfg = cfg_for(Path("."))
    ctx = ToolContext(cfg=cfg)

    out = plan_mod.plan(ctx, steps=[
        {"title": "leggere il modulo", "status": "fatto"},
        {"title": "scrivere il fix", "status": "in_progress"},   # sinonimo inglese
        "eseguire i test",                                        # stringa semplice
    ])
    check("plan: intestazione con conteggio", out.startswith("📋 Plan (1/3 done)"), out)
    check("plan: simboli di stato", "✔ leggere il modulo" in out and "▸ scrivere il fix (in progress)" in out
          and "○ eseguire i test" in out, out)

    # Tolleranza: l'intera lista arriva come JSON string → la coercizione la ripara.
    via_call = plan_mod.plan(ctx, **{"steps": '[{"title": "a"}, {"title": "b", "status": "done"}]'})
    check("plan: steps come JSON string (coercizione array)", via_call.startswith("📋 Plan (1/2 done)"), via_call)

    check("plan: vuoto → errore pulito", plan_mod.plan(ctx, steps=[]).startswith("❌"))
    check("plan: voci senza titolo → errore pulito", plan_mod.plan(ctx, steps=[{"status": "fatto"}]).startswith("❌"))
    over = plan_mod.plan(ctx, steps=[f"passo {i}" for i in range(40)])
    check("plan: tetto sui passi", "oltre il limite" in over and over.count("○") == 30, over)

    # Cablaggio: plan nel coding agent, NON in explorer/general.
    from flair.agents import coding as ca
    from flair.agents import explorer as ea
    from flair.agents import general as ga
    names = lambda a: {s["function"]["name"] for s in a.toolset.schemas()}  # noqa: E731
    check("plan: nel coding agent", "plan" in names(ca.build(cfg, FakeProvider([]))))
    check("plan: NON nell'explorer", "plan" not in names(ea.build(cfg, FakeProvider([]))))
    check("plan: NON nel general", "plan" not in names(ga.build(cfg, FakeProvider([]))))


def _tool_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _asst_msg(*tool_calls: ToolCall) -> dict:
    import json as _json
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": t.id, "type": "function",
         "function": {"name": t.name, "arguments": _json.dumps(t.arguments, ensure_ascii=False)}}
        for t in tool_calls
    ]}


def test_prune_superseded_rules():
    from flair.core import prune

    big = "x" * 1000

    # Regola 1: duplicati — pota la più vecchia, conserva l'ultima.
    a, b = tc("grep", pattern="foo", path="."), tc("grep", pattern="foo", path=".")
    msgs = [_asst_msg(a), _tool_msg(a.id, big), _asst_msg(b), _tool_msg(b.id, big + "fresh")]
    n = prune.prune_superseded(msgs)
    check("prune dup: una potata", n == 1, n)
    check("prune dup: la vecchia è stub", msgs[1]["content"] == prune.STUB)
    check("prune dup: la recente intatta", msgs[3]["content"].endswith("fresh"))
    check("prune dup: pairing intatto", msgs[1]["tool_call_id"] == a.id and len(msgs) == 4)

    # Regola 2: read superata da write_file (overwrite sì, append no).
    r1, w1 = tc("read_file", path="src/a.py"), tc("write_file", path="./src\\a.py", content="nuovo")
    r2, w2 = tc("read_file", path="b.py"), tc("write_file", path="b.py", content="+", append=True)
    msgs = [_asst_msg(r1), _tool_msg(r1.id, big), _asst_msg(w1), _tool_msg(w1.id, "✓ Sovrascritto"),
            _asst_msg(r2), _tool_msg(r2.id, big), _asst_msg(w2), _tool_msg(w2.id, "✓ Aggiunto")]
    n = prune.prune_superseded(msgs)
    check("prune write: solo la read sovrascritta (path normalizzato)", n == 1 and msgs[1]["content"] == prune.STUB, n)
    check("prune write: append NON invalida", msgs[5]["content"] == big)

    # Regola 3: read parziale coperta da read INTERA successiva (stesso path).
    p1 = tc("read_file", path="m.py", offset=10, limit=40)
    p2 = tc("read_file", path="m.py")
    other = tc("read_file", path="altro.py", offset=1, limit=5)
    msgs = [_asst_msg(p1), _tool_msg(p1.id, big), _asst_msg(other), _tool_msg(other.id, big),
            _asst_msg(p2), _tool_msg(p2.id, big)]
    n = prune.prune_superseded(msgs)
    check("prune full-read: parziale coperta potata, altro path intatto",
          n == 1 and msgs[1]["content"] == prune.STUB and msgs[3]["content"] == big, n)

    # Garanzie: output piccoli e tool di scrittura mai potati; idempotente.
    s1, s2 = tc("read_file", path="s.py"), tc("read_file", path="s.py")
    msgs = [_asst_msg(s1), _tool_msg(s1.id, "corto"), _asst_msg(s2), _tool_msg(s2.id, "corto2")]
    check("prune: output piccoli ignorati", prune.prune_superseded(msgs) == 0)
    e1, e2 = tc("edit_file", path="c.py", old_string="a", new_string="b"), tc("read_file", path="c.py")
    msgs = [_asst_msg(e2), _tool_msg(e2.id, big), _asst_msg(e1), _tool_msg(e1.id, "✓ Modificato")]
    check("prune: edit_file NON invalida le letture", prune.prune_superseded(msgs) == 0)
    a2, b2 = tc("grep", pattern="q", path="."), tc("grep", pattern="q", path=".")
    msgs = [_asst_msg(a2), _tool_msg(a2.id, big), _asst_msg(b2), _tool_msg(b2.id, big)]
    check("prune: idempotente", prune.prune_superseded(msgs) == 1 and prune.prune_superseded(msgs) == 0)


def test_prune_in_agent():
    """Stadio 0 nell'Agent: sopra soglia pota e, se basta, SALTA il riassunto LLM;
    callback on_prune; /compact manuale conta la potatura; kill-switch rispettato."""
    from flair.core import prune
    from flair.core.agent import Conversation

    big = "x" * 4000

    def history():
        a, b = tc("read_file", path="f.py"), tc("read_file", path="f.py")
        return [{"role": "user", "content": "task"},
                _asst_msg(a), _tool_msg(a.id, big),
                _asst_msg(b), _tool_msg(b.id, big),
                {"role": "assistant", "content": "ok"}]

    # 1) La potatura basta → NESSUNA chiamata di riassunto (provider mai invocato).
    cfg = cfg_for(Path("."))
    cfg.context_window = 4000          # soglia = 3000 token
    pruned_counts: list[int] = []
    prov = FakeProvider([])
    convo = Conversation()
    convo.messages = history()
    convo.last_prompt_tokens = 3500    # sopra soglia
    agent = coding_agent.build(cfg, prov, conversation=convo, on_prune=pruned_counts.append)
    agent._maybe_compact()
    check("prune agent: ha potato (callback)", pruned_counts == [1], str(pruned_counts))
    check("prune agent: stub in contesto", convo.messages[2]["content"] == prune.STUB)
    check("prune agent: riassunto LLM SALTATO", len(prov.calls) == 0, str(prov.calls))
    check("prune agent: contatori azzerati (prefisso spezzato)",
          convo.last_prompt_tokens == 0 and convo.sent_upto == 0)
    check("prune agent: sotto soglia dopo potatura", agent._ctx_estimate() <= cfg.compact_threshold,
          str(agent._ctx_estimate()))

    # 2) /compact manuale: la sola potatura conta come "qualcosa fatto" (la storia è
    #    troppo corta per il riassunto: keep_recent default la copre tutta).
    cfg2 = cfg_for(Path("."))
    prov2 = FakeProvider([])
    convo2 = Conversation()
    convo2.messages = history()
    agent2 = coding_agent.build(cfg2, prov2, conversation=convo2)
    check("prune agent: compact() manuale → True con la sola potatura", agent2.compact() is True)
    check("prune agent: compact() manuale non ha riassunto (storia corta)", len(prov2.calls) == 0)

    # 3) Kill-switch: FLAIR_COMPACT_PRUNE=false → nessuna potatura, compaction classica.
    cfg3 = cfg_for(Path("."))
    cfg3.compact_prune = False
    cfg3.context_window = 4000
    cfg3.compact_keep_recent = 2       # storia corta: serve un keep basso perché ci sia testa da riassumere
    prov3 = FakeProvider([LLMResponse(content="riassunto", usage=Usage(total_tokens=5))])
    convo3 = Conversation()
    convo3.messages = history()
    convo3.last_prompt_tokens = 3500
    agent3 = coding_agent.build(cfg3, prov3, conversation=convo3)
    agent3._maybe_compact()
    check("prune agent: kill-switch → niente stub", all(m.get("content") != prune.STUB for m in convo3.messages))
    check("prune agent: kill-switch → riassunto LLM eseguito", len(prov3.calls) == 1, str(prov3.calls))


def test_router_continuation():
    """Le continuazioni nude restano sull'current agent SENZA chiamata LLM (è il
    misroute visto dal vivo: 'Procedi.' instradato a general a metà task coding)."""
    from flair.core import router
    from flair.core.agent import Conversation

    # 1) Continuazioni nude → sticky deterministico, provider MAI chiamato.
    for text, last in [("Procedi.", "coding"), ("vai", "coding"), ("Ok, continua pure!", "coding"),
                       ("sì grazie, procedi", "coding"), ("go ahead", "coding"), ("do it", "coding"),
                       ("va bene così", "general"), ("riprova", "general"), ("Avanti.", "general")]:
        prov = FakeProvider([])
        convo = Conversation()
        got = router.classify(text, prov, last, convo=convo)
        check(f"router continuazione: {text!r} resta su {last}", got == last, got)
        check(f"router continuazione: {text!r} senza chiamata LLM", prov.calls == [], str(prov.calls))
        check(f"router continuazione: {text!r} nessun usage aggiunto", convo.total_usage.total_tokens == 0)

    # 2) Controesempi: contenuto reale → routing normale (LLM consultato).
    for text, last in [("vai su google e cerca le notizie", "coding"),
                       ("procedi con il refactor del modulo auth", "general"),
                       ("ok ma prima dimmi che ore sono", "coding")]:
        prov = FakeProvider([LLMResponse(content="general", usage=Usage(total_tokens=3))])
        got = router.classify(text, prov, last)
        check(f"router continuazione: {text!r} NON corto-circuitato", len(prov.calls) == 1, str(prov.calls))

    # 3) Senza last_agent (primo messaggio) niente sticky: si instrada normalmente.
    prov = FakeProvider([LLMResponse(content="coding", usage=Usage(total_tokens=3))])
    got = router.classify("procedi", prov, None)
    check("router continuazione: primo messaggio → LLM consultato", len(prov.calls) == 1 and got == "coding")

    # 4) Messaggio lungo di soli filler → oltre il limite, niente corto-circuito.
    long_filler = "ok " * 20
    prov = FakeProvider([LLMResponse(content="general", usage=Usage(total_tokens=3))])
    router.classify(long_filler, prov, "coding")
    check("router continuazione: oltre 40 char → routing normale", len(prov.calls) == 1)


def test_budget_abort():
    """Il tetto di costo HARD ferma il loop prima della chiamata successiva."""
    from flair.core.agent import Conversation

    cfg = cfg_for(Path("."))
    cfg.context_window = 1_000_000   # evita interferenze della compaction
    # Prezzi espliciti per un test deterministico: 1.0 USD per 1M token di input
    # (cache-miss) e output. La prima risposta (110k token) costa ~0.11 USD.
    cfg.price_cache_hit = 1.0
    cfg.price_cache_miss = 1.0
    cfg.price_output = 1.0
    cfg.max_cost = 0.05              # USD: 0.11 > 0.05 → al giro dopo si ferma.
    def U(p, c):
        return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)
    prov = FakeProvider([
        LLMResponse(tool_calls=[tc("read_file", path="a.py")], usage=U(100_000, 10_000)),
        LLMResponse(content="non dovrei arrivare qui", usage=U(100, 10)),
    ])
    convo = Conversation()
    agent = coding_agent.build(cfg, prov, conversation=convo)
    result = agent.run("compito lungo")
    check("budget: stop con reason 'budget'", result.stopped_reason == "budget", result.stopped_reason)
    check("budget: una sola chiamata al modello", len(prov.calls) == 1, str(prov.calls))
    check("budget: nessuna seconda risposta", "non dovrei" not in (result.content or ""))
    # Senza tetto (default) il loop NON si ferma per budget.
    cfg2 = cfg_for(Path("."))
    cfg2.context_window = 1_000_000
    prov2 = FakeProvider([
        LLMResponse(tool_calls=[tc("read_file", path="a.py")], usage=U(100_000, 10_000)),
        LLMResponse(content="ok finito", usage=U(50, 5)),
    ])
    agent2 = coding_agent.build(cfg2, prov2, conversation=Conversation())
    res2 = agent2.run("compito lungo")
    check("budget off: arriva a done", res2.stopped_reason == "done", res2.stopped_reason)


def test_read_only_mode():
    """read_only filtra i tool distruttivi dai due agenti; explorer è già read-only."""
    cfg = cfg_for(Path("."))
    cfg.read_only = True
    names = lambda a: {s["function"]["name"] for s in a.toolset.schemas()}  # noqa: E731

    cnames = names(coding_agent.build(cfg, FakeProvider([])))
    for dest in ("write_file", "edit_file", "multi_edit", "run_command"):
        check(f"read-only coding: {dest} assente", dest not in cnames, dest)
    for ro in ("read_file", "grep", "repo_map", "explore", "plan"):
        check(f"read-only coding: {ro} presente", ro in cnames, ro)

    gnames = names(general_agent.build(cfg, FakeProvider([])))
    for dest in ("write_file", "edit_file", "run_command", "run_powershell"):
        check(f"read-only general: {dest} assente", dest not in gnames, dest)
    for ro in ("open_url", "search_files", "read_file"):
        check(f"read-only general: {ro} presente", ro in gnames, ro)

    # Default (read_only off): i tool distruttivi restano.
    cfg2 = cfg_for(Path("."))
    cnames2 = names(coding_agent.build(cfg2, FakeProvider([])))
    check("read-only off: write_file presente", "write_file" in cnames2)
    check("read-only off: run_command presente", "run_command" in cnames2)


def test_automation_helpers():
    """Exit code per esito e oggetto JSON one-shot (modalità --json)."""
    from flair.cli import build_result_json, exit_code_for
    from flair.core.agent import AgentResult

    check("exit done=0", exit_code_for("done") == 0)
    check("exit max_steps=2", exit_code_for("max_steps") == 2)
    check("exit loop=3", exit_code_for("loop") == 3)
    check("exit stopped=4", exit_code_for("stopped") == 4)
    check("exit budget=5", exit_code_for("budget") == 5)
    check("exit sconosciuto=1", exit_code_for("boh") == 1)

    res = AgentResult(content="ciao", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                      steps=2, stopped_reason="done")
    events = [{"name": "read_file", "ok": True, "args": {"path": "a.py"}},
              {"name": "write_file", "ok": True, "args": {"path": "b.py"}},
              {"name": "write_file", "ok": False, "args": {"path": "c.py"}}]  # fallito → escluso dai file
    obj = build_result_json("coding", "task", res, events, cost_usd=0.0012345678)
    check("json: ok true", obj["ok"] is True)
    check("json: agente", obj["agent"] == "coding")
    check("json: risposta", obj["response"] == "ciao")
    check("json: passi", obj["steps"] == 2)
    check("json: costo arrotondato", obj["cost_usd"] == round(0.0012345678, 6))
    check("json: solo file scritti con successo", obj["files_changed"] == ["b.py"], str(obj["files_changed"]))
    check("json: elenco tool con esito",
          obj["tools"] == [{"name": "read_file", "ok": True}, {"name": "write_file", "ok": True},
                           {"name": "write_file", "ok": False}], str(obj["tools"]))
    check("json: usage totale", obj["usage"]["total_tokens"] == 15)
    json_module.dumps(obj)  # deve essere serializzabile
    check("json: serializzabile", True)
    # stopped_reason ≠ done → ok False
    res2 = AgentResult(content="", stopped_reason="budget")
    check("json: budget → ok false", build_result_json("coding", "t", res2, [], 0.0)["ok"] is False)


def test_parallel_tools():
    import time

    from flair.core.agent import Agent
    from flair.core.tool import Tool, Toolset
    NOOBJ = {"type": "object", "properties": {}, "required": []}

    def agent_with(tools, script, **kw):
        cfg = cfg_for(Path("."))
        cfg.context_window = 10_000_000      # niente compaction
        cfg.stream = False
        return Agent("coding", cfg, FakeProvider(script), Toolset(tools), "sys",
                     conversation=Conversation(), **kw)

    def slow(nm, delay):
        def f(ctx):
            time.sleep(delay)
            return f"out-{nm}"
        return Tool(nm, nm, NOOBJ, f, destructive=False)

    def tool_calls_resp(calls):
        return LLMResponse(tool_calls=calls, finish_reason="tool_calls")
    done_resp = LLMResponse(content="done", finish_reason="stop")   # usage 0: non sporca i conteggi

    # 1) Happy path: 3 tool read-only; append nell'ORDINE delle tool_call anche se il
    #    completamento è fuori ordine (il primo dorme di più).
    calls = [tc("a"), tc("b"), tc("c")]
    ag = agent_with([slow("a", 0.03), slow("b", 0.01), slow("c", 0.0)], [tool_calls_resp(calls)])
    ag.run("vai")
    tmsgs = [m for m in ag.convo.messages if m.get("role") == "tool"]
    check("parallel: tutti i risultati presenti", len(tmsgs) == 3)
    check("parallel: append nell'ordine delle tool_call (non di completamento)",
          [m["tool_call_id"] for m in tmsgs] == [c.id for c in calls])
    check("parallel: contenuti corretti", [m["content"] for m in tmsgs] == ["out-a", "out-b", "out-c"])

    # 2) Usage delegato (come explore): N tool deleganti in parallelo → somma ESATTA,
    #    nessuna perdita (ogni worker usa un ctx isolato). Lo sleep allarga la finestra
    #    in cui un canale condiviso perderebbe aggiornamenti.
    def deleg(nm):
        def f(ctx):
            time.sleep(0.005)
            ctx.delegated_usage = (ctx.delegated_usage or Usage()) + Usage(total_tokens=1, completion_tokens=1)
            return f"ok-{nm}"
        return Tool(nm, nm, NOOBJ, f, destructive=False)
    n = 8
    calls = [tc(f"d{i}") for i in range(n)]
    ag = agent_with([deleg(f"d{i}") for i in range(n)], [tool_calls_resp(calls), done_resp])
    ag.run("vai")
    check("parallel: usage delegato sommato senza perdite",
          ag.convo.total_usage.total_tokens == n, f"atteso {n}, ottenuto {ag.convo.total_usage.total_tokens}")

    # 3) Batch con un tool DISTRUTTIVO → resta sequenziale: il gate di approvazione scatta
    #    (in parallelo non potrebbe), l'ordine è preservato e lo 'stop' funziona.
    ran = []
    def rec(nm, destr):
        def f(ctx):
            ran.append(nm)
            return f"out-{nm}"
        return Tool(nm, nm, NOOBJ, f, destructive=destr)
    cfg = cfg_for(Path("."))
    cfg.context_window = 10_000_000
    cfg.stream = False
    cfg.auto_approve = False
    approvals = []
    calls = [tc("r"), tc("w")]
    ag = Agent("coding", cfg, FakeProvider([tool_calls_resp(calls)]),
               Toolset([rec("r", False), rec("w", True)]), "sys", conversation=Conversation(),
               approve=lambda name, args: (approvals.append(name), "stop")[1])
    res = ag.run("vai")
    check("parallel: batch con distruttivo resta sequenziale (approval invocato)", approvals == ["w"], str(approvals))
    check("parallel: stop sul distruttivo → stopped", res.stopped_reason == "stopped", res.stopped_reason)
    check("parallel: read-only prima del distruttivo eseguito (ordine sequenziale)", ran == ["r"], str(ran))

    # 4) Associazione corretta risultato↔tool (il bug del tentativo precedente): handler
    #    stile CLI (append in on_tool, aggiorna l'ultimo in on_result).
    turn_tools = []
    calls = [tc("x"), tc("y"), tc("z")]
    ag = agent_with([slow("x", 0.02), slow("y", 0.005), slow("z", 0.0)], [tool_calls_resp(calls)],
                    on_tool=lambda name, args: turn_tools.append({"name": name}),
                    on_result=lambda name, output, ok: turn_tools[-1].update(result_of=name) if turn_tools else None)
    ag.run("vai")
    mism = [t for t in turn_tools if t.get("name") != t.get("result_of")]
    check("parallel: callback appaiati → nessun risultato associato al tool sbagliato", not mism, str(turn_tools))

    # 5) parallel_tools=False → percorso sequenziale, comunque corretto e in ordine.
    cfg = cfg_for(Path("."))
    cfg.context_window = 10_000_000
    cfg.stream = False
    cfg.parallel_tools = False
    calls = [tc("a"), tc("b")]
    ag = Agent("coding", cfg, FakeProvider([tool_calls_resp(calls)]),
               Toolset([slow("a", 0.0), slow("b", 0.0)]), "sys", conversation=Conversation())
    ag.run("vai")
    tmsgs = [m for m in ag.convo.messages if m.get("role") == "tool"]
    check("parallel: flag off → sequenziale, risultati in ordine",
          [m["tool_call_id"] for m in tmsgs] == [c.id for c in calls])


def test_atomic_writes():
    import os as _os
    import tempfile as _tf

    from flair.tools import fs
    d = Path(_tf.mkdtemp(prefix="flair_atomic_"))
    f = d / "code.py"

    # Creazione (file nuovo): scrittura diretta, contenuto corretto.
    fs.write_file_impl(None, str(f), "v1\n")
    check("atomic: creazione contenuto", f.read_text() == "v1\n")

    # Sovrascrittura di file esistente → atomica; contenuto aggiornato, nessun .tmp residuo.
    fs.write_file_impl(None, str(f), "v2 molto piu lungo\n")
    check("atomic: sovrascrittura contenuto", f.read_text() == "v2 molto piu lungo\n")
    check("atomic: nessun .tmp dopo sovrascrittura", not list(d.glob(".*.tmp")))

    # Edit di file esistente → atomico; nessun .tmp residuo.
    fs.edit_file_impl(None, str(f), "v2 molto piu lungo", "v3")
    check("atomic: edit contenuto", f.read_text() == "v3\n")
    check("atomic: nessun .tmp dopo edit", not list(d.glob(".*.tmp")))

    # Su POSIX i permessi del file esistente vanno preservati dopo un edit atomico.
    if _os.name != "nt":
        _os.chmod(f, 0o640)
        fs.edit_file_impl(None, str(f), "v3", "v4")
        check("atomic: permessi preservati dopo edit (POSIX)", (f.stat().st_mode & 0o777) == 0o640,
              oct(f.stat().st_mode & 0o777))


def test_session_memory():
    import io as _io
    import tempfile as _tf

    from rich.console import Console

    from flair.memory import SessionMemory

    # ── modulo: add, dedup, filtri, tetto ────────────────────────────────────
    m = SessionMemory(max_chars=4000)
    ok, _ = m.add("I test si lanciano con `python tests/test_smoke.py`")
    check("memoria: add ok", ok and len(m.notes) == 1)
    ok, msg = m.add("  i test SI lanciano   con `python tests/test_smoke.py` ")
    check("memoria: dedup (case/spazi)", not ok and "already in memory" in msg, msg)
    ok, msg = m.add("la chiave è sk-abcdef1234567890abcd")
    check("memoria: filtro segreti (sk-)", not ok and "secrets" in msg, msg)
    ok, msg = m.add("api_key=xyz123segreto per il deploy")
    check("memoria: filtro segreti (api_key=)", not ok, msg)
    ok, msg = m.add("x" * 300)
    check("memoria: note too long rifiutata", not ok and "too long" in msg, msg)
    piccolo = SessionMemory(max_chars=200)
    piccolo.add("nota uno abbastanza lunga da occupare spazio nel tetto totale qui")
    piccolo.add("nota due abbastanza lunga da occupare spazio nel tetto totale qui")
    ok, msg = piccolo.add("nota tre che non deve entrare perché il tetto è stato raggiunto")
    check("memoria: tetto totale → rifiuto azionabile", not ok and "full" in msg, msg)

    # ── blocco per il prompt ─────────────────────────────────────────────────
    check("memoria: vuota → blocco vuoto (zero token)", SessionMemory().block() == "")
    blk = m.block()
    check("memoria: blocco con header e note",
          "## Session memory" in blk and "test_smoke.py" in blk, blk[:80])

    # ── serializzazione: roundtrip, dedup difensivo, manomissioni ────────────
    m.add("In questo repo i file devono essere LF")
    testo = m.to_text()
    m2 = SessionMemory()
    n, trunc = m2.load_text(testo)
    check("memoria: roundtrip sidecar", n == 2 and not trunc and m2.notes == m.notes)
    m.notes.append(m.notes[0])                      # doppione teorico da batch parallelo
    check("memoria: to_text() ripulisce i doppioni", m.to_text().count("test_smoke.py") == 1)
    manomesso = ("# commento\n- nota valida\n- " + "y" * 500 + "\nriga ignorata\n"
                 "- password=hunter2 da saltare\n- nota valida\n")
    m3 = SessionMemory(max_chars=4000, max_note_chars=100)
    n, trunc = m3.load_text(manomesso)
    check("memoria: load tollerante (tronca lunghe, salta segreti, dedup)",
          n == 2 and trunc and len(m3.notes[1]) == 100, str((n, trunc)))
    stretto = SessionMemory(max_chars=210)
    n, trunc = stretto.load_text("- " + "a" * 90 + "\n- " + "b" * 90 + "\n- " + "c" * 90 + "\n")
    check("memoria: load oltre il tetto → troncato con flag", n == 2 and trunc, str((n, trunc)))

    # ── SessionStore: sidecar atomico, rimozione a vuoto ─────────────────────
    from flair.session_store import SessionStore
    d = Path(_tf.mkdtemp(prefix="flair_mem_"))
    st = SessionStore(d)
    st.save_memory("lavoro", m2.to_text())
    check("store: sidecar scritto", (d / "lavoro.memory.md").exists())
    check("store: load_memory roundtrip", "LF" in st.load_memory("lavoro"))
    check("store: nessun .tmp residuo", not list(d.glob("*.tmp")) and not list(d.glob(".*.tmp")))
    st.save_memory("lavoro", "")
    check("store: memoria vuota → sidecar rimosso", not (d / "lavoro.memory.md").exists())
    check("store: load di sidecar assente → ''", st.load_memory("mai-esistita") == "")

    # ── agente: tool remember, anche in batch parallelo (ctx isolati) ────────
    from flair.agents import coding as ca
    cfg = cfg_for(Path("."))
    cfg.context_window = 10_000_000
    cfg.stream = False
    ag = ca.build(cfg, FakeProvider([
        LLMResponse(tool_calls=[tc("remember", note="Fatto A sul progetto"),
                                tc("remember", note="Fatto B sul progetto"),
                                tc("remember", note="fatto a SUL progetto")],   # dup di A
                    finish_reason="tool_calls"),
        LLMResponse(content="fine", finish_reason="stop"),
    ]), conversation=Conversation())
    mem = SessionMemory()
    ag.ctx.memory = mem
    ag.run("ricorda queste cose")
    check("memoria: remember via agente (batch parallelo, ctx isolati)",
          sorted(mem.notes) == ["Fatto A sul progetto", "Fatto B sul progetto"], str(mem.notes))

    cfg_ro = cfg_for(Path("."))
    cfg_ro.read_only = True
    names_ro = {sc["function"]["name"] for sc in ca.build(cfg_ro, FakeProvider([]), conversation=Conversation()).toolset.schemas()}
    check("memoria: remember disponibile in read-only", "remember" in names_ro, str(sorted(names_ro)))
    cfg_off = cfg_for(Path("."))
    cfg_off.memory_enabled = False
    names_off = {sc["function"]["name"] for sc in ca.build(cfg_off, FakeProvider([]), conversation=Conversation()).toolset.schemas()}
    check("memoria: flag off → tool assente (zero schema)", "remember" not in names_off)
    from flair.agents import general as ga
    names_gen = {sc["function"]["name"] for sc in ga.build(cfg, FakeProvider([]), conversation=Conversation()).toolset.schemas()}
    check("memoria: remember anche nell'agente generico", "remember" in names_gen)
    from flair.tools.memory import remember as rem_tool
    out = rem_tool.func(ToolContext(cfg=cfg), note="qualcosa")
    check("memoria: ctx senza memoria → errore pulito", out.startswith("❌"), out)

    # ── CLI end-to-end: iniezione, save/load, /reset conserva, clear ─────────
    from flair.cli import CLI
    cfg2 = cfg_for(Path("."))
    cfg2.session_dir = d
    cli = CLI(cfg2)
    cli.console = Console(file=_io.StringIO())
    base_len = len(cli.agents["coding"].system_prompt)
    cli.memory.add("Il comando di build è `make all`")
    check("cli: remember NON riscrive il prompt in corso (cache preservata)",
          len(cli.agents["coding"].system_prompt) == base_len)
    cli.session_name = "memtest"
    cli._save_session()
    check("cli: /save scrive il sidecar", (d / "memtest.memory.md").exists())
    cli2 = CLI(cfg2)
    cli2.console = Console(file=_io.StringIO())
    check("cli: sessione nuova → memoria vuota, prompt base", cli2.memory.notes == []
          and "## Session memory" not in cli2.agents["coding"].system_prompt)
    check("cli: /load ripristina la memoria", cli2._load_session("memtest") and
          cli2.memory.notes == ["Il comando di build è `make all`"])
    check("cli: dopo /load il prompt contiene il blocco (entrambi gli agenti)",
          all("make all" in ag.system_prompt for ag in cli2.agents.values()))
    cli2.convo.reset()   # è ciò che fa /reset: la memoria resta
    check("cli: /reset conserva la memoria", cli2.memory.notes != [] and
          "make all" in cli2.agents["coding"].system_prompt)
    cli2.memory.clear()
    cli2._refresh_memory_prompts()
    cli2._save_session()
    check("cli: clear → blocco rimosso dal prompt e sidecar cancellato",
          "make all" not in cli2.agents["coding"].system_prompt
          and not (d / "memtest.memory.md").exists())


def test_grep_context_and_move():
    import tempfile as _tf

    from flair.tools import coding
    root = Path(_tf.mkdtemp(prefix="flair_gcm_")).resolve()
    (root / "a.py").write_text("uno\ndue\nBERSAGLIO qui\nquattro\ncinque\nBERSAGLIO due\nsette\n")
    (root / "sub").mkdir()
    (root / "sub" / "b.py").write_text("niente\nBERSAGLIO b\ncoda\n")
    cfg = cfg_for(root)
    ctx = ToolContext(cfg=cfg)

    # context=1: match marcati con ':', contesto con '-', blocchi fusi se adiacenti
    out = coding.grep(ctx, pattern="BERSAGLIO", path=".", context=1)
    check("grep ctx: match marcato ':'", "a.py:3: BERSAGLIO qui" in out, out)
    check("grep ctx: contesto marcato '-'", "a.py-2- due" in out and "a.py-4- quattro" in out, out)
    check("grep ctx: intervalli adiacenti fusi (riga 4 una sola volta)",
          out.count("quattro") == 1 and out.count("-5-") == 1, out)
    check("grep ctx: separatore tra blocchi/file", "--" in out)
    check("grep ctx: conteggio dei MATCH (non delle righe emesse)", out.startswith("3 matches"), out[:30])
    # coercion da stringa + clamp
    out2 = coding.grep(ctx, pattern="BERSAGLIO", path="a.py", context="1")
    check("grep ctx: context come stringa coercito", "a.py-2- due" in out2)
    out3 = coding.grep(ctx, pattern="BERSAGLIO", path="a.py", context=999)
    check("grep ctx: clamp a 10 (nessuna esplosione)", out3.count("\n") < 20, str(out3.count("\n")))
    # files_only: solo file+conteggio, niente testo delle righe
    outf = coding.grep(ctx, pattern="BERSAGLIO", files_only=True)
    check("grep files_only: elenca file con conteggio", "a.py (2)" in outf and "b.py (1)" in outf, outf)
    check("grep files_only: nessuna riga di testo", "qui" not in outf and ":3:" not in outf, outf)
    check("grep files_only: etichetta corretta", outf.startswith("2 files with matches"), outf[:40])
    outfc = coding.grep(ctx, pattern="BERSAGLIO", files_only="true", context=3)
    check("grep files_only: vince su context (e coercion stringa)", "a.py (2)" in outfc and "-2-" not in outfc)

    # move_path: rinomina, sposta con parents, semantica prudente, confinamento
    ok = coding.move_path(ctx, src="a.py", dst="renamed.py")
    check("move: rinomina ok", ok.startswith("✓") and (root / "renamed.py").exists()
          and not (root / "a.py").exists(), ok)
    check("move: contenuto preservato", "BERSAGLIO qui" in (root / "renamed.py").read_text())
    ok = coding.move_path(ctx, src="renamed.py", dst="nuova/dir/pro.py")
    check("move: crea cartelle intermedie", ok.startswith("✓") and (root / "nuova" / "dir" / "pro.py").exists(), ok)
    out = coding.move_path(ctx, src="sub/b.py", dst="nuova/dir/pro.py")
    check("move: destinazione esistente → rifiuto", out.startswith("❌") and "already exists" in out, out)
    out = coding.move_path(ctx, src="fantasma.py", dst="x.py")
    check("move: origine mancante → errore pulito", out.startswith("❌"), out)
    out = coding.move_path(ctx, src="sub", dst="sub2")
    check("move: sposta cartelle", out.startswith("✓") and (root / "sub2" / "b.py").exists(), out)
    out = coding.move_path(ctx, src="sub2", dst="sub2/dentro")
    check("move: destinazione inside the source → rifiuto", out.startswith("❌"), out)
    try:
        coding.move_path(ctx, src="renamed.py", dst="../fuori.py")
        escaped = False
    except ToolError:
        escaped = True
    check("move: uscita dalla root bloccata (dst)", escaped)
    try:
        coding.move_path(ctx, src="../../etc/passwd", dst="dentro.py")
        escaped = False
    except ToolError:
        escaped = True
    check("move: uscita dalla root bloccata (src)", escaped)
    check("move: è distruttivo (gate approvazione)", coding.move_path.destructive is True)


def test_honest_reads_and_inventory():
    import tempfile as _tf

    from flair.core.agent import Agent
    from flair.tools import fs

    # ── read_file onesto: header = range CONSEGNATO, hint sempre presente ────
    d = Path(_tf.mkdtemp(prefix="flair_hr_"))
    big = d / "big.py"
    big.write_text("\n".join(f"riga_{i} = {i}" for i in range(1, 2001)))
    out = fs.read_file_impl(None, str(big), offset=1, limit=None, max_chars=2000)
    import re as _re
    m = _re.search(r"\(lines (\d+)-(\d+) of (\d+)\)", out)
    lo, hi, tot = int(m.group(1)), int(m.group(2)), int(m.group(3))
    check("read onesto: header dichiara meno del totale", lo == 1 and hi < tot and tot == 2000, m.group(0))
    body_lines = [x for x in out.splitlines()[1:] if _re.match(r"\s*\d+ \| ", x)]
    check("read onesto: header == righe davvero consegnate", len(body_lines) == hi, f"{len(body_lines)} vs {hi}")
    check("read onesto: hint di continuazione presente e corretto",
          f"continue with read_file(path, offset={hi + 1})" in out, out[-90:])
    check("read onesto: budget rispettato", len(out) <= 2000, str(len(out)))
    out2 = fs.read_file_impl(None, str(big), offset=hi + 1, limit=None, max_chars=2000)
    check("read onesto: la continuazione riparte dal punto giusto", f"(lines {hi + 1}-" in out2, out2[:60])
    # file piccolo: comportamento invariato, nessun hint
    small = d / "s.py"
    small.write_text("a = 1\nb = 2\n")
    outs = fs.read_file_impl(None, str(small), offset=1, limit=None, max_chars=12000)
    check("read onesto: file piccolo integro senza hint", "(lines 1-2 of 2)" in outs and " more lines;" not in outs, outs)
    # riga singola enorme: consegna comunque qualcosa, con la rete _trunc
    mono = d / "min.js"
    mono.write_text("x" * 9000)
    outm = fs.read_file_impl(None, str(mono), offset=1, limit=None, max_chars=1000)
    check("read onesto: riga enorme → rete _trunc, nessun crash",
          outm.startswith(str(mono)[:0] + fs.display(None, mono)) and "truncated" in outm, outm[:60])

    # ── mechanical inventory dal transcript potato ────────────────────────────
    def rf_call(cid, path):
        return {"role": "assistant", "tool_calls": [{"id": cid, "type": "function",
                "function": {"name": "read_file", "arguments": json_module.dumps({"path": path})}}]}
    msgs = [
        rf_call("1", "pyproject.toml"),
        {"role": "tool", "tool_call_id": "1", "content": "pyproject.toml (lines 1-49 of 49)\n..."},
        rf_call("2", "tests/test_smoke.py"),
        {"role": "tool", "tool_call_id": "2", "content": "...\n...[2241 more lines; continue with read_file(path, offset=301)]"},
        rf_call("3", "pyproject.toml"),   # riletto: resta completo
        {"role": "tool", "tool_call_id": "3", "content": "pyproject.toml (lines 1-49 of 49)"},
        {"role": "assistant", "tool_calls": [{"id": "4", "type": "function",
            "function": {"name": "grep", "arguments": "{\"pattern\": \"x\"}"}}]},
        {"role": "tool", "tool_call_id": "4", "content": "1 corrispondenza"},
    ]
    inv = Agent._read_inventory(msgs)
    check("inventario: path in ordine di prima lettura, dedup", inv.startswith("pyproject.toml, tests/test_smoke.py"), inv)
    check("inventario: parziale marcato", "tests/test_smoke.py (partial)" in inv, inv)
    check("inventario: completo non marcato", "pyproject.toml (partial)" not in inv, inv)
    check("inventario: tool non-read ignorati", "grep" not in inv, inv)
    many = []
    for i in range(80):
        many.append(rf_call(f"m{i}", f"cartella/file_{i:03}.py"))
        many.append({"role": "tool", "tool_call_id": f"m{i}", "content": "ok (lines 1-3 of 3)"})
    invm = Agent._read_inventory(many)
    check("inventario: tetto con conteggio dei rimanenti", len(invm) < 1000 and "… and" in invm, invm[-30:])
    check("inventario: vuoto se nessuna read_file", Agent._read_inventory(msgs[-2:]) == "")


def test_deepseek_reasoning_effort():
    import os as _os

    from flair.config import load_config
    from flair.llm.deepseek import DeepSeekProvider

    def mk(effort=None):
        _os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        if effort is None:
            _os.environ.pop("DEEPSEEK_REASONING_EFFORT", None)
        else:
            _os.environ["DEEPSEEK_REASONING_EFFORT"] = effort
        cfg = load_config()
        cfg.provider = "deepseek"
        return DeepSeekProvider(cfg)

    # --think senza effort configurato: solo il parametro thinking, nessun effort
    prov = mk()
    params = prov._build_params([], None, think=True, max_tokens=100)
    check("effort ds: --think attiva thinking", params.get("extra_body") == {"thinking": {"type": "enabled"}}, params)
    check("effort ds: senza env nessun reasoning_effort", "reasoning_effort" not in params, params)
    # fast mode: comportamento invariato (nessun parametro thinking/effort → default server)
    params = prov._build_params([], None, think=False, max_tokens=100)
    check("effort ds: fast mode intatto (no thinking param)", "extra_body" not in params, params)
    check("effort ds: fast mode intatto (no effort)", "reasoning_effort" not in params, params)
    check("effort ds: fast mode manda temperature", params.get("temperature") == 0.0, params)
    # env configurata: pass-through verbatim col --think
    prov = mk("max")
    params = prov._build_params([], None, think=True, max_tokens=100)
    check("effort ds: env letta e inviata verbatim", params.get("reasoning_effort") == "max", params)
    check("effort ds: thinking sempre presente col --think", params.get("extra_body") == {"thinking": {"type": "enabled"}}, params)
    # ...ma MAI sul fast, anche se configurata (proprietà del --think)
    params = prov._build_params([], None, think=False, max_tokens=100)
    check("effort ds: env configurata ma fast pulito", "reasoning_effort" not in params and "extra_body" not in params, params)
    # alias legacy: niente parametro thinking (modalità nel nome), niente effort
    _os.environ["DEEPSEEK_THINK_MODEL"] = "deepseek-reasoner"
    prov = mk("max")
    params = prov._build_params([], None, think=True, max_tokens=100)
    _os.environ.pop("DEEPSEEK_THINK_MODEL", None)
    check("effort ds: alias legacy senza thinking param", "extra_body" not in params and "reasoning_effort" not in params, params)
    _os.environ.pop("DEEPSEEK_REASONING_EFFORT", None)


def test_reasoning_passback():
    import tempfile as _tf

    from flair.core.agent import Agent
    from flair.llm.deepseek import DeepSeekProvider
    from flair.llm.openai import OpenAIProvider

    # ── _assistant_msg: il reasoning viaggia SOLO nei turni con tool call ────
    resp = LLMResponse(reasoning="piano segreto", tool_calls=[tc("read_file", path="a.py")],
                       usage=Usage(total_tokens=1))
    msg = Agent._assistant_msg(None, resp)
    check("passback: reasoning incluso nel turno con tool", msg.get("reasoning_content") == "piano segreto", msg)
    resp = LLMResponse(tool_calls=[tc("read_file", path="a.py")], usage=Usage(total_tokens=1))
    check("passback: senza reasoning niente campo", "reasoning_content" not in Agent._assistant_msg(None, resp))
    resp = LLMResponse(content="fine", reasoning="pensieri finali", usage=Usage(total_tokens=1))
    check("passback: turno finale senza tool → campo omesso (il server lo ignorerebbe)",
          "reasoning_content" not in Agent._assistant_msg(None, resp))

    # ── stima token: le tracce ora pesano nel contesto ───────────────────────
    base_msgs = [{"role": "assistant", "content": "x" * 40}]
    with_r = [{"role": "assistant", "content": "x" * 40, "reasoning_content": "r" * 400}]
    check("passback: _estimate_tokens conta il reasoning",
          Agent._estimate_tokens(with_r) - Agent._estimate_tokens(base_msgs) == 100)

    # ── sanitizzazione per-provider (switch /provider a metà sessione) ───────
    import os as _os
    _os.environ["DEEPSEEK_API_KEY"] = "sk-test"
    _os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg = load_config()
    history = [
        {"role": "user", "content": "ciao"},
        {"role": "assistant", "content": "", "reasoning_content": "trace", "tool_calls": []},
    ]
    cfg.provider = "openai"
    oai = OpenAIProvider(cfg)
    sent = oai._build_params(history, None, think=False, max_tokens=64)["messages"]
    check("passback: OpenAI spoglia il campo a request-time", "reasoning_content" not in sent[1], sent[1])
    check("passback: la cronologia NON viene mutata", history[1]["reasoning_content"] == "trace")
    check("passback: messaggi puliti passano per identità (zero copie inutili)", sent[0] is history[0])
    cfg2 = load_config()
    cfg2.provider = "deepseek"
    ds = DeepSeekProvider(cfg2)
    sent = ds._build_params(history, None, think=False, max_tokens=64)["messages"]
    check("passback: DeepSeek conserva il campo", sent[1].get("reasoning_content") == "trace")
    check("passback: DeepSeek passa la lista senza copie", sent is history)

    # ── end-to-end: la traccia entra in cronologia e arriva al provider dopo ─
    root = Path(_tf.mkdtemp(prefix="flair_rp_")).resolve()
    (root / "a.py").write_text("x = 1\n")
    cfg3 = cfg_for(root)
    fake = FakeProvider([
        LLMResponse(reasoning="prima leggo a.py, poi decido",
                    tool_calls=[tc("read_file", path="a.py")], usage=Usage(total_tokens=1)),
        LLMResponse(content="Fatto.", reasoning="ho visto abbastanza", usage=Usage(total_tokens=1)),
    ])
    agent = coding_agent.build(cfg3, fake)
    out = agent.run("leggi a.py")
    check("passback e2e: run completata", out.content == "Fatto.")
    tool_turns = [m for m in agent.convo.messages
                  if m.get("role") == "assistant" and m.get("tool_calls")]
    check("passback e2e: la traccia è in cronologia sul turno con tool",
          tool_turns and tool_turns[0].get("reasoning_content") == "prima leggo a.py, poi decido")
    finals = [m for m in agent.convo.messages
              if m.get("role") == "assistant" and not m.get("tool_calls")]
    check("passback e2e: il turno finale resta senza campo",
          finals and "reasoning_content" not in finals[-1])
    second_call_msgs = fake.seen[1]
    sent_tool_turn = [m for m in second_call_msgs
                      if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")]
    check("passback e2e: la richiesta successiva porta il reasoning al provider",
          sent_tool_turn and sent_tool_turn[0].get("reasoning_content") == "prima leggo a.py, poi decido")


def test_reasoning_regimes():
    import os as _os
    import tempfile as _tf

    from flair.llm.deepseek import DeepSeekProvider

    # ── FLAIR_THINK_STEPS: parsing e validazione ─────────────────────────────
    _os.environ.pop("FLAIR_THINK_STEPS", None)
    _os.environ["DEEPSEEK_API_KEY"] = "sk-test"
    check("regimi: default think_steps=first", load_config().think_steps == "first")
    _os.environ["FLAIR_THINK_STEPS"] = "ALL"
    check("regimi: env 'ALL' normalizzata", load_config().think_steps == "all")
    _os.environ["FLAIR_THINK_STEPS"] = "sempre"
    try:
        load_config()
        check("regimi: valore invalido rifiutato", False)
    except ValueError as exc:
        check("regimi: valore invalido rifiutato", "FLAIR_THINK_STEPS" in str(exc), exc)
    _os.environ.pop("FLAIR_THINK_STEPS", None)

    # ── il knob modula il loop: first = solo step 0; all = tutti gli step ────
    root = Path(_tf.mkdtemp(prefix="flair_ts_")).resolve()
    (root / "a.py").write_text("x = 1\n")

    def script():
        return [
            LLMResponse(tool_calls=[tc("read_file", path="a.py")], usage=Usage(total_tokens=1)),
            LLMResponse(tool_calls=[tc("list_directory", path=".")], usage=Usage(total_tokens=1)),
            LLMResponse(content="Fatto.", usage=Usage(total_tokens=1)),
        ]

    cfg1 = cfg_for(root)
    fake1 = FakeProvider(script())
    coding_agent.build(cfg1, fake1).run("task", think=True)
    thinks = [c["think"] for c in fake1.calls]
    check("regimi: first → think solo allo step 0", thinks == [True, False, False], thinks)

    cfg2 = cfg_for(root)
    cfg2.think_steps = "all"
    fake2 = FakeProvider(script())
    coding_agent.build(cfg2, fake2).run("task", think=True)
    thinks = [c["think"] for c in fake2.calls]
    check("regimi: all → think su ogni step del turno", thinks == [True, True, True], thinks)

    cfg3 = cfg_for(root)
    cfg3.think_steps = "all"
    fake3 = FakeProvider(script())
    coding_agent.build(cfg3, fake3).run("task", think=False)
    thinks = [c["think"] for c in fake3.calls]
    check("regimi: all senza --think non forza nulla", thinks == [False, False, False], thinks)

    # ── DEEPSEEK_FAST_REASONING_EFFORT: la via di mezzo, opt-in ──────────────
    _os.environ["DEEPSEEK_FAST_REASONING_EFFORT"] = "max"
    _os.environ.pop("DEEPSEEK_REASONING_EFFORT", None)
    c = load_config()
    c.provider = "deepseek"
    ds = DeepSeekProvider(c)
    fast = ds._build_params([], None, think=False, max_tokens=64)
    check("regimi: fast effort → thinking esplicito",
          fast.get("extra_body") == {"thinking": {"type": "enabled"}}, fast)
    check("regimi: fast effort → parametro inviato verbatim", fast.get("reasoning_effort") == "max", fast)
    thinkp = ds._build_params([], None, think=True, max_tokens=64)
    check("regimi: sul --think comanda l'altro knob (qui assente)", "reasoning_effort" not in thinkp, thinkp)
    check("regimi: --think mantiene il thinking esplicito",
          thinkp.get("extra_body") == {"thinking": {"type": "enabled"}})
    _os.environ["DEEPSEEK_REASONING_EFFORT"] = "high"
    c2 = load_config()
    c2.provider = "deepseek"
    both = DeepSeekProvider(c2)._build_params([], None, think=True, max_tokens=64)
    check("regimi: knob indipendenti (think usa il suo, 'high')", both.get("reasoning_effort") == "high", both)
    _os.environ.pop("DEEPSEEK_REASONING_EFFORT", None)
    _os.environ.pop("DEEPSEEK_FAST_REASONING_EFFORT", None)
    c3 = load_config()
    c3.provider = "deepseek"
    clean = DeepSeekProvider(c3)._build_params([], None, think=False, max_tokens=64)
    check("regimi: senza knob il fast resta byte-identico",
          "extra_body" not in clean and "reasoning_effort" not in clean, clean)
    _os.environ["DEEPSEEK_FAST_REASONING_EFFORT"] = "max"
    _os.environ["DEEPSEEK_MODEL"] = "deepseek-chat"
    c4 = load_config()
    c4.provider = "deepseek"
    leg = DeepSeekProvider(c4)._build_params([], None, think=False, max_tokens=64)
    check("regimi: alias legacy intatti anche col knob",
          "extra_body" not in leg and "reasoning_effort" not in leg, leg)
    _os.environ.pop("DEEPSEEK_FAST_REASONING_EFFORT", None)
    _os.environ.pop("DEEPSEEK_MODEL", None)


def test_cost_attribution():
    import os as _os

    from flair.config import price_for
    from flair.llm.base import Usage, _usage_cost
    from flair.llm.deepseek import DeepSeekProvider

    _os.environ["DEEPSEEK_API_KEY"] = "sk-test"
    for k in ("FLAIR_PRICE_CACHE_HIT", "FLAIR_PRICE_CACHE_MISS", "FLAIR_PRICE_OUTPUT"):
        _os.environ.pop(k, None)

    # ── price_for: listino del modello indicato, non del modello attivo ──────
    check("costi: price_for distingue flash e pro",
          price_for("deepseek", "deepseek-v4-flash") == (0.0028, 0.14, 0.28)
          and price_for("deepseek", "deepseek-v4-pro") == (0.003625, 0.435, 0.87))
    _os.environ["FLAIR_PRICE_CACHE_MISS"] = "9.9"
    check("costi: override env vince campo per campo, su ogni modello",
          price_for("deepseek", "deepseek-v4-pro")[1] == 9.9
          and price_for("deepseek", "deepseek-v4-flash")[1] == 9.9
          and price_for("deepseek", "deepseek-v4-pro")[2] == 0.87)
    _os.environ.pop("FLAIR_PRICE_CACHE_MISS", None)

    # ── Usage: il costo si somma come i token ────────────────────────────────
    a = Usage(prompt_tokens=10, cost_usd=0.5)
    b = Usage(prompt_tokens=5, cost_usd=0.25)
    check("costi: __add__ somma cost_usd", (a + b).cost_usd == 0.75)
    check("costi: default 0.0 (nessuna attribuzione)", Usage().cost_usd == 0.0)

    # ── _request_cost: la stessa Usage costa diverso su flash e su pro ───────
    cfgd = load_config()
    cfgd.provider = "deepseek"
    ds = DeepSeekProvider(cfgd)
    u = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000,
              cache_hit_tokens=0, cache_miss_tokens=1_000_000)
    flash = ds._request_cost(u, "deepseek-v4-flash")
    pro = ds._request_cost(u, "deepseek-v4-pro")
    check("costi: richiesta prezzata col modello reale (flash)", abs(flash - (0.14 + 0.28)) < 1e-9, flash)
    check("costi: richiesta prezzata col modello reale (pro)", abs(pro - (0.435 + 0.87)) < 1e-9, pro)

    # ── estimate_cost: preferisce l'accumulato, ricade sull'aggregato ────────
    attributed = Usage(prompt_tokens=1_000_000, cache_miss_tokens=1_000_000, cost_usd=1.234)
    check("costi: estimate_cost usa l'accumulato quando c'è",
          ds.estimate_cost(attributed, cfgd) == 1.234)
    legacy = Usage(prompt_tokens=1_000_000, cache_miss_tokens=1_000_000)
    expected = _usage_cost(legacy, cfgd.price_cache_hit, cfgd.price_cache_miss, cfgd.price_output)
    check("costi: senza attribuzione ricade sul listino unico (retrocompat)",
          abs(ds.estimate_cost(legacy, cfgd) - expected) < 1e-12)

    # ── il caso che ha motivato il fix: turno misto flash+pro ────────────────
    step_pro = Usage(cache_miss_tokens=100_000, completion_tokens=10_000)
    step_pro.cost_usd = ds._request_cost(step_pro, "deepseek-v4-pro")
    step_flash = Usage(cache_miss_tokens=100_000, completion_tokens=10_000)
    step_flash.cost_usd = ds._request_cost(step_flash, "deepseek-v4-flash")
    turn = step_pro + step_flash
    check("costi: turno misto = somma dei listini reali (non 2x flash)",
          abs(ds.estimate_cost(turn, cfgd) - (step_pro.cost_usd + step_flash.cost_usd)) < 1e-12
          and turn.cost_usd > 2 * step_flash.cost_usd)


def main():
    test_arg_parse()
    test_usage_normalization()
    test_reasoning_detection()
    test_router()
    test_apply_edit()
    test_provider_request_path()
    test_streaming_assembly()
    test_coding_agent()
    test_general_agent()
    test_approval_gate()
    test_compaction()
    test_safe_split()
    test_compaction_valid_pairing()
    test_overflow_retry()
    test_web_search()
    test_session_persistence()
    test_session_memory()
    test_grep_context_and_move()
    test_honest_reads_and_inventory()
    test_deepseek_reasoning_effort()
    test_reasoning_passback()
    test_reasoning_regimes()
    test_cost_attribution()
    test_parallel_tools()
    test_cli_session_roundtrip()
    test_shared_memory()
    test_router_llm()
    test_multi_edit()
    test_web_fetch()
    test_runtime_switch_and_context()
    test_streaming_reasoning_order()
    test_system_write_edit()
    test_cli_always_per_tool()
    test_approval_prompt_brackets()
    test_help_renders()
    test_stop_flow()
    test_keyboard_interrupt_during_model_call()
    test_keyboard_interrupt_mid_tools()
    test_repl_survives_turn_error()
    test_streaming_fallback_no_duplication()
    test_cost_estimate_cache()
    test_run_once_exit_codes()
    test_tools_command()
    test_cli_approve_stop_and_yes()
    test_shell_multiline_routing()
    test_shell_decoding_robust()
    test_search_files_coercion()
    test_bool_coercion()
    test_powershell_temp_cleanup()
    test_tool_schemas()
    test_tool_robustness()
    test_root_chdir()
    test_arg_coercion()
    test_truncated_args_guidance()
    test_finish_reason_truncation_note()
    test_write_file_append()
    test_atomic_writes()
    test_repo_map()
    test_repo_map_languages()
    test_explore_subagent()
    test_explore_usage_accounting()
    test_explore_usage_on_stop()
    test_router_usage_accounting()
    test_router_continuation()
    test_plan_tool()
    test_prune_superseded_rules()
    test_prune_in_agent()
    test_budget_abort()
    test_read_only_mode()
    test_automation_helpers()
    test_evals_harness()
    print(f"\nTUTTI I {len(PASS)} TEST PASSATI ✅")


if __name__ == "__main__":
    main()
