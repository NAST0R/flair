"""Suite di test offline (nessuna rete).

Esercita: normalizzazione usage dei due provider, parsing robusto args,
rilevamento reasoning model, router euristico, ed entrambi gli agenti
end-to-end (con un provider fittizio) sui tool reali — coding sandboxato e
generico cross-platform.
"""

from __future__ import annotations

import io
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

    def estimate_cost(self, usage, cfg):
        return 0.0


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
    check("coding: edit ambiguo rifiutato", "compare 2 volte" in tmsg and (root / "amb.py").read_text() == "x = 1\nx = 1\n")

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
    # write_file non è nei tool generici → usa coding per il gate
    agent = coding_agent.build(cfg, fake, approve=lambda name, args: False)
    agent.run("scrivi")
    check("approval: edit distruttivo negato non scrive", (root / "f.py").read_text() == "a=1\n")
    tmsg = [m for m in agent.messages if m["role"] == "tool"][0]["content"]
    check("approval: messaggio di annullamento", "annullata" in tmsg)


# ── 9. matcher resiliente di edit_file ────────────────────────────────────────

def test_apply_edit():
    # esatto
    r, k = apply_edit("def f():\n    return 1\n", "return 1", "return 2")
    check("apply_edit: esatto", k == "esatto" and "return 2" in r)
    # fine-riga tollerato (trailing space nel file)
    r, k = apply_edit("def f():\n    x = 1   \n    return x\n", "    x = 1\n    return x",
                      "    x = 2\n    return x")
    check("apply_edit: fine-riga", k == "fine-riga tollerato" and "x = 2" in r, k)
    # indentazione tollerata + re-indentazione corretta
    src = "class A:\n    def m(self):\n        a = 1\n        b = 2\n        return a + b\n"
    r, k = apply_edit(src, "  a = 1\n  b = 2\n  return a + b", "  a = 10\n  b = 20\n  return a * b")
    check("apply_edit: indentazione", k == "indentazione tollerata", k)
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
    agent.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "r1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "r2"},
    ]
    for keep in range(0, 7):
        split = agent._safe_split(keep)
        ok = split >= len(agent.messages) or agent.messages[split]["role"] != "tool"
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
    agent.messages += [
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
    check("web fail: errore onesto", out.startswith("❌ Nessun risultato") and "TAVILY_API_KEY" in out, out)
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
    agent.messages.append({"role": "user", "content": "ciao"})
    agent.messages.append({"role": "assistant", "content": "ciao a te"})
    agent.total_usage = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                              cache_hit_tokens=50, cache_miss_tokens=50, reasoning_tokens=5)

    store = SessionStore(cfg.session_dir)
    path = store.save("lavoro", {"last_agent": "coding", "agents": {"coding": agent.dump_state()}})
    check("sessione: salvataggio crea file", path is not None and path.exists())
    check("sessione: exists()", store.exists("lavoro"))
    check("sessione: latest()", store.latest() == "lavoro")

    agent2 = coding_agent.build(cfg, prov)
    state = store.load("lavoro")
    agent2.load_state(state["agents"]["coding"])
    check("sessione: messaggi ripristinati",
          [m.get("content") for m in agent2.messages[-2:]] == ["ciao", "ciao a te"])
    check("sessione: uso ripristinato",
          agent2.total_usage.total_tokens == 120 and agent2.total_usage.cache_hit_tokens == 50)
    check("sessione: caricamento mancante → None", store.load("inesistente") is None)


def test_cli_session_roundtrip():
    from flair.cli import CLI
    root = Path("/tmp/flair_cli_sess")
    shutil.rmtree(root, ignore_errors=True)
    cfg = cfg_for(root)
    cfg.session_dir = root / "sessions"
    cli = CLI(cfg)
    cli.agents["general"].messages.append({"role": "user", "content": "ricordami"})
    cli.last_agent = "general"
    cli.session_name = "s1"
    cli._save_session()

    cli2 = CLI(cfg)
    check("cli sessione: load ok", cli2._load_session("s1"))
    check("cli sessione: messaggio presente",
          any(m.get("content") == "ricordami" for m in cli2.agents["general"].messages))
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
    check("multi_edit: due modifiche applicate", out.startswith("✓") and "2 modifiche" in out)
    check("multi_edit: contenuto corretto", (root / "a.py").read_text() == "x = 10\ny = 2\nz = 30\n")

    before = (root / "a.py").read_text()
    out = coding_tools.multi_edit(ctx, path="a.py", edits=[
        {"old_string": "x = 10", "new_string": "x = 99"},
        {"old_string": "NON_ESISTE", "new_string": "!"},
    ])
    check("multi_edit: fallimento indica la modifica", out.startswith("❌ Modifica #2"))
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
    check("web_fetch: errore pulito", out.startswith("❌ Impossibile scaricare"))


def test_runtime_switch_and_context():
    from flair.llm import create_provider
    cfg = cfg_for(Path("."))
    cfg.provider = "deepseek"
    check("switch: factory deepseek", isinstance(create_provider(cfg), DeepSeekProvider) and cfg.active is cfg.deepseek)
    cfg.provider = "openai"
    check("switch: factory openai", isinstance(create_provider(cfg), OpenAIProvider) and cfg.active is cfg.openai)

    prov = SimpleNamespace()
    agent = coding_agent.build(cfg, prov)
    agent.messages.append({"role": "user", "content": "x" * 4000})
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
    test_overflow_retry()
    test_web_search()
    test_session_persistence()
    test_cli_session_roundtrip()
    test_multi_edit()
    test_web_fetch()
    test_runtime_switch_and_context()
    test_streaming_reasoning_order()
    test_tool_schemas()
    print(f"\nTUTTI I {len(PASS)} TEST PASSATI ✅")


if __name__ == "__main__":
    main()
