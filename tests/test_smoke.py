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
    agent.convo.messages.append({"role": "user", "content": "ciao"})
    agent.convo.messages.append({"role": "assistant", "content": "ciao a te"})
    agent.convo.total_usage = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                                    cache_hit_tokens=50, cache_miss_tokens=50, reasoning_tokens=5)

    store = SessionStore(cfg.session_dir)
    path = store.save("lavoro", {"last_agent": "coding", "conversation": agent.convo.dump()})
    check("sessione: salvataggio crea file", path is not None and path.exists())
    check("sessione: exists()", store.exists("lavoro"))
    check("sessione: latest()", store.latest() == "lavoro")

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
    check("system write_file: crea", target.exists() and "Creato" in out)
    check("system write_file: contenuto", target.read_text() == "# Titolo\n\nCorpo.\n")
    out = st.write_file(ctx, path=str(target), content="nuovo")
    check("system write_file: sovrascrive", "Sovrascritto" in out and target.read_text() == "nuovo")

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
    c.print(r"procedo? \[y]es / \[n]o / \[a]lways")  # parentesi escape-ate
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
    check("help: argomenti opzionali [nome] mostrati", "[nome]" in out and "<task>" in out, out[:400])


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
    check("stop: l'interruzione diventa informazione", any("Interrotto dall'utente" in t for t in tmsgs))


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
    check("ctrl-c (tool): interruzione registrata", any("Interrotto dall'utente" in t for t in tmsgs))


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
    # A3: la modalità one-shot non crasha; ritorna un exit code pulito.
    import io as _io

    from rich.console import Console

    from flair.cli import CLI
    cli = CLI(cfg_for(Path(".")))
    cli.console = Console(file=_io.StringIO())

    cli.run_task = lambda *a, **k: None          # type: ignore  # successo
    check("A3: run_once ok → 0", cli.run_once("ciao") == 0)

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

    # 3) PowerShell assente (sandbox): il tool risponde con errore pulito, niente crash.
    from flair.tools import system as st
    out = st.run_powershell(ToolContext(cfg=cfg_for(Path("."))), script="Write-Output hi", timeout=5)
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
    check("grep: path inesistente → ❌", r.startswith("❌") and "non esiste" in r, r)
    g = coding.glob(ctx, pattern="*.py", path="non_esiste")
    check("glob: path inesistente → ❌", g.startswith("❌"), g)

    # 2) grep puntato su un FILE → cerca in quel file (prima tornava vuoto in silenzio).
    r2 = coding.grep(ctx, pattern="class Bar", path="alpha.py")
    check("grep: su un singolo file trova le corrispondenze",
          "alpha.py" in r2 and "Nessuna" not in r2 and not r2.startswith("❌"), r2)

    # Caso normale invariato: ricorsivo su cartella, attraversa le sottocartelle.
    r3 = coding.grep(ctx, pattern="foo")
    check("grep: ricorsivo su cartella (invariato)", "alpha.py" in r3 and "beta.txt" in r3, r3)

    # 3) Argomento sconosciuto: il tool gira lo stesso, con nota; nessuna eccezione.
    r4 = coding.read_file(ctx, path="alpha.py", raw=True)
    check("dispatch: kwarg sconosciuto ignorato con nota",
          r4.startswith("ℹ️ Argomenti ignorati") and "raw" in r4.splitlines()[0], r4.splitlines()[0])
    r5 = coding.read_file(ctx, path="alpha.py", limit=1)
    check("dispatch: kwarg validi → nessuna nota", not r5.startswith("ℹ️"), r5.splitlines()[0])

    # Un argomento OBBLIGATORIO mancante resta un errore: i refusi seri restano visibili.
    raised = False
    try:
        coding.grep(ctx, path="alpha.py")  # manca 'pattern'
    except TypeError:
        raised = True
    check("dispatch: obbligatorio mancante → TypeError", raised)

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
              not out.startswith("❌") and "riga2" in out and "righe 2-2" in out, out.splitlines()[0])
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
        check("troncamento: suggerisce append/parti", "append=true" in out and "troncat" in out.lower(), out)
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
              "RIPRENDI" in stored and "risposta a metà" in stored, stored[-120:])

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
        check("finish_reason: la CLI mostra la nota anche in streaming", "troncata" in out, out[-160:])

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
              "FLAIR_MAX_TOKENS" in out2 and "Nessuna risposta" in out2, out2[-200:])
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
        check("append: messaggio 'aggiunto in coda'", "Aggiunto in coda" in out, out)
        check("append: contenuto concatenato", (root / "big.txt").read_text() == "parte1\nparte2\n")
        # append su file inesistente = crea
        out2 = coding.write_file(ctx, path="nuovo.txt", content="x\n", append=True)
        check("append: su file nuovo crea", (root / "nuovo.txt").read_text() == "x\n" and "Creato" in out2)
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
    print(f"\nTUTTI I {len(PASS)} TEST PASSATI ✅")


if __name__ == "__main__":
    main()
