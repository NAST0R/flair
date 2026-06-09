"""Runner del banco di prova (eval harness) di flair.

Esegue ogni task su una cartella temporanea con un agente reale e verifica il
risultato, riportando successo, passi, token e cache-hit per task. Dal vivo
servono le API key (le stesse di flair, lette dall'ambiente / .env).

  python tests/evals/run_evals.py                # esegue tutti i task (dal vivo)
  python tests/evals/run_evals.py fix-failing    # solo i task il cui nome contiene "fix-failing"
  python tests/evals/run_evals.py --list         # elenca i task
  python tests/evals/run_evals.py --think         # usa il modello "think"
  python tests/evals/run_evals.py --self-test     # prova il runner SENZA rete (provider fittizio)

Codice d'uscita: 0 se tutti i task passano, 1 altrimenti (utile in CI).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

# Rende importabili sia il pacchetto `flair` (radice repo) sia `tasks` (questa cartella),
# qualunque sia il modo di invocazione.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from tasks import TASKS, EvalTask  # noqa: E402

from flair.agents import coding as coding_agent  # noqa: E402
from flair.agents import general as general_agent  # noqa: E402
from flair.config import load_config  # noqa: E402


def _build_agent(cfg, provider, which: str):
    builder = general_agent if which == "general" else coding_agent
    return builder.build(cfg, provider)


def _estimate_cost(provider, usage, cfg) -> float:
    # I prezzi (price_cache_hit/miss/output) vivono sul Config intero, come per la
    # CLI: passare cfg.active (ProviderConfig) darebbe AttributeError → costo 0.
    try:
        return provider.estimate_cost(usage, cfg)
    except Exception:
        return 0.0


def _run_one(task: EvalTask, provider_factory, cfg, think: bool) -> dict:
    wd = Path(tempfile.mkdtemp(prefix=f"flaireval_{task.name}_"))
    try:
        task.setup(wd)
        cfg.root = wd.resolve()
        cfg.auto_approve = True
        provider = provider_factory(cfg)
        agent = _build_agent(cfg, provider, task.agent)
        result = agent.run(task.prompt, think=think or task.think)
        ok = bool(task.check(wd, result.content or ""))
        u = agent.convo.total_usage
        return {
            "name": task.name, "ok": ok, "steps": result.steps,
            "tokens": u.total_tokens, "hit": u.cache_hit_tokens, "miss": u.cache_miss_tokens,
            "cost": _estimate_cost(provider, u, cfg),
        }
    except Exception as exc:  # un task non deve far cadere l'intero giro
        return {"name": task.name, "ok": False, "steps": 0, "tokens": 0,
                "hit": 0, "miss": 0, "cost": 0.0, "error": repr(exc)}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ── Provider fittizio per il self-test (nessuna rete) ─────────────────────────
class _FakeProvider:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def complete(self, messages, tools=None, think=False, max_tokens=None, stream=False,
                 on_delta=None, on_reasoning=None):
        from flair.llm.base import LLMResponse, Usage
        if self.i >= len(self.script):
            return LLMResponse(content="FINE", usage=Usage(total_tokens=1))
        r = self.script[self.i]
        self.i += 1
        return r

    def estimate_cost(self, usage, cfg) -> float:
        return 0.0


def _selftest_provider_factory(cfg):
    from flair.llm.base import LLMResponse, ToolCall, Usage
    slug_src = (
        "import re\n\n\n"
        "def slugify(s):\n"
        "    s = s.strip().lower()\n"
        "    s = re.sub(r'[^a-z0-9\\s-]', '', s)\n"
        "    s = re.sub(r'\\s+', '-', s)\n"
        "    return s\n"
    )
    return _FakeProvider([
        LLMResponse(
            tool_calls=[ToolCall(id="c1", name="write_file",
                                 arguments={"path": "textutils.py", "content": slug_src})],
            usage=Usage(prompt_tokens=50, completion_tokens=30, total_tokens=80,
                        cache_hit_tokens=40, cache_miss_tokens=10)),
        LLMResponse(content="Aggiunta slugify in textutils.py.",
                    usage=Usage(prompt_tokens=90, completion_tokens=10, total_tokens=100,
                                cache_hit_tokens=80, cache_miss_tokens=10)),
    ])


def _print_table(rows: list[dict]) -> int:
    print(f"\n{'TASK':<18} {'ESITO':<8} {'PASSI':>5} {'TOKEN':>8} {'CACHE-HIT':>10} {'COSTO$':>9}")
    print("─" * 62)
    passed = 0
    for r in rows:
        passed += r["ok"]
        tot = r["hit"] + r["miss"]
        hit_pct = f"{(100 * r['hit'] / tot):.0f}%" if tot else "—"
        print(f"{r['name']:<18} {'PASS ✅' if r['ok'] else 'FAIL ❌':<8} "
              f"{r['steps']:>5} {r['tokens']:>8} {hit_pct:>10} {r['cost']:>9.4f}")
        if r.get("error"):
            print(f"  ⚠ errore: {r['error']}")
    print("─" * 62)
    print(f"Passati {passed}/{len(rows)}\n")
    return 0 if passed == len(rows) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Banco di prova di flair")
    ap.add_argument("filter", nargs="?", default="", help="esegue solo i task il cui nome contiene questa stringa")
    ap.add_argument("--list", action="store_true", help="elenca i task e termina")
    ap.add_argument("--think", action="store_true", help="usa il modello think")
    ap.add_argument("--self-test", action="store_true", help="prova il runner senza rete (provider fittizio)")
    args = ap.parse_args(argv)

    tasks = [t for t in TASKS if args.filter in t.name]

    if args.list:
        for t in tasks:
            print(f"{t.name:<18} [{t.agent}] {t.description}")
        return 0

    if args.self_test:
        # Esegue offline solo il task che il provider fittizio è scritto per risolvere.
        tasks = [t for t in TASKS if t.name == "add-function"]
        cfg = load_config()
        rows = [_run_one(t, _selftest_provider_factory, cfg, think=False) for t in tasks]
        return _print_table(rows)

    from flair.llm.factory import create_provider
    cfg = load_config()
    rows = [_run_one(t, create_provider, cfg, think=args.think) for t in tasks]
    return _print_table(rows)


if __name__ == "__main__":
    raise SystemExit(main())
