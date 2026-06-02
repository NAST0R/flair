"""CLI interattiva (e batch) per flair.

Uso:
    flair                                   # REPL: instrada da solo tra coding e generico
    flair --provider openai                 # usa OpenAI invece di DeepSeek
    flair --root /path/progetto             # radice di lavoro per l'agente coding
    flair -p "apri youtube"                 # one-shot
    flair --agent coding -p "spiega auth/"  # forza un agente
    flair --think -p "rifattorizza X"       # primo passo col modello thinking
    flair --yes                             # auto-approva i tool distruttivi
    flair --no-stream                       # disabilita lo streaming
    flair --log ./logs                      # scrive il log di sessione (JSONL)
    flair --session lavoro                  # usa/crea la sessione "lavoro" (autosalvataggio)
    flair --continue                        # riprende l'ultima sessione salvata
    flair --version                         # stampa la versione

Comandi nel REPL:
    /code <task>          forza l'agente di coding
    /do <task>            forza l'agente generico
    /think <task>         esegue col modello thinking al primo passo
    /agent                mostra l'agente corrente (sticky)
    /provider [nome]      mostra, oppure cambia provider a runtime (deepseek|openai)
    /model <nome>         cambia il modello veloce a runtime
    /think-model <nome>   cambia il modello thinking a runtime
    /compact              compatta subito il contesto dell'agente attivo
    /cost                 riepilogo token/costo della sessione
    /save [nome]          salva la sessione (default: nome corrente)
    /load <nome>          riprende una sessione salvata
    /sessions             elenca le sessioni salvate
    /reset                azzera la conversazione di entrambi gli agenti
    /root <path>          cambia la radice di lavoro (ricarica le istruzioni di progetto)
    /help                 aiuto
    exit | quit           esci
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .agents import coding as coding_agent
from .agents import general as general_agent
from .config import Config, load_config
from .core import router
from .core.tool import ToolError
from .llm import Usage, create_provider
from .session_log import SessionLogger, setup_file_logging
from .session_store import SessionStore
from .tools import fs

_TOOL_ICON = {
    "read_file": "📄", "list_directory": "📁", "glob": "🔎", "grep": "🔎",
    "edit_file": "✏️ ", "multi_edit": "✏️ ", "write_file": "📝", "run_command": "⚙️ ",
    "open_url": "🌐", "open_path": "📂", "open_application": "🚀",
    "search_files": "🔦", "system_info": "🖥️ ", "get_datetime": "🕒",
    "clipboard_get": "📋", "clipboard_set": "📋", "web_search": "🌍", "web_fetch": "🌍",
}


def _short(v, n: int = 70) -> str:
    s = str(v).replace("\n", "↵")
    return s if len(s) <= n else s[: n - 1] + "…"


def _kfmt(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


class CLI:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.console = Console()
        self.provider = create_provider(cfg)
        self.last_agent: str | None = None
        self._mid_line = False
        self._turn_tools: list[dict] = []
        self._always_allow: set[str] = set()
        self._cost_warned = False

        self.session = SessionStore(cfg.session_dir) if cfg.session_dir else SessionStore(Path.home() / ".flair" / "sessions")
        self.session_name: str | None = None

        self.logger: SessionLogger | None = None
        if cfg.log_dir:
            setup_file_logging(cfg.log_dir)
            self.logger = SessionLogger(cfg.log_dir)

        self.agents = {
            "coding": coding_agent.build(cfg, self.provider, **self._callbacks()),
            "general": general_agent.build(cfg, self.provider, **self._callbacks()),
        }

    def _callbacks(self) -> dict:
        return dict(
            on_tool=self._on_tool,
            on_result=self._on_result,
            on_reasoning=self._on_reasoning,
            on_delta=self._on_delta,
            on_compact=self._on_compact,
            approve=self._approve,
        )

    # ── sessioni (persistenza) ────────────────────────────────────────────────

    def _session_state(self) -> dict:
        return {
            "last_agent": self.last_agent,
            "agents": {k: a.dump_state() for k, a in self.agents.items()},
        }

    def _save_session(self) -> None:
        if self.session_name:
            self.session.save(self.session_name, self._session_state())

    def _load_session(self, name: str) -> bool:
        state = self.session.load(name)
        if not state:
            return False
        for key, agent in self.agents.items():
            data = (state.get("agents") or {}).get(key)
            if data:
                agent.load_state(data)
        self.last_agent = state.get("last_agent")
        self.session_name = name
        return True

    # ── callback UI ─────────────────────────────────────────────────────────

    def _newline_if_needed(self) -> None:
        if self._mid_line:
            self.console.file.write("\n")
            self.console.file.flush()
            self._mid_line = False

    def _on_delta(self, piece: str) -> None:
        sys.stdout.write(piece)
        sys.stdout.flush()
        self._mid_line = not piece.endswith("\n")

    def _on_tool(self, name: str, args: dict) -> None:
        self._newline_if_needed()
        icon = _TOOL_ICON.get(name, "🔧")
        shown = {}
        for k, v in args.items():
            if k in ("content", "new_string", "old_string", "text"):
                shown[k] = f"<{len(str(v))} char>"
            else:
                shown[k] = _short(v)
        argstr = "  ".join(f"[cyan]{k}[/cyan]={v}" for k, v in shown.items())
        self.console.print(f"  {icon} [bold]{name}[/bold]  {argstr}", highlight=False)
        self._turn_tools.append({"name": name, "args": {k: _short(v, 200) for k, v in args.items()}})

    def _on_result(self, name: str, output: str, ok: bool) -> None:
        self._newline_if_needed()
        first = output.splitlines()[0] if output else ""
        self.console.print(f"     [{'green' if ok else 'red'}]{_short(first, 100)}[/]", highlight=False)
        if self._turn_tools:
            self._turn_tools[-1].update(ok=ok, output=_short(output, 300))

    def _on_reasoning(self, text: str) -> None:
        self._newline_if_needed()
        self.console.print(Panel(Text(text.strip(), style="italic dim"),
                                 title="[dim]ragionamento[/dim]", border_style="dim", padding=(0, 1)))

    def _on_compact(self, before: int, after: int) -> None:
        self._newline_if_needed()
        self.console.print(f"[dim]  ⟳ contesto compattato: {before} → {after} messaggi[/dim]")

    # ── approvazione + anteprima diff ─────────────────────────────────────────

    @staticmethod
    def _sig(name: str, args: dict) -> str:
        key = args.get("command") or args.get("path") or args.get("name") or ""
        return f"{name}::{key}"

    def _approve(self, name: str, args: dict) -> bool:
        self._newline_if_needed()
        sig = self._sig(name, args)
        if sig in self._always_allow:
            return True

        preview = self._preview(name, args)
        if preview is not None:
            self.console.print(preview)
        else:
            target = args.get("command") or args.get("path") or args.get("name") or ""
            self.console.print(f"  [yellow]⚠ conferma[/yellow] [bold]{name}[/bold] → {_short(target, 80)}")

        try:
            ans = self.console.input("    procedo? [y]es / [n]o / [a]lways ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans in ("a", "always", "sempre"):
            self._always_allow.add(sig)
            return True
        return ans in ("y", "yes", "s", "si", "sì")

    def _preview(self, name: str, args: dict):
        """Anteprima dell'effetto per i tool distruttivi (diff per edit/write)."""
        try:
            if name == "write_file":
                p = fs.resolve(self.cfg.root, args.get("path", ""))
                old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                return self._diff_panel(name, fs.display(self.cfg.root, p), old, args.get("content", ""))
            if name == "edit_file":
                p = fs.resolve(self.cfg.root, args.get("path", ""))
                old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                try:
                    new, _ = fs.apply_edit(old, args.get("old_string", ""), args.get("new_string", ""),
                                           args.get("replace_all", False))
                except ToolError as exc:
                    return Panel(
                        Text(f"⚠ {exc}\nL'edit probabilmente fallirà (old_string non trovato o ambiguo).",
                             style="yellow"),
                        title=f"[yellow]{name}[/yellow] · {fs.display(self.cfg.root, p)}",
                        border_style="yellow", padding=(0, 1))
                return self._diff_panel(name, fs.display(self.cfg.root, p), old, new)
        except Exception:  # noqa: BLE001
            return None
        return None  # run_command e altri: nessuna diff

    def _diff_panel(self, name: str, path: str, old: str, new: str) -> Panel:
        diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                         lineterm="", n=2))[2:]  # salta header ---/+++
        body = Text()
        shown = 0
        for line in diff:
            if shown >= 60:
                body.append("…[diff troncata]\n", style="dim")
                break
            if line.startswith("+"):
                body.append(line + "\n", style="green")
            elif line.startswith("-"):
                body.append(line + "\n", style="red")
            elif line.startswith("@@"):
                body.append(line + "\n", style="cyan dim")
            else:
                body.append(line + "\n", style="dim")
            shown += 1
        if not diff:
            body.append("(nessuna differenza testuale rilevata)\n", style="dim")
        return Panel(body, title=f"[yellow]{name}[/yellow] · {path}", border_style="yellow", padding=(0, 1))

    # ── esecuzione ──────────────────────────────────────────────────────────

    def run_task(self, task: str, agent_key: str | None = None, think: bool = False) -> None:
        if agent_key is None:
            agent_key = router.classify(task, self.provider, self.last_agent)
        self.last_agent = agent_key
        agent = self.agents[agent_key]
        self._turn_tools = []
        self._mid_line = False

        self.console.print(f"[dim]→ agente: {agent_key}[/dim]")
        if self.cfg.stream:
            self.console.print(f"[bold cyan]flair · {agent_key}[/bold cyan]")
            result = agent.run(task, think=think)
            self._newline_if_needed()
            self.console.print()
        else:
            result = agent.run(task, think=think)
            self.console.print(Panel(
                Markdown(result.content or "(vuoto)"),
                title=f"[bold cyan]flair · {agent_key}[/bold cyan]",
                border_style="cyan", padding=(1, 2),
            ))

        if self.logger:
            self.logger.log_turn(agent_key, task, result, self._turn_tools)

        self._print_turn(result.usage, result.steps, result.stopped_reason)
        self._print_session()
        self._save_session()

    def _session_usage(self) -> Usage:
        total = Usage()
        for a in self.agents.values():
            total = total + a.total_usage
        return total

    def _cost_line(self, usage: Usage) -> str:
        cost = self.provider.estimate_cost(usage, self.cfg)
        denom = usage.cache_hit_tokens + usage.cache_miss_tokens
        cache_pct = round(100 * usage.cache_hit_tokens / denom) if denom else 0
        reasoning = f", reasoning {usage.reasoning_tokens}" if usage.reasoning_tokens else ""
        return (f"token {usage.total_tokens} (in {usage.prompt_tokens}, out {usage.completion_tokens}{reasoning}) "
                f"| cache hit {cache_pct}% | ~${cost:.4f}")

    def _print_turn(self, usage: Usage, steps: int, reason: str) -> None:
        flag = "" if reason == "done" else f" | [yellow]stop: {reason}[/yellow]"
        self.console.print(f"[dim]  questo turno · step {steps} · {self._cost_line(usage)}{flag}[/dim]")

    def _print_session(self) -> None:
        self.console.print(f"[dim]  sessione     · {self._cost_line(self._session_usage())}[/dim]")
        if self.last_agent:
            tokens, frac = self.agents[self.last_agent].context_fill()
            self.console.print(
                f"[dim]  contesto     · {self.last_agent}: {round(frac * 100)}% "
                f"({_kfmt(tokens)}/{_kfmt(self.cfg.context_window)})[/dim]")
        self._maybe_cost_warn()
        self.console.print()

    def _maybe_cost_warn(self) -> None:
        if self.cfg.cost_warn and not self._cost_warned:
            cost = self.provider.estimate_cost(self._session_usage(), self.cfg)
            if cost >= self.cfg.cost_warn:
                self.console.print(
                    f"[yellow]  ⚠ costo sessione ~${cost:.4f}: superata la soglia di "
                    f"${self.cfg.cost_warn:.2f} (FLAIR_COST_WARN)[/yellow]")
                self._cost_warned = True

    # ── REPL ──────────────────────────────────────────────────────────────────

    def repl(self) -> None:
        pc = self.cfg.active
        log_note = f"\nlog: {self.logger.path}" if self.logger else ""
        sess_note = f" | sessione: {self.session_name}" if self.session_name else ""
        self.console.print(Panel(
            Text.from_markup(
                f"[bold cyan]flair {__version__}[/bold cyan] [dim]— assistente AI (coding + generico)[/dim]\n"
                f"[dim]provider: {self.cfg.provider} | modello: {pc.model} | thinking: {pc.think_model}{sess_note}\n"
                f"root: {self.cfg.root}{log_note}[/dim]"
            ),
            border_style="cyan", padding=(1, 2),
        ))
        self.console.print("[dim]/help per i comandi. Scrivi una richiesta (coding o generica).[/dim]\n")

        while True:
            try:
                line = self.console.input("[bold green]▶[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]ciao![/dim]")
                return
            if not line:
                continue
            low = line.lower()

            if low in ("exit", "quit", "q"):
                self.console.print("[dim]ciao![/dim]")
                return
            if low == "/help":
                self.console.print(Markdown(__doc__ or ""))
                continue
            if low == "/reset":
                for a in self.agents.values():
                    a.reset()
                self.last_agent = None
                self.console.print("[yellow]conversazioni azzerate.[/yellow]\n")
                continue
            if low == "/cost":
                self.console.print(f"[dim]  sessione · {self._cost_line(self._session_usage())}[/dim]\n")
                continue
            if low == "/agent":
                self.console.print(f"[dim]agente corrente (sticky): {self.last_agent or 'nessuno'}[/dim]\n")
                continue
            if low == "/sessions":
                items = self.session.list()
                if not items:
                    self.console.print("[dim]nessuna sessione salvata.[/dim]\n")
                else:
                    body = "\n".join(f"  • {n}  [dim]{ts}[/dim]" for n, ts in items)
                    self.console.print(f"[dim]sessioni salvate:[/dim]\n{body}\n")
                continue
            if low.startswith("/save"):
                parts = line.split(maxsplit=1)
                name = parts[1].strip() if len(parts) == 2 else (self.session_name or "default")
                self.session_name = name
                path = self.session.save(name, self._session_state())
                msg = f"[green]sessione salvata: {name}[/green]" if path else "[red]salvataggio fallito (vedi log).[/red]"
                self.console.print(msg + "\n")
                continue
            if low.startswith("/load"):
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    self.console.print("[dim]uso: /load <nome>[/dim]\n")
                elif self._load_session(parts[1].strip()):
                    self.console.print(f"[green]sessione ripresa: {self.session_name}[/green]\n")
                else:
                    self.console.print(f"[yellow]sessione '{parts[1].strip()}' non trovata.[/yellow]\n")
                continue
            if low == "/compact":
                if self.last_agent:
                    if not self.agents[self.last_agent].compact():
                        self.console.print("[dim]niente da compattare.[/dim]\n")
                else:
                    self.console.print("[dim]nessuna conversazione attiva.[/dim]\n")
                continue
            if low.startswith("/provider"):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    target = parts[1].strip().lower()
                    if target not in ("deepseek", "openai"):
                        self.console.print("[yellow]provider non valido (deepseek|openai).[/yellow]\n")
                    else:
                        self.cfg.provider = target
                        self.cfg.refresh_pricing()
                        self.provider = create_provider(self.cfg)
                        for a in self.agents.values():
                            a.provider = self.provider
                        pc = self.cfg.active
                        self.console.print(f"[yellow]provider → {target} | modello: {pc.model} | thinking: {pc.think_model}[/yellow]\n")
                else:
                    pc = self.cfg.active
                    self.console.print(f"[dim]provider: {self.cfg.provider} | modello: {pc.model} | thinking: {pc.think_model}[/dim]\n")
                continue
            if low.startswith("/think-model"):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    self.cfg.active.think_model = parts[1].strip()
                    self.console.print(f"[yellow]modello thinking → {self.cfg.active.think_model}[/yellow]\n")
                else:
                    self.console.print("[dim]uso: /think-model <nome>[/dim]\n")
                continue
            if low.startswith("/model"):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    self.cfg.active.model = parts[1].strip()
                    self.cfg.refresh_pricing()
                    self.console.print(f"[yellow]modello → {self.cfg.active.model}[/yellow]\n")
                else:
                    self.console.print("[dim]uso: /model <nome>[/dim]\n")
                continue
            if low.startswith("/root"):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    self.cfg.root = Path(parts[1]).expanduser().resolve()
                    # ricostruisce il coding agent per ricaricare le istruzioni di progetto
                    self.agents["coding"] = coding_agent.build(self.cfg, self.provider, **self._callbacks())
                    self.console.print(f"[yellow]root → {self.cfg.root}[/yellow]\n")
                continue
            if low.startswith("/code"):
                task = line[len("/code"):].strip()
                if task:
                    self.run_task(task, agent_key="coding")
                continue
            if low.startswith("/do"):
                task = line[len("/do"):].strip()
                if task:
                    self.run_task(task, agent_key="general")
                continue
            if low.startswith("/think"):
                task = line[len("/think"):].strip()
                if task:
                    self.run_task(task, think=True)
                continue

            self.run_task(line)


def _build_config(args) -> Config:
    cfg = load_config()
    if args.provider:
        cfg.provider = args.provider
        cfg.refresh_pricing()
    if args.root:
        cfg.root = Path(args.root).expanduser().resolve()
    if args.yes:
        cfg.auto_approve = True
    if args.no_stream:
        cfg.stream = False
    if args.log:
        cfg.log_dir = Path(args.log).expanduser()
    if args.model:
        cfg.active.model = args.model
        cfg.refresh_pricing()
    if args.think_model:
        cfg.active.think_model = args.think_model
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flair", description="Assistente AI agentico (coding + generico) su DeepSeek/OpenAI.")
    ap.add_argument("--version", action="version", version=f"flair {__version__}")
    ap.add_argument("-p", "--prompt", help="esegue un singolo task e esce")
    ap.add_argument("--provider", choices=["deepseek", "openai"], help="provider LLM")
    ap.add_argument("--agent", choices=["coding", "general", "auto"], default="auto", help="forza un agente (default: auto)")
    ap.add_argument("--root", help="radice di lavoro per l'agente coding")
    ap.add_argument("--think", action="store_true", help="usa il modello thinking al primo passo")
    ap.add_argument("--yes", action="store_true", help="auto-approva i tool distruttivi")
    ap.add_argument("--no-stream", dest="no_stream", action="store_true", help="disabilita lo streaming")
    ap.add_argument("--log", help="cartella in cui scrivere il log di sessione (JSONL)")
    ap.add_argument("--model", help="override del modello veloce")
    ap.add_argument("--think-model", dest="think_model", help="override del modello thinking")
    ap.add_argument("--session", help="usa/crea una sessione con questo nome (autosalvataggio)")
    ap.add_argument("--continue", dest="continue_", action="store_true", help="riprende l'ultima sessione salvata")
    args = ap.parse_args(argv)

    console = Console()
    cfg = _build_config(args)
    try:
        cfg.validate()
    except RuntimeError as exc:
        console.print(f"[bold red]Configurazione non valida:[/bold red] {exc}")
        return 1

    cli = CLI(cfg)

    # Ripresa sessione (prima di eseguire qualsiasi cosa).
    if args.session:
        cli.session_name = args.session
        if cli._load_session(args.session):
            console.print(f"[dim]sessione ripresa: {args.session}[/dim]")
        else:
            console.print(f"[dim]nuova sessione: {args.session}[/dim]")
    elif args.continue_:
        latest = cli.session.latest()
        if latest and cli._load_session(latest):
            console.print(f"[dim]ripresa ultima sessione: {latest}[/dim]")
        else:
            console.print("[dim]nessuna sessione da riprendere.[/dim]")

    if args.prompt:
        key = None if args.agent == "auto" else args.agent
        cli.run_task(args.prompt, agent_key=key, think=args.think)
        return 0
    cli.repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
