"""CLI interattiva (e batch) per flair.

`flair` apre il REPL (instrada da solo tra agente coding e generico); `flair -p "..."`
esegue un singolo task ed esce. I flag di avvio sono documentati da `flair -h`; i
comandi disponibili nel REPL da `/help`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .agents import coding as coding_agent
from .agents import general as general_agent
from .config import Config, load_config
from .core import router
from .core.agent import Conversation
from .core.tool import ToolError
from .llm import Usage, create_provider
from .session_log import SessionLogger, setup_file_logging
from .session_store import SessionStore
from .tools import fs

_TOOL_ICON = {
    "read_file": "📄", "list_directory": "📁", "glob": "🔎", "grep": "🔎",
    "repo_map": "🗺️ ", "explore": "🔭", "plan": "📋",
    "edit_file": "✏️ ", "multi_edit": "✏️ ", "write_file": "📝", "run_command": "⚙️ ",
    "run_powershell": "⚙️ ",
    "open_url": "🌐", "open_path": "📂", "open_application": "🚀",
    "search_files": "🔦", "system_info": "🖥️ ", "get_datetime": "🕒",
    "clipboard_get": "📋", "clipboard_set": "📋", "web_search": "🌍", "web_fetch": "🌍",
}


def _short(v, n: int = 70) -> str:
    s = str(v).replace("\n", "↵")
    return s if len(s) <= n else s[: n - 1] + "…"


def _kfmt(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


# Exit code per l'uso non presidiato (cron/CI): distinguono ESITO, non solo ok/ko,
# così uno scheduler può ramificare. 0 = completato; gli altri = motivi di stop.
EXIT_CODES = {"done": 0, "max_steps": 2, "loop": 3, "stopped": 4, "budget": 5}


def exit_code_for(reason: str) -> int:
    """Mappa stopped_reason → exit code. Sconosciuto/errore → 1."""
    return EXIT_CODES.get(reason, 1)


_FILE_WRITE_TOOLS = {"write_file", "edit_file", "multi_edit"}


def build_result_json(agent_key: str | None, task: str, result, tool_events: list[dict],
                      cost_usd: float) -> dict:
    """Oggetto machine-readable di un turno one-shot (modalità --json). Puro e
    serializzabile: riassume esito, risposta, passi, usage/costo, tool e file toccati."""
    u = result.usage
    files: list[str] = []
    seen: set[str] = set()
    for ev in tool_events:
        if ev.get("ok") and ev.get("name") in _FILE_WRITE_TOOLS:
            p = (ev.get("args") or {}).get("path")
            if p and p not in seen:
                seen.add(p)
                files.append(p)
    return {
        "ok": result.stopped_reason == "done",
        "agent": agent_key,
        "stopped_reason": result.stopped_reason,
        "response": result.content or "",
        "steps": result.steps,
        "truncated": result.truncated,
        "usage": {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
            "cache_hit_tokens": u.cache_hit_tokens,
            "cache_miss_tokens": u.cache_miss_tokens,
            "reasoning_tokens": u.reasoning_tokens,
        },
        "cost_usd": round(cost_usd, 6),
        "tools": [{"name": e.get("name"), "ok": bool(e.get("ok"))} for e in tool_events],
        "files_changed": files,
    }


class CLI:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.console = Console()
        # Allinea subito la directory di processo a cfg.root: la root vale così anche
        # per l'agente general (vedi _chdir_root). Allo startup root == cwd se non è
        # stata passata, quindi di norma è un no-op.
        self._chdir_root()
        self.provider = create_provider(cfg)
        self.last_agent: str | None = None
        self._mid_line = False
        self._turn_tools: list[dict] = []
        self._always_allow: set[str] = set()
        self._cost_warned = False
        # "human" (REPL/default), "json" o "quiet": le ultime due sono per l'uso
        # non presidiato (-p) e silenziano l'output decorato su stdout.
        self.output_mode = "human"

        self.session = SessionStore(cfg.session_dir) if cfg.session_dir else SessionStore(Path.home() / ".flair" / "sessions")
        self.session_name: str | None = None

        self.logger: SessionLogger | None = None
        if cfg.log_dir:
            setup_file_logging(cfg.log_dir)
            self.logger = SessionLogger(cfg.log_dir)

        # Memoria CONDIVISA: i due agenti ragionano sulla stessa conversazione, così
        # un cambio di agente (anche se il router sbaglia) non perde il contesto.
        self.convo = Conversation()
        self.agents = {
            "coding": coding_agent.build(cfg, self.provider, conversation=self.convo, **self._callbacks()),
            "general": general_agent.build(cfg, self.provider, conversation=self.convo, **self._callbacks()),
        }

    def _callbacks(self) -> dict:
        return dict(
            on_tool=self._on_tool,
            on_result=self._on_result,
            on_reasoning=self._on_reasoning,
            on_delta=self._on_delta,
            on_compact=self._on_compact,
            on_prune=self._on_prune,
            approve=self._approve,
        )

    def _chdir_root(self) -> None:
        """Allinea la directory di processo a cfg.root. La modalità coding usa già
        cwd=root nei comandi; questo allineamento estende la stessa cartella di lavoro
        anche all'agente general (che resta SENZA confinamento, ma eredita la cwd):
        così «crea un report nella cartella attuale» è coerente con la root impostata."""
        try:
            os.chdir(self.cfg.root)
        except OSError as exc:
            self.console.print(f"[yellow]⚠ non riesco a spostarmi in {self.cfg.root}: {exc}[/yellow]")

    def _apply_root(self, new_root: Path) -> None:
        """Cambia la root a runtime (comando /root): aggiorna cfg.root, allinea la
        directory di processo e ricostruisce il coding agent per ricaricare le
        istruzioni di progetto, preservando la memoria condivisa."""
        self.cfg.root = new_root
        self._chdir_root()
        self.agents["coding"] = coding_agent.build(self.cfg, self.provider, conversation=self.convo, **self._callbacks())

    # ── sessioni (persistenza) ────────────────────────────────────────────────

    def _session_state(self) -> dict:
        return {
            "last_agent": self.last_agent,
            "conversation": self.convo.dump(),
        }

    def _save_session(self) -> None:
        if self.session_name:
            self.session.save(self.session_name, self._session_state())

    def _load_session(self, name: str) -> bool:
        state = self.session.load(name)
        if not state:
            return False
        convo_state = state.get("conversation")
        if convo_state is None:
            # Retro-compatibilità: i salvataggi vecchi tenevano una storia per agente.
            # Recuperiamo quella più sostanziosa come conversazione condivisa.
            agents = state.get("agents") or {}
            best = max((a for a in agents.values()),
                       key=lambda a: len(a.get("messages") or []), default=None)
            convo_state = best or {}
        self.convo.load(convo_state)
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
        if self.output_mode != "human":
            return
        sys.stdout.write(piece)
        sys.stdout.flush()
        self._mid_line = not piece.endswith("\n")

    def _on_tool(self, name: str, args: dict) -> None:
        # Raccogliamo sempre l'evento (serve a logger e a --json); stampiamo solo in human.
        self._turn_tools.append({"name": name, "args": {k: _short(v, 200) for k, v in args.items()}})
        if self.output_mode != "human":
            return
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

    def _on_result(self, name: str, output: str, ok: bool) -> None:
        if self._turn_tools:
            self._turn_tools[-1].update(ok=ok, output=_short(output, 300))
        if self.output_mode != "human":
            return
        self._newline_if_needed()
        if name == "plan" and ok:
            # La scaletta è l'output più utile da mostrare per intero (è corta).
            for line in output.splitlines():
                self.console.print(f"     [cyan]{line}[/cyan]", highlight=False)
        else:
            first = output.splitlines()[0] if output else ""
            self.console.print(f"     [{'green' if ok else 'red'}]{_short(first, 100)}[/]", highlight=False)

    def _on_prune(self, count: int) -> None:
        if self.output_mode != "human":
            return
        self._newline_if_needed()
        self.console.print(f"[dim]  ✂ contesto: potati {count} output di tool superati[/dim]")

    def _on_reasoning(self, text: str) -> None:
        if self.output_mode != "human":
            return
        self._newline_if_needed()
        self.console.print(Panel(Text(text.strip(), style="italic dim"),
                                 title="[dim]ragionamento[/dim]", border_style="dim", padding=(0, 1)))

    def _on_compact(self, before: int, after: int) -> None:
        if self.output_mode != "human":
            return
        self._newline_if_needed()
        self.console.print(f"[dim]  ⟳ contesto compattato: {before} → {after} messaggi[/dim]")

    # ── approvazione + anteprima diff ─────────────────────────────────────────

    def _approve(self, name: str, args: dict) -> bool | str:
        self._newline_if_needed()
        if name in self._always_allow:   # "always" vale per l'intero tool, per la sessione
            return True

        preview = self._preview(name, args)
        if preview is not None:
            self.console.print(preview)
        else:
            target = args.get("command") or args.get("path") or args.get("name") or args.get("script") or ""
            self.console.print(f"  [yellow]⚠ conferma[/yellow] [bold]{name}[/bold] → {_short(target, 80)}")

        try:
            # Le parentesi quadre sono escape-ate: Rich le interpreterebbe come markup.
            ans = self.console.input(r"    procedo? \[y]es / \[n]o / \[a]lways / \[s]top ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return "stop"   # Ctrl-C/EOF al prompt = ferma il flusso
        if ans in ("s", "stop"):
            return "stop"
        if ans in ("a", "always", "sempre"):
            self._always_allow.add(name)
            self.console.print(f"[dim]  ok: non chiederò più conferma per «{name}» in questa sessione.[/dim]")
            return True
        # NB: 's' è riservato a stop; per il sì in italiano si usa 'si'/'sì'.
        return ans in ("y", "yes", "si", "sì")

    def _preview(self, name: str, args: dict):
        """Anteprima dell'effetto per i tool distruttivi (diff per edit/write)."""
        try:
            if name == "write_file":
                p = fs.resolve(self.cfg.root, args.get("path", ""))
                old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                content = args.get("content", "")
                # In append l'effetto è old + content: mostra solo le aggiunte.
                new = old + content if fs.as_bool(args.get("append", False)) else content
                return self._diff_panel(name, fs.display(self.cfg.root, p), old, new)
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

    def _safe_run_task(self, task: str, agent_key: str | None = None, think: bool = False) -> None:
        """Esegue un turno proteggendo la REPL. Ctrl-C interrompe il turno e riporta al
        prompt; un errore (es. timeout di rete del modello esaurita la coda di retry)
        viene segnalato senza far crashare flair. La conversazione resta utilizzabile."""
        try:
            self.run_task(task, agent_key=agent_key, think=think)
        except KeyboardInterrupt:
            self._newline_if_needed()
            self.console.print("[yellow]⏹ Turno interrotto. Sei tornato al prompt.[/yellow]\n")
        except Exception as exc:  # noqa: BLE001
            self._newline_if_needed()
            self.console.print(f"[red]⚠ Il turno è fallito: {type(exc).__name__}: {exc}[/red]")
            self.console.print("[dim]Puoi riprovare. Se è un timeout di rete del modello, "
                               "riprova tra poco o abbassa FLAIR_TIMEOUT.[/dim]\n")

    def _emit_json(self, obj: dict) -> None:
        # Una sola riga JSON su stdout (JSONL-friendly), nient'altro in modalità json.
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def run_once(self, task: str, agent_key: str | None = None, think: bool = False) -> int:
        """Esegue un singolo task (modalità `-p`) e ritorna un exit code per gli script:
        0 done, 2 max-step, 3 loop, 4 fermato (serviva approvazione o stop), 5 budget,
        1 errore, 130 interruzione. In modalità --json emette SEMPRE un oggetto su stdout
        (anche su errore/interruzione), così il contratto resta affidabile per l'automazione."""
        try:
            result = self.run_task(task, agent_key=agent_key, think=think)
        except KeyboardInterrupt:
            self._newline_if_needed()
            if self.output_mode == "json":
                self._emit_json({"ok": False, "agent": self.last_agent, "stopped_reason": "interrupted",
                                 "response": "", "error": "interrotto"})
            elif self.output_mode == "human":
                self.console.print("[yellow]⏹ Interrotto.[/yellow]")
            return 130
        except Exception as exc:  # noqa: BLE001
            self._newline_if_needed()
            if self.output_mode == "json":
                self._emit_json({"ok": False, "agent": self.last_agent, "stopped_reason": "error",
                                 "response": "", "error": f"{type(exc).__name__}: {exc}"})
            elif self.output_mode == "human":
                self.console.print(f"[red]⚠ Errore: {type(exc).__name__}: {exc}[/red]")
            return 1

        if self.output_mode == "json":
            cost = self.provider.estimate_cost(result.usage, self.cfg)
            self._emit_json(build_result_json(self.last_agent, task, result, self._turn_tools, cost))
        elif self.output_mode == "quiet":
            sys.stdout.write((result.content or "") + "\n")
            sys.stdout.flush()
        return exit_code_for(result.stopped_reason)

    def run_task(self, task: str, agent_key: str | None = None, think: bool = False):
        if agent_key is None:
            agent_key = router.classify(task, self.provider, self.last_agent, convo=self.convo)
        self.last_agent = agent_key
        agent = self.agents[agent_key]
        self._turn_tools = []
        self._mid_line = False
        human = self.output_mode == "human"

        if human:
            self.console.print(f"[dim]→ agente: {agent_key}[/dim]")
        if human and self.cfg.stream:
            self.console.print(f"[bold cyan]flair · {agent_key}[/bold cyan]")
            result = agent.run(task, think=think)
            self._newline_if_needed()
            self.console.print()
        else:
            result = agent.run(task, think=think)
            if human and result.stopped_reason not in ("stopped", "budget"):
                self.console.print(Panel(
                    Markdown(result.content or "(vuoto)"),
                    title=f"[bold cyan]flair · {agent_key}[/bold cyan]",
                    border_style="cyan", padding=(1, 2),
                ))

        if human and result.stopped_reason == "stopped":
            self.console.print("[yellow]⏹ Flusso interrotto: hai ripreso il controllo. Dimmi come procedere.[/yellow]\n")
        if human and result.stopped_reason == "budget":
            self.console.print("[yellow]⏹ Interrotto: raggiunto il tetto di costo "
                               "(--max-cost / FLAIR_MAX_COST).[/yellow]\n")
        if human and result.truncated:
            if (result.content or "").strip():
                self.console.print("[yellow]⚠ Risposta troncata: raggiunto il limite di token in output. "
                                   "Chiedi di continuare, o aumenta FLAIR_MAX_TOKENS.[/yellow]")
            else:
                # Tutto il budget è finito nel ragionamento, prima di produrre una risposta:
                # "continuare" non aiuta (il ragionamento non si riporta). Indica il fix vero.
                self.console.print("[yellow]⚠ Nessuna risposta: il budget di output (FLAIR_MAX_TOKENS) "
                                   "si è esaurito durante il ragionamento. Con un modello 'thinking' "
                                   "serve molto più di 8000: alza FLAIR_MAX_TOKENS, oppure usa il modello "
                                   "veloce (senza --think) per il lavoro coi tool.[/yellow]")

        if self.logger:
            self.logger.log_turn(agent_key, task, result, self._turn_tools)

        if human:
            self._print_turn(result.usage, result.steps, result.stopped_reason)
            self._print_session()
        self._save_session()
        return result

    def _session_usage(self) -> Usage:
        return self.convo.total_usage

    def _cost_line(self, usage: Usage) -> str:
        cost = self.provider.estimate_cost(usage, self.cfg)
        denom = usage.cache_hit_tokens + usage.cache_miss_tokens
        cache_pct = round(100 * usage.cache_hit_tokens / denom) if denom else 0
        reasoning = f", reasoning {usage.reasoning_tokens}" if usage.reasoning_tokens else ""
        return (f"token {usage.total_tokens} (in {usage.prompt_tokens}, out {usage.completion_tokens}{reasoning}) "
                f"| cache hit {cache_pct}% | ~${cost:.4f}")

    def _print_turn(self, usage: Usage, steps: int, reason: str) -> None:
        labels = {"max_steps": "max step", "loop": "loop rilevato", "stopped": "interrotto", "budget": "budget"}
        flag = f" | [yellow]{labels[reason]}[/yellow]" if reason in labels else ""
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

    def _print_help(self) -> None:
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan",
                      padding=(0, 3, 0, 0), border_style="dim")
        table.add_column("comando", style="bold", no_wrap=True)
        table.add_column("cosa fa", style="dim")
        for cmd, desc in (
            ("/code <task>", "forza l'agente di coding"),
            ("/do <task>", "forza l'agente generico"),
            ("/think <task>", "primo passo col modello thinking"),
            ("/agent", "mostra l'agente corrente (sticky)"),
            ("/tools", "elenca i tool dell'agente attivo"),
            ("/provider [nome]", "mostra o cambia provider (deepseek|openai)"),
            ("/model <nome>", "cambia il modello veloce a runtime"),
            ("/think-model <nome>", "cambia il modello thinking a runtime"),
            ("/compact", "compatta subito il contesto dell'agente attivo"),
            ("/cost", "riepilogo token/costo della sessione"),
            ("/save [nome]", "salva la sessione (default: nome corrente)"),
            ("/load <nome>", "riprende una sessione salvata"),
            ("/sessions", "elenca le sessioni salvate"),
            ("/reset", "azzera la conversazione condivisa"),
            ("/root <path>", "cambia la cartella di lavoro (coding + general; ricarica le istruzioni)"),
            ("/help", "questo aiuto"),
            ("exit | quit", "esci"),
        ):
            table.add_row(Text(cmd), desc)
        self.console.print(table)
        self.console.print(
            "[dim]Flag di avvio (CLI): «flair -h». Esempi: flair --think -p \"...\", "
            "flair --session lavoro, flair --continue, flair --provider openai.[/dim]\n")

    def _print_tools(self) -> None:
        key = self.last_agent or "general"
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan",
                      padding=(0, 3, 0, 0), border_style="dim")
        table.add_column("tool", style="bold", no_wrap=True)
        table.add_column("cosa fa", style="dim")
        for name, desc in self.agents[key].toolset.catalog():
            icon = _TOOL_ICON.get(name, "🔧")
            line = " ".join(desc.split())                  # normalizza gli spazi
            if len(line) > 100:
                line = line[:99] + "…"
            table.add_row(Text(f"{icon} {name}"), line)
        self.console.print(table)
        self.console.print(
            f"[dim]Tool dell'agente «{key}»"
            f"{' (sticky)' if self.last_agent else ' (default; nessun turno ancora)'}. "
            "Cambia agente con /code, /do.[/dim]\n")

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
                self._print_help()
                continue
            if low == "/tools":
                self._print_tools()
                continue
            if low == "/reset":
                self.convo.reset()
                self.last_agent = None
                self.console.print("[yellow]conversazione azzerata.[/yellow]\n")
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
                    new_root = Path(parts[1]).expanduser().resolve()
                    if not new_root.is_dir():
                        self.console.print(f"[yellow]cartella inesistente: {new_root}[/yellow]\n")
                    else:
                        self._apply_root(new_root)
                        self.console.print(
                            f"[yellow]root → {self.cfg.root} "
                            "(cartella di lavoro per coding e general)[/yellow]\n")
                continue
            if low.startswith("/code"):
                task = line[len("/code"):].strip()
                if task:
                    self._safe_run_task(task, agent_key="coding")
                continue
            if low.startswith("/do"):
                task = line[len("/do"):].strip()
                if task:
                    self._safe_run_task(task, agent_key="general")
                continue
            if low.startswith("/think"):
                task = line[len("/think"):].strip()
                if task:
                    self._safe_run_task(task, think=True)
                continue

            self._safe_run_task(line)


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
        cfg.log_dir = Path(args.log).expanduser().resolve()
    if args.model:
        cfg.active.model = args.model
        cfg.refresh_pricing()
    if args.think_model:
        cfg.active.think_model = args.think_model
    if args.read_only:
        cfg.read_only = True
    if args.max_cost is not None:
        cfg.max_cost = args.max_cost
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flair", description="Assistente AI agentico (coding + generico) su DeepSeek/OpenAI.")
    ap.add_argument("--version", action="version", version=f"flair {__version__}")
    ap.add_argument("-p", "--prompt", help="esegue un singolo task e esce (usa '-' per leggere da stdin)")
    ap.add_argument("--provider", choices=["deepseek", "openai"], help="provider LLM")
    ap.add_argument("--agent", choices=["coding", "general", "auto"], default="auto", help="forza un agente (default: auto)")
    ap.add_argument("--root", help="radice di lavoro per l'agente coding")
    ap.add_argument("--think", action="store_true", help="usa il modello thinking al primo passo")
    ap.add_argument("--yes", action="store_true", help="auto-approva i tool distruttivi")
    ap.add_argument("--read-only", dest="read_only", action="store_true",
                    help="esecuzione non presidiata: disabilita i tool distruttivi (write/edit/comandi)")
    ap.add_argument("--max-cost", dest="max_cost", type=float, default=None,
                    help="tetto HARD di costo della sessione in USD: oltre, il task si ferma")
    ap.add_argument("--json", action="store_true", help="con -p: emette un oggetto JSON (per automazioni)")
    ap.add_argument("-q", "--quiet", action="store_true", help="con -p: stampa solo la risposta finale")
    ap.add_argument("--no-stream", dest="no_stream", action="store_true", help="disabilita lo streaming")
    ap.add_argument("--log", help="cartella in cui scrivere il log di sessione (JSONL)")
    ap.add_argument("--model", help="override del modello veloce")
    ap.add_argument("--think-model", dest="think_model", help="override del modello thinking")
    ap.add_argument("--session", help="usa/crea una sessione con questo nome (autosalvataggio)")
    ap.add_argument("--continue", dest="continue_", action="store_true", help="riprende l'ultima sessione salvata")
    args = ap.parse_args(argv)

    # Modalità di output one-shot: json/quiet valgono solo con -p (la REPL resta human).
    json_mode = bool(args.prompt is not None and args.json)
    quiet_mode = bool(args.prompt is not None and args.quiet and not args.json)
    headless = json_mode or quiet_mode

    console = Console(stderr=headless)  # in headless i messaggi umani vanno su stderr
    cfg = _build_config(args)
    if headless:
        cfg.stream = False              # niente delta su stdout: resta pulito per la macchina
    try:
        cfg.validate()
    except RuntimeError as exc:
        if json_mode:
            sys.stdout.write(json.dumps(
                {"ok": False, "stopped_reason": "config_error", "response": "", "error": str(exc)}) + "\n")
        else:
            console.print(f"[bold red]Configurazione non valida:[/bold red] {exc}")
        return 1

    cli = CLI(cfg)
    if json_mode:
        cli.output_mode = "json"
    elif quiet_mode:
        cli.output_mode = "quiet"

    # Ripresa sessione (prima di eseguire qualsiasi cosa). In headless i messaggi
    # informativi vanno su stderr, così stdout resta riservato all'output macchina.
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

    if args.prompt is not None:
        prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
        prompt = prompt.strip()
        if not prompt:
            if json_mode:
                cli._emit_json({"ok": False, "agent": None, "stopped_reason": "error",
                                "response": "", "error": "prompt vuoto"})
            else:
                console.print("[red]Prompt vuoto.[/red]")
            return 1
        key = None if args.agent == "auto" else args.agent
        return cli.run_once(prompt, agent_key=key, think=args.think)
    cli.repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
