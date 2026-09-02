"""CLI interattiva (e batch) per flair.

`flair` apre il REPL (instrada da solo tra agente coding e generico); `flair -p "..."`
esegue un singolo task ed esce. I flag di avvio sono documentati da `flair -h`; i
comandi disponibili nel REPL da `/help`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from . import __version__
from .agents import coding as coding_agent
from .agents import general as general_agent
from .config import Config, load_config
from .core import router
from .core.agent import _SUMMARY_HEADER, Approval, Conversation, content_text
from .core.tool import ToolError
from .llm import Usage, create_provider
from .memory import SessionMemory
from .session_log import SessionLogger, setup_file_logging
from .session_store import SessionStore
from .tools import fs, images
from .tools.jobs import BackgroundJobs

_TOOL_ICON = {
    "read_file": "📄", "list_directory": "📁", "glob": "🔎", "grep": "🔎",
    "repo_map": "🗺️ ", "explore": "🔭", "plan": "📋",
    "edit_file": "✏️ ", "multi_edit": "✏️ ", "write_file": "📝", "run_command": "⚙️ ",
    "run_powershell": "⚙️ ",
    "open_url": "🌐", "open_path": "📂", "open_application": "🚀",
    "search_files": "🔦", "system_info": "🖥️ ", "get_datetime": "🕒",
    "clipboard_get": "📋", "clipboard_set": "📋", "web_search": "🌍", "web_fetch": "🌍",
}


def _pick_spinner(windows: bool, env: dict, encoding: str | None) -> str:
    """Sceglie lo spinner in base alle capacità note del terminale: "dots" (Braille,
    elegante) solo dove i glifi sono affidabili; altrimenti "line" (ASCII puro).
    Su Windows il conhost classico (cmd.exe) spesso non ha i Braille nel font:
    servono segnali espliciti di un terminale moderno. Su POSIX basta l'encoding."""
    if windows:
        modern = ("WT_SESSION" in env or env.get("ConEmuANSI") == "ON"
                  or env.get("TERM_PROGRAM") == "vscode" or "ANSICON" in env)
        return "dots" if modern else "line"
    return "dots" if encoding and "utf" in encoding.lower() else "line"


def _spinner_name() -> str:
    return _pick_spinner(os.name == "nt", dict(os.environ), getattr(sys.stdout, "encoding", None))


def _fmt_thinking(chars: int, secs: float) -> str:
    volume = f"{chars / 1000:.1f}k" if chars >= 1000 else str(chars)
    return f"{volume} chars · {int(secs)}s"


def _recap_messages(messages: list[dict], max_msgs: int = 4,
                    max_chars: int = 400) -> tuple[list[tuple[str, str]], int]:
    """Coda del dialogo per il recap post-load: solo turni user/assistant con del
    testo (niente messaggi tool né tool-call vuote), ultimi `max_msgs`, ciascuno
    compattato su una riga e troncato. Il riassunto di compaction (iniettato come
    messaggio user) viene etichettato per quello che è. Ritorna (righe, quanti
    messaggi di dialogo precedenti non vengono mostrati)."""
    dialog: list[tuple[str, str]] = []
    header = _SUMMARY_HEADER.strip()
    for m in messages:
        role = str(m.get("role") or "")
        text = content_text(m.get("content")).strip()
        if role not in ("user", "assistant") or not text:
            continue
        if role == "user" and text.startswith(header):
            role = "summary"
            text = text[len(header):].strip()
        dialog.append((role, text))
    tail = dialog[-max_msgs:] if max_msgs > 0 else []
    out: list[tuple[str, str]] = []
    for role, text in tail:
        t = " ".join(text.split())
        if len(t) > max_chars:
            t = t[: max_chars - 1] + "…"
        out.append((role, t))
    return out, len(dialog) - len(tail)


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


# Comandi del REPL: (nome, come si scrive nell'help, cosa fa, metodo che lo esegue).
# UNICA fonte: da qui nascono sia il dispatch sia la tabella di /help, quindi un
# comando non documentato o una voce di help senza handler sono impossibili (c'è un
# test che lo verifica). Il dispatch avviene sul PRIMO TOKEN esatto: prima si
# confrontavano i prefissi con startswith, e "/documenta il codice" finiva
# all'handler di "/do" con il task mutilato in "cumenta il codice".
_COMMANDS: tuple[tuple[str, str, str, str], ...] = (
    ("code", "/code <task>", "force the coding agent", "_cmd_code"),
    ("do", "/do <task>", "force the general agent", "_cmd_do"),
    ("think", "/think <task>", "first step with the thinking model", "_cmd_think"),
    ("agent", "/agent", "show the current (sticky) agent", "_cmd_agent"),
    ("tools", "/tools", "list the active agent's tools", "_cmd_tools"),
    ("provider", "/provider [name]", "show or switch provider (deepseek|openai)", "_cmd_provider"),
    ("model", "/model <name>", "switch the fast model at runtime", "_cmd_model"),
    ("think-model", "/think-model <name>", "switch the thinking model at runtime", "_cmd_think_model"),
    ("compact", "/compact", "compact the active agent's context now", "_cmd_compact"),
    ("cost", "/cost", "token/cost summary for the session", "_cmd_cost"),
    ("save", "/save [name]", "save the session (default: current name)", "_cmd_save"),
    ("load", "/load <name>", "resume a saved session", "_cmd_load"),
    ("sessions", "/sessions", "list saved sessions", "_cmd_sessions"),
    ("memory", "/memory [clear]", "show (or clear) the session memory", "_cmd_memory"),
    ("remember", "/remember <note>", "jot a durable note into session memory yourself", "_cmd_remember"),
    ("reset", "/reset", "reset the shared conversation", "_cmd_reset"),
    ("root", "/root <path>", "change the working folder (coding + general; reloads instructions)", "_cmd_root"),
    ("img", "/img <path> [prompt]", "attach an image to the turn (vision endpoints only)", "_cmd_img"),
    ("jobs", "/jobs [stop <id>]", "background jobs still running (or stop one)", "_cmd_jobs"),
    ("help", "/help", "this help", "_cmd_help"),
)
_QUIT_WORDS = ("exit", "quit", "q")


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
        # Memoria di SESSIONE: fatti durevoli, ancorati alla sessione (non alla root).
        # Condivisa per riferimento dai due agenti via ToolContext (tool `remember`);
        # iniettata nel system prompt SOLO ai confini di sessione (qui, /load,
        # /memory clear) così il prefisso in cache non si rompe mai a metà lavoro.
        self.memory = SessionMemory(max_chars=cfg.memory_max_chars)
        # --think in modalità interattiva: default di sessione per ogni turno.
        # Prima veniva letto solo nel one-shot -p e nel REPL non faceva NULLA,
        # in silenzio — l'ambiguità peggiore. /think resta il rinforzo per turno.
        self.default_think = False
        # Status vivo durante le fasi di pensiero (spinner + contatore): parte al
        # primo delta di reasoning, si ferma PRIMA di qualunque altra stampa.
        self._think_status: Status | None = None
        self._think_chars = 0
        self._think_t0 = 0.0
        self._base_prompts = {k: ag.system_prompt for k, ag in self.agents.items()}
        # Registro dei job in background: uno per sessione, condiviso per riferimento
        # con entrambi gli agenti (e con i worker paralleli, che ricevono un ctx
        # isolato ma la stessa istanza). I processi vengono terminati su OGNI uscita
        # (v. _shutdown_jobs): un comando che sopravvive a flair è un bug, non una
        # feature.
        self.jobs = BackgroundJobs(max_jobs=cfg.bg_max_jobs, buffer_chars=cfg.bg_buffer_chars,
                                   max_lifetime=cfg.bg_max_lifetime, stop_grace=cfg.bg_stop_grace,
                                   keep_finished=cfg.bg_keep_finished)
        for ag in self.agents.values():
            ag.ctx.memory = self.memory
            ag.ctx.jobs = self.jobs
        self._refresh_memory_prompts()

    def _shutdown_jobs(self) -> None:
        """Termina i job ancora vivi. Chiamata su TUTTE le uscite (fine del REPL,
        EOF/Ctrl-C, one-shot) più `atexit` dentro il registro come ultima rete: senza
        questo, una scansione lanciata in background continuerebbe a girare dopo la
        chiusura di flair, invisibile e non più fermabile dall'interfaccia."""
        try:
            stopped = self.jobs.stop_all()
        except Exception as exc:  # noqa: BLE001 — l'uscita non deve mai fallire per questo
            logging.getLogger("flair.cli").warning("Chiusura dei job non completata: %s", exc)
            return
        if stopped:
            self.console.print(f"[dim]stopped {stopped} background job(s).[/dim]")

    def _callbacks(self) -> dict:
        return dict(
            on_tool=self._on_tool,
            on_result=self._on_result,
            on_reasoning=self._on_reasoning,
            on_reasoning_delta=self._on_reasoning_delta,
            on_delta=self._on_delta,
            on_compact=self._on_compact,
            on_prune=self._on_prune,
            approve=self._approve,
            on_interrupt=self._on_interrupt,
        )

    def _chdir_root(self) -> None:
        """Allinea la directory di processo a cfg.root. La modalità coding usa già
        cwd=root nei comandi; questo allineamento estende la stessa cartella di lavoro
        anche all'agente general (che resta SENZA confinamento, ma eredita la cwd):
        così «crea un report nella cartella attuale» è coerente con la root impostata."""
        try:
            os.chdir(self.cfg.root)
        except OSError as exc:
            self.console.print(f"[yellow]⚠ could not change directory to {self.cfg.root}: {exc}[/yellow]")

    def _apply_root(self, new_root: Path) -> None:
        """Cambia la root a runtime (comando /root): aggiorna cfg.root, allinea la
        directory di processo e ricostruisce il coding agent per ricaricare le
        istruzioni di progetto, preservando la memoria condivisa. La memoria di
        SESSIONE non viene toccata: è ancorata alla sessione, non al percorso."""
        self.cfg.root = new_root
        self._chdir_root()
        self.agents["coding"] = coding_agent.build(self.cfg, self.provider, conversation=self.convo, **self._callbacks())
        self._base_prompts["coding"] = self.agents["coding"].system_prompt
        self.agents["coding"].ctx.memory = self.memory
        self._refresh_memory_prompts()

    def _switch_provider(self, target: str) -> bool:
        """Cambia provider a runtime (/provider). Oltre a ricreare il client,
        riallinea OGNI riferimento che gli agenti tengono al provider — incluso
        ctx.provider, usato dai tool che delegano (explore): senza, il percorso
        sequenziale costruiva il sub-agente sul provider VECCHIO (client, chiave
        e listino di prima). Ritorna False su target sconosciuto (nulla cambia)."""
        if target not in ("deepseek", "openai", "local"):
            return False
        self.cfg.provider = target
        self.cfg.refresh_pricing()
        self.provider = create_provider(self.cfg)
        for a in self.agents.values():
            a.provider = self.provider
            a.ctx.provider = self.provider
        self._announce_provider_profile()
        return True

    def _announce_provider_profile(self) -> None:
        """Il degrado a profilo compat non deve MAI essere silenzioso: se lo slot
        deepseek punta a un endpoint terzo (o l'ufficiale cambiasse dominio senza
        riallineare la lista host), lo si dichiara all'avvio del REPL e allo
        switch — una riga, così una settimana di stime sballate non passa mai."""
        if self.cfg.provider == "deepseek" and not getattr(self.provider, "first_party", True):
            self.console.print("[dim]deepseek: third-party endpoint → compat profile (no reasoning "
                               "passback, flat pricing — set FLAIR_PRICE_* to the host's card; "
                               "FLAIR_DEEPSEEK_FIRST_PARTY=true to force the full protocol)[/dim]")

    def _refresh_memory_prompts(self) -> None:
        """(Ri)compone i system prompt: base (prompt + istruzioni di progetto) + blocco
        memoria. Da chiamare SOLO ai confini di sessione (avvio, /load, /memory clear,
        /root): lì la cache del prefisso si rinnova comunque, quindi è gratis. Memoria
        vuota o disabilitata → prompt base, zero caratteri in più."""
        block = self.memory.block() if self.cfg.memory_enabled else ""
        for key, ag in self.agents.items():
            ag.system_prompt = self._base_prompts[key] + block

    # ── sessioni (persistenza) ────────────────────────────────────────────────

    def _session_state(self) -> dict:
        return {
            "last_agent": self.last_agent,
            "conversation": self.convo.dump(),
        }

    def _remember_note(self, note: str) -> None:
        # Porta manuale sulla memoria di sessione: stessa meccanica e stesse
        # guardie del tool `remember` dell'agente (dedup, filtro segreti, tetto),
        # ma deterministica e senza round-trip LLM. Prima di questo comando
        # digitare "/remember X" cadeva al modello come prompt e funzionava solo
        # per sua gentile interpretazione — ora è un contratto.
        if not self.cfg.memory_enabled:
            self.console.print("[dim]memory disabled (FLAIR_MEMORY=false).[/dim]\n")
            return
        if not note:
            self.console.print("[dim]usage: /remember <note>[/dim]\n")
            return
        ok, msg = self.memory.add(note)
        if ok:
            # NIENTE _refresh_memory_prompts qui: stesso contratto del tool
            # `remember` dell'agente — toccare il system prompt a metà sessione
            # romperebbe il prefisso in cache. La nota entra nel prompt al
            # prossimo confine di sessione (avvio, /load, /root, /memory clear).
            self._save_session()             # sessione salvata → sidecar aggiornato
            self.console.print(f"[green]✓ noted.[/green] [dim]({len(self.memory.notes)} in memory)[/dim]\n")
        else:
            self.console.print(f"[yellow]⚠ {msg}[/yellow]\n")

    def _save_session(self) -> None:
        if self.session_name:
            self.session.save(self.session_name, self._session_state())
            if self.cfg.memory_enabled:
                # Sidecar della memoria accanto al JSON (vuota → sidecar rimosso).
                self.session.save_memory(self.session_name, self.memory.to_text())

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
        # Ripristina la memoria della sessione (sidecar assente/illeggibile → vuota:
        # le sessioni pre-memoria si caricano senza errori) e ricompone i prompt.
        # Siamo a un confine di sessione: la cache del prefisso si rinnova comunque.
        if self.cfg.memory_enabled:
            _, truncated = self.memory.load_text(self.session.load_memory(name))
            if truncated:
                self.console.print("[yellow]⚠ memory over the cap: loaded truncated "
                                   f"({self.memory.used_chars()}/{self.memory.max_chars} chars).[/yellow]")
            self._refresh_memory_prompts()
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
        self._stop_thinking()   # mai scrivere su stdout con lo status attivo
        sys.stdout.write(piece)
        sys.stdout.flush()
        self._mid_line = not piece.endswith("\n")

    def _on_tool(self, name: str, args: dict) -> None:
        # Raccogliamo sempre l'evento (serve a logger e a --json); stampiamo solo in human.
        self._turn_tools.append({"name": name, "args": {k: _short(v, 200) for k, v in args.items()}})
        if self.output_mode != "human":
            return
        self._stop_thinking()
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
        self.console.print(f"[dim]  ✂ context: pruned {count} superseded tool outputs[/dim]")

    def _on_reasoning_delta(self, piece: str) -> None:
        # Le fasi di pensiero lunghe (effort max: anche 10k+ char) congelavano la
        # CLI senza segni di vita fino al pannello. Qui: spinner + contatore che
        # cresce coi delta; il pannello completo resta il render finale (leggibile),
        # lo status è solo il battito cardiaco.
        if self.output_mode != "human":
            return
        if self._think_status is None:
            self._newline_if_needed()
            self._think_chars = 0
            self._think_t0 = time.monotonic()
            self._think_status = self.console.status("[dim]reasoning…[/dim]", spinner=_spinner_name())
            self._think_status.start()
        self._think_chars += len(piece)
        elapsed = time.monotonic() - self._think_t0
        self._think_status.update(f"[dim]reasoning… {_fmt_thinking(self._think_chars, elapsed)}[/dim]")

    def _stop_thinking(self) -> None:
        status, self._think_status = self._think_status, None
        if status is not None:
            status.stop()

    def _on_reasoning(self, text: str) -> None:
        if self.output_mode != "human":
            return
        self._stop_thinking()
        self._newline_if_needed()
        self.console.print(Panel(Text(text.strip(), style="italic dim"),
                                 title="[dim]reasoning[/dim]", border_style="dim", padding=(0, 1)))

    def _on_compact(self, before: int, after: int) -> None:
        if self.output_mode != "human":
            return
        self._newline_if_needed()
        self.console.print(f"[dim]  ⟳ context compacted: {before} → {after} messages[/dim]")

    # ── approvazione + anteprima diff ─────────────────────────────────────────

    def _approve(self, name: str, args: dict) -> Approval:
        self._newline_if_needed()
        if name in self._always_allow:   # "always" vale per l'intero tool, per la sessione
            return Approval(allowed=True)

        preview = self._preview(name, args)
        if preview is not None:
            self.console.print(preview)
        else:
            target = args.get("command") or args.get("path") or args.get("name") or args.get("script") or ""
            self.console.print(f"  [yellow]⚠ confirm[/yellow] [bold]{name}[/bold] → {_short(target, 80)}")

        try:
            # Le parentesi quadre sono escape-ate: Rich le interpreterebbe come markup.
            raw = self.console.input(
                r"    proceed? \[y]es / \[n]o / \[a]lways / \[s]top "
                r"[dim](or answer with a message: «no, use port 8080»)[/dim] ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return Approval(allowed=False, stop=True)   # Ctrl-C/EOF al prompt = ferma il flusso
        # Prima parola = decisione, resto = messaggio per il modello. La virgola
        # iniziale di «no, usa la porta 8080» non deve far parte della decisione.
        head, _, rest = raw.partition(" ")
        verb = head.strip().strip(",;:").lower()
        note = rest.strip().lstrip(",;:").strip()
        if verb in ("s", "stop"):
            return Approval(allowed=False, stop=True)
        if verb in ("a", "always", "sempre"):
            self._always_allow.add(name)
            self.console.print(f"[dim]  ok: I won't ask again for «{name}» in this session.[/dim]")
            return Approval(allowed=True)
        # NB: 's' è riservato a stop; per il sì in italiano si usa 'si'/'sì'. Una
        # risposta non riconosciuta resta un NO (come prima), ma ora l'intero testo
        # diventa il messaggio: rispondere con una frase è il modo naturale di dire
        # «no, fai invece così», e prima quella frase andava perduta.
        if verb in ("y", "yes", "si", "sì", "ok"):
            return Approval(allowed=True, note=note)
        if verb in ("n", "no"):
            return Approval(allowed=False, note=note)
        return Approval(allowed=False, note=raw)

    def _on_interrupt(self) -> tuple[str, str]:
        """Cosa fare dopo un Ctrl-C a metà turno. Prima l'interruzione chiudeva
        sempre il turno; ora si può proseguire allegando un messaggio, che è il modo
        di correggere la rotta senza perdere il lavoro già fatto.

        In assenza di un terminale (headless, `-p`, pipe) NON si chiede nulla e si
        conserva il comportamento storico: là non c'è nessuno a cui domandare, e un
        prompt su stdin non interattivo si tradurrebbe in un EOF."""
        if not sys.stdin or not sys.stdin.isatty():
            return ("stop", "")
        self._newline_if_needed()
        try:
            raw = self.console.input(
                r"  [yellow]⏸ interrupted[/yellow] \[s]top / \[c]ontinue / or type a message "
                r"[dim](it goes to the model and the turn continues)[/dim] ").strip()
        except (EOFError, KeyboardInterrupt):
            # Un secondo Ctrl-C significa «basta davvero».
            self.console.print()
            return ("stop", "")
        head, _, rest = raw.partition(" ")
        verb = head.strip().strip(",;:").lower()
        if verb in ("", "s", "stop"):
            return ("stop", "")
        if verb in ("c", "continue", "continua"):
            note = rest.strip().lstrip(",;:").strip()
            self.console.print("[dim]  continuing…[/dim]" if not note else "[dim]  continuing with your note.[/dim]")
            return ("continue", note)
        # Qualunque altra frase = «continua, e tieni conto di questo»: è la forma
        # naturale di interloquire, e chiederlo con una parola chiave in più
        # renderebbe la feature scomoda proprio nel momento in cui serve.
        self.console.print("[dim]  continuing with your message.[/dim]")
        return ("continue", raw)

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
                        Text(f"⚠ {exc}\nThe edit will likely fail (old_string not found or ambiguous).",
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
                body.append("…[diff truncated]\n", style="dim")
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
            body.append("(no textual difference detected)\n", style="dim")
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
            self.console.print("[yellow]⏹ Turn interrupted. You're back at the prompt.[/yellow]\n")
        except Exception as exc:  # noqa: BLE001
            self._newline_if_needed()
            self.console.print(f"[red]⚠ The turn failed: {type(exc).__name__}: {exc}[/red]")
            self.console.print("[dim]You can retry. If it is a model network timeout, "
                               "try again shortly or lower FLAIR_TIMEOUT.[/dim]\n")

    def _emit_json(self, obj: dict) -> None:
        # Una sola riga JSON su stdout (JSONL-friendly), nient'altro in modalità json.
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def run_once(self, task: str, agent_key: str | None = None, think: bool = False,
                 attachments: list[dict] | None = None) -> int:
        """Esegue un singolo task (modalità `-p`) e ritorna un exit code per gli script:
        0 done, 2 max-step, 3 loop, 4 fermato (serviva approvazione o stop), 5 budget,
        1 errore, 130 interruzione. In modalità --json emette SEMPRE un oggetto su stdout
        (anche su errore/interruzione), così il contratto resta affidabile per l'automazione."""
        # try/finally su TUTTO il corpo: qualunque via d'uscita (successo, errore,
        # Ctrl-C, budget) deve passare dalla terminazione dei job in background,
        # altrimenti un one-shot in uno script lascerebbe processi vivi a ogni giro.
        try:
            try:
                result = self.run_task(task, agent_key=agent_key, think=think, attachments=attachments)
            except KeyboardInterrupt:
                self._newline_if_needed()
                if self.output_mode == "json":
                    self._emit_json({"ok": False, "agent": self.last_agent, "stopped_reason": "interrupted",
                                     "response": "", "error": "interrupted"})
                elif self.output_mode == "human":
                    self.console.print("[yellow]⏹ Interrupted.[/yellow]")
                return 130
            except Exception as exc:  # noqa: BLE001
                self._newline_if_needed()
                if self.output_mode == "json":
                    self._emit_json({"ok": False, "agent": self.last_agent, "stopped_reason": "error",
                                     "response": "", "error": f"{type(exc).__name__}: {exc}"})
                elif self.output_mode == "human":
                    self.console.print(f"[red]⚠ Error: {type(exc).__name__}: {exc}[/red]")
                return 1

            if self.output_mode == "json":
                cost = self.provider.estimate_cost(result.usage, self.cfg)
                self._emit_json(build_result_json(self.last_agent, task, result, self._turn_tools, cost))
            elif self.output_mode == "quiet":
                sys.stdout.write((result.content or "") + "\n")
                sys.stdout.flush()
            return exit_code_for(result.stopped_reason)
        finally:
            self._shutdown_jobs()

    def run_task(self, task: str, agent_key: str | None = None, think: bool = False,
                 attachments: list[dict] | None = None):
        if agent_key is None:
            agent_key = router.classify(task, self.provider, self.last_agent, convo=self.convo)
        self.last_agent = agent_key
        agent = self.agents[agent_key]
        # Allegati (/img, --image): il content del turno diventa multipart, ma il
        # router classifica sul TESTO e il JSONL logga il testo — mai i blob base64.
        content: str | list = task
        if attachments:
            content = [{"type": "text", "text": task}, *attachments]
        self._turn_tools = []
        self._mid_line = False
        human = self.output_mode == "human"

        if human:
            self.console.print(f"[dim]→ agent: {agent_key}[/dim]")
        if human and self.cfg.stream:
            self.console.print(f"[bold cyan]flair · {agent_key}[/bold cyan]")
            try:
                result = agent.run(content, think=think)
            finally:
                self._stop_thinking()   # rete di sicurezza: mai spinner orfani su errore
            self._newline_if_needed()
            self.console.print()
        else:
            result = agent.run(content, think=think)
            if human and result.stopped_reason not in ("stopped", "budget"):
                self.console.print(Panel(
                    Markdown(result.content or "(empty)"),
                    title=f"[bold cyan]flair · {agent_key}[/bold cyan]",
                    border_style="cyan", padding=(1, 2),
                ))

        if human and result.stopped_reason == "stopped":
            self.console.print("[yellow]⏹ Flow stopped: you're back in control. Tell me how to proceed.[/yellow]\n")
        if human and result.stopped_reason == "budget":
            self.console.print("[yellow]⏹ Stopped: cost cap reached "
                               "(--max-cost / FLAIR_MAX_COST).[/yellow]\n")
        if human and result.truncated:
            if (result.content or "").strip():
                self.console.print("[yellow]⚠ Response truncated: output token limit reached. "
                                   "Ask to continue, or raise FLAIR_MAX_TOKENS.[/yellow]")
            else:
                # Tutto il budget è finito nel ragionamento, prima di produrre una risposta:
                # "continuare" non aiuta (il ragionamento non si riporta). Indica il fix vero.
                self.console.print("[yellow]⚠ No answer: the output budget (FLAIR_MAX_TOKENS) "
                                   "was exhausted during reasoning. A 'thinking' model needs "
                                   "far more than 8000: raise FLAIR_MAX_TOKENS, or use the fast "
                                   "model (without --think) for tool work.[/yellow]")

        if self.logger:
            self.logger.log_turn(agent_key, task, result, self._turn_tools,
                                 cache_breaks=self.convo.cache_breaks,
                                 provider=self.cfg.provider, model=self.cfg.active.model,
                                 cost_usd=self.provider.estimate_cost(result.usage, self.cfg))

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
        labels = {"max_steps": "max steps", "loop": "loop detected", "stopped": "stopped", "budget": "budget"}
        flag = f" | [yellow]{labels[reason]}[/yellow]" if reason in labels else ""
        self.console.print(f"[dim]  this turn · step {steps} · {self._cost_line(usage)}{flag}[/dim]")

    def _print_session(self) -> None:
        self.console.print(f"[dim]  session   · {self._cost_line(self._session_usage())}[/dim]")
        if self.last_agent:
            tokens, frac = self.agents[self.last_agent].context_fill()
            self.console.print(
                f"[dim]  context   · {self.last_agent}: {round(frac * 100)}% "
                f"({_kfmt(tokens)}/{_kfmt(self.cfg.context_window)})[/dim]")
        self._maybe_cost_warn()
        self.console.print()

    def _print_session_recap(self) -> None:
        """Dopo un load: mostra dove si era rimasti, così si riprende contestualmente
        senza doversi ricordare a memoria cosa si stava facendo. Coda breve e dim:
        orienta, non ristampa la sessione."""
        lines, earlier = _recap_messages(self.convo.messages)
        if not lines:
            return
        note = f" ({earlier} earlier messages not shown)" if earlier else ""
        self.console.print(f"[dim]── where you left off{note} ──[/dim]")
        labels = {"user": "[green]you ▶[/green]", "assistant": "[cyan]flair ·[/cyan]",
                  "summary": "[yellow]⟳ summary[/yellow]"}
        for role, text in lines:
            self.console.print(f"{labels.get(role, role)} [dim]{escape(text)}[/dim]", highlight=False)
        self.console.print()

    def _maybe_cost_warn(self) -> None:
        if self.cfg.cost_warn and not self._cost_warned:
            cost = self.provider.estimate_cost(self._session_usage(), self.cfg)
            if cost >= self.cfg.cost_warn:
                self.console.print(
                    f"[yellow]  ⚠ session cost ~${cost:.4f}: over the warning threshold of "
                    f"${self.cfg.cost_warn:.2f} (FLAIR_COST_WARN)[/yellow]")
                self._cost_warned = True

    # ── REPL ──────────────────────────────────────────────────────────────────

    def _print_help(self) -> None:
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan",
                      padding=(0, 3, 0, 0), border_style="dim")
        table.add_column("command", style="bold", no_wrap=True)
        table.add_column("what it does", style="dim")
        # Voci derivate da _COMMANDS (unica fonte con il dispatch) + le parole di
        # uscita, che non sono comandi con handler.
        rows = [(disp, desc) for _n, disp, desc, _m in _COMMANDS]
        rows.append(("exit | quit", "leave the REPL"))
        for cmd, desc in rows:
            table.add_row(Text(cmd), desc)
        self.console.print(table)
        self.console.print(
            "[dim]Startup flags (CLI): «flair -h». Examples: flair --think -p \"...\", "
            "flair --session work, flair --continue, flair --provider openai.[/dim]\n")

    def _print_tools(self) -> None:
        key = self.last_agent or "general"
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan",
                      padding=(0, 3, 0, 0), border_style="dim")
        table.add_column("tool", style="bold", no_wrap=True)
        table.add_column("what it does", style="dim")
        for name, desc in self.agents[key].toolset.catalog():
            icon = _TOOL_ICON.get(name, "🔧")
            line = " ".join(desc.split())                  # normalizza gli spazi
            if len(line) > 100:
                line = line[:99] + "…"
            table.add_row(Text(f"{icon} {name}"), line)
        self.console.print(table)
        self.console.print(
            f"[dim]Tools of the «{key}» agent"
            f"{' (sticky)' if self.last_agent else ' (default; no turn yet)'}. "
            "Switch agent with /code, /do.[/dim]\n")

    # ── Handler dei comandi ──────────────────────────────────────────────────
    # Uno per voce di _COMMANDS. Ognuno riceve gli ARGOMENTI già separati dal nome
    # (stringa vuota se assenti): il parsing della riga sta tutto in _dispatch, così
    # ogni handler è una funzione piccola e testabile da sola.

    def _cmd_help(self, arg: str) -> None:
        self._print_help()

    def _cmd_tools(self, arg: str) -> None:
        self._print_tools()

    def _cmd_agent(self, arg: str) -> None:
        self.console.print(f"[dim]current agent (sticky): {self.last_agent or 'none'}[/dim]\n")

    def _cmd_reset(self, arg: str) -> None:
        self.convo.reset()
        self.last_agent = None
        self.console.print("[yellow]conversation cleared.[/yellow]\n")

    def _cmd_cost(self, arg: str) -> None:
        self.console.print(f"[dim]  session · {self._cost_line(self._session_usage())} "
                           f"| prefix breaks {self.convo.cache_breaks}[/dim]\n")

    def _cmd_sessions(self, arg: str) -> None:
        items = self.session.list()
        if not items:
            self.console.print("[dim]no saved sessions.[/dim]\n")
        else:
            body = "\n".join(f"  • {n}  [dim]{ts}[/dim]" for n, ts in items)
            self.console.print(f"[dim]saved sessions:[/dim]\n{body}\n")

    def _cmd_remember(self, arg: str) -> None:
        self._remember_note(arg)

    def _cmd_memory(self, arg: str) -> None:
        if not self.cfg.memory_enabled:
            self.console.print("[dim]memory disabled (FLAIR_MEMORY=false).[/dim]\n")
            return
        sub = arg.strip().lower()
        if sub == "clear":
            if not self.memory.notes:
                self.console.print("[dim]memory already empty.[/dim]\n")
                return
            try:
                ans = self.console.input(f"    clear {len(self.memory.notes)} notes? \\[y]es / \\[n]o ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return
            if ans in ("y", "yes", "si", "sì"):
                self.memory.clear()
                self._refresh_memory_prompts()   # confine esplicito: il prompt perde il blocco
                self._save_session()             # se la sessione è salvata, rimuove anche il sidecar
                self.console.print("[yellow]memory cleared.[/yellow]\n")
            return
        if sub:
            self.console.print("[dim]usage: /memory  or  /memory clear[/dim]\n")
            return
        if not self.memory.notes:
            self.console.print("[dim]memory is empty. The agent jots durable facts here with the "
                               "`remember` tool; notes follow the session (/save, /load).[/dim]\n")
            return
        body = "\n".join(f"  {i}. {n}" for i, n in enumerate(self.memory.notes, 1))
        self.console.print(f"[dim]session memory "
                           f"({self.memory.used_chars()}/{self.memory.max_chars} chars):[/dim]\n{body}\n")

    def _cmd_save(self, arg: str) -> None:
        name = arg.strip() or (self.session_name or "default")
        self.session_name = name
        path = self.session.save(name, self._session_state())
        msg = f"[green]session saved: {name}[/green]" if path else "[red]save failed (see the log).[/red]"
        self.console.print(msg + "\n")

    def _cmd_load(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self.console.print("[dim]usage: /load <name>[/dim]\n")
        elif self._load_session(name):
            self.console.print(f"[green]session resumed: {self.session_name}[/green]\n")
            self._print_session_recap()
        else:
            self.console.print(f"[yellow]session '{name}' not found.[/yellow]\n")

    def _cmd_compact(self, arg: str) -> None:
        if self.last_agent:
            if not self.agents[self.last_agent].compact():
                self.console.print("[dim]nothing to compact.[/dim]\n")
        else:
            self.console.print("[dim]no active conversation.[/dim]\n")

    def _cmd_provider(self, arg: str) -> None:
        target = arg.strip().lower()
        if target:
            if not self._switch_provider(target):
                self.console.print("[yellow]invalid provider (deepseek|openai|local).[/yellow]\n")
            else:
                pc = self.cfg.active
                self.console.print(f"[yellow]provider → {target} | model: {pc.model} | thinking: {pc.think_model}[/yellow]\n")
        else:
            pc = self.cfg.active
            self.console.print(f"[dim]provider: {self.cfg.provider} | model: {pc.model} | thinking: {pc.think_model}[/dim]\n")

    def _cmd_think_model(self, arg: str) -> None:
        if arg.strip():
            self.cfg.active.think_model = arg.strip()
            self.console.print(f"[yellow]thinking model → {self.cfg.active.think_model}[/yellow]\n")
        else:
            self.console.print("[dim]usage: /think-model <name>[/dim]\n")

    def _cmd_model(self, arg: str) -> None:
        if arg.strip():
            self.cfg.active.model = arg.strip()
            self.cfg.refresh_pricing()
            self.console.print(f"[yellow]model → {self.cfg.active.model}[/yellow]\n")
        else:
            self.console.print("[dim]usage: /model <name>[/dim]\n")

    def _cmd_root(self, arg: str) -> None:
        if not arg.strip():
            return
        new_root = Path(arg.strip()).expanduser().resolve()
        if not new_root.is_dir():
            self.console.print(f"[yellow]nonexistent folder: {new_root}[/yellow]\n")
        else:
            self._apply_root(new_root)
            self.console.print(
                f"[yellow]root → {self.cfg.root} "
                "(working folder for coding and general)[/yellow]\n")

    def _cmd_img(self, arg: str) -> None:
        parts = arg.split(None, 1)
        if not parts:
            self.console.print("[yellow]usage: /img <path> [prompt][/yellow]\n")
            return
        if not getattr(self.cfg.active, "vision", False):
            self.console.print(
                "[yellow]The current provider/endpoint has no vision support: enable it "
                "only if the endpoint accepts images (e.g. llama-server with --mmproj) "
                "via DEEPSEEK_VISION / OPENAI_VISION / LOCAL_VISION=true in .env.[/yellow]\n")
            return
        prompt = parts[1].strip() if len(parts) == 2 else "Describe what you see in this image."
        try:
            # Path scelto dall'UTENTE sulla propria macchina: niente sandbox
            # (come aprire un file a mano), solo expanduser+resolve.
            part, note = images.load_image_part(self.cfg, Path(parts[0]).expanduser().resolve())
        except ToolError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]\n")
            return
        self.console.print(f"[dim]📎 {note}[/dim]")
        self.run_task(prompt, attachments=[part], think=self.default_think)

    def _cmd_jobs(self, arg: str) -> None:
        """Vista UMANA dei job: quando il turno torna a te, dice cosa sta ancora
        girando — informazione che altrimenti resterebbe solo nel contesto del
        modello. `/jobs stop <id>` ferma un job dall'interfaccia."""
        from .tools.jobs import _job_line
        self.jobs.reap()
        parts = arg.split(None, 1)
        if parts and parts[0].lower() == "stop":
            if len(parts) != 2:
                self.console.print("[dim]usage: /jobs stop <id>[/dim]\n")
                return
            job = self.jobs.stop(parts[1].strip())
            if job is None:
                self.console.print(f"[yellow]no job with id '{parts[1].strip()}'.[/yellow]\n")
            else:
                self.console.print(f"[yellow]job {job.id} stopped ({job.status()}).[/yellow]\n")
            return
        rows = self.jobs.all()
        if not rows:
            self.console.print("[dim]no background jobs in this session.[/dim]\n")
            return
        body = "\n".join(f"  {_job_line(j)}" for j in rows)
        running = sum(1 for j in rows if j.running)
        self.console.print(f"[dim]{len(rows)} job(s), {running} running:[/dim]\n{body}\n")

    def _cmd_code(self, arg: str) -> None:
        if arg:
            self._safe_run_task(arg, agent_key="coding", think=self.default_think)

    def _cmd_do(self, arg: str) -> None:
        if arg:
            self._safe_run_task(arg, agent_key="general", think=self.default_think)

    def _cmd_think(self, arg: str) -> None:
        if arg:
            self._safe_run_task(arg, think=True)

    def _dispatch(self, line: str) -> bool:
        """Esegue una riga del REPL; ritorna False quando la sessione va chiusa.

        Il match è sul PRIMO TOKEN ESATTO, non per prefisso: la vecchia catena di
        `low.startswith("/do")` intercettava anche "/documenta il codice" e mandava
        all'agente generico il task mutilato "cumenta il codice". E una riga che
        inizia con "/" ma non è un comando non viene più spedita al modello come
        task — un typo (`/comapct`) costava un turno a pagamento: ora riceve un
        suggerimento, calcolato sui nomi della tabella."""
        low = line.lower()
        if low in _QUIT_WORDS:
            self.console.print("[dim]bye![/dim]")
            return False
        if not line.startswith("/"):
            self._safe_run_task(line, think=self.default_think)
            return True
        parts = line.split(None, 1)
        name = parts[0][1:].lower()
        arg = parts[1].strip() if len(parts) == 2 else ""
        handler = {n: getattr(self, m) for n, _d, _h, m in _COMMANDS}.get(name)
        if handler is None:
            close = difflib.get_close_matches(name, [n for n, *_ in _COMMANDS], n=1)
            hint = f" Did you mean /{close[0]}?" if close else " Type /help for the list of commands."
            self.console.print(f"[yellow]unknown command: {parts[0]}.{hint}[/yellow]\n")
            return True
        handler(arg)
        return True

    def repl(self) -> None:
        pc = self.cfg.active
        log_note = f"\nlog: {self.logger.path}" if self.logger else ""
        sess_note = f" | session: {self.session_name}" if self.session_name else ""
        self.console.print(Panel(
            Text.from_markup(
                f"[bold cyan]flair {__version__}[/bold cyan] [dim]— AI assistant (coding + general)[/dim]\n"
                f"[dim]provider: {self.cfg.provider} | model: {pc.model} | thinking: {pc.think_model}"
                f"{' | think: ON every turn' if self.default_think else ''}{sess_note}\n"
                f"root: {self.cfg.root}{log_note}[/dim]"
            ),
            border_style="cyan", padding=(1, 2),
        ))
        self.console.print("[dim]/help for commands. Type a request (coding or general).[/dim]\n")
        self._announce_provider_profile()
        if self.convo.messages:
            # Sessione ripresa da --session/--continue: orienta subito.
            self._print_session_recap()

        while True:
            try:
                line = self.console.input("[bold green]▶[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]bye![/dim]")
                self._shutdown_jobs()
                return
            if not line:
                continue
            if not self._dispatch(line):
                self._shutdown_jobs()
                return


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


def _build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser dei flag di avvio. Estratto da main() per poterlo
    testare direttamente (es. la regressione: --provider deve accettare 'local',
    come già fanno FLAIR_PROVIDER e il comando /provider del REPL)."""
    ap = argparse.ArgumentParser(prog="flair", description="Agentic AI assistant (coding + general) on DeepSeek/OpenAI.")
    ap.add_argument("--version", action="version", version=f"flair {__version__}")
    ap.add_argument("-p", "--prompt", help="run a single task and exit (use '-' to read from stdin)")
    ap.add_argument("--provider", choices=["deepseek", "openai", "local"], help="LLM provider")
    ap.add_argument("--agent", choices=["coding", "general", "auto"], default="auto", help="force an agent (default: auto)")
    ap.add_argument("--root", help="working root for the coding agent")
    ap.add_argument("--think", action="store_true", help="use the thinking model (one-shot: the task; REPL: every turn of the session)")
    ap.add_argument("--image", action="append", metavar="PATH",
                    help="with -p: attach an image to the task (repeatable; vision endpoints only)")
    ap.add_argument("--yes", action="store_true", help="auto-approve destructive tools")
    ap.add_argument("--read-only", dest="read_only", action="store_true",
                    help="unattended execution: disables destructive tools (writes/edits/commands)")
    ap.add_argument("--max-cost", dest="max_cost", type=float, default=None,
                    help="HARD session cost cap in USD: past it, the task stops")
    ap.add_argument("--json", action="store_true", help="with -p: emit a JSON object (for automation)")
    ap.add_argument("-q", "--quiet", action="store_true", help="with -p: print only the final answer")
    ap.add_argument("--no-stream", dest="no_stream", action="store_true", help="disable streaming")
    ap.add_argument("--log", help="folder to write the session log to (JSONL)")
    ap.add_argument("--model", help="override the fast model")
    ap.add_argument("--think-model", dest="think_model", help="override the thinking model")
    ap.add_argument("--session", help="use/create a session with this name (autosave)")
    ap.add_argument("--continue", dest="continue_", action="store_true", help="resume the latest saved session")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Modalità di output one-shot: json/quiet valgono solo con -p (la REPL resta human).
    json_mode = bool(args.prompt is not None and args.json)
    quiet_mode = bool(args.prompt is not None and args.quiet and not args.json)
    headless = json_mode or quiet_mode

    console = Console(stderr=headless)  # in headless i messaggi umani vanno su stderr
    # Costruzione E validazione nello stesso try: load_config() può già rifiutare un
    # valore (es. FLAIR_THINK_STEPS fuori dominio solleva ValueError) e prima usciva
    # con un traceback invece del messaggio pulito riservato agli errori di config.
    try:
        cfg = _build_config(args)
        if headless:
            cfg.stream = False          # niente delta su stdout: resta pulito per la macchina
        cfg.validate()
    except (RuntimeError, ValueError) as exc:
        if json_mode:
            sys.stdout.write(json.dumps(
                {"ok": False, "stopped_reason": "config_error", "response": "", "error": str(exc)}) + "\n")
        else:
            console.print(f"[bold red]Invalid configuration:[/bold red] {exc}")
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
            console.print(f"[dim]session resumed: {args.session}[/dim]")
        else:
            console.print(f"[dim]new session: {args.session}[/dim]")
    elif args.continue_:
        latest = cli.session.latest()
        if latest and cli._load_session(latest):
            console.print(f"[dim]resumed latest session: {latest}[/dim]")
        else:
            console.print("[dim]no session to resume.[/dim]")

    if args.prompt is not None:
        prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
        prompt = prompt.strip()
        if not prompt:
            if json_mode:
                cli._emit_json({"ok": False, "agent": None, "stopped_reason": "error",
                                "response": "", "error": "empty prompt"})
            else:
                console.print("[red]Empty prompt.[/red]")
            return 1
        key = None if args.agent == "auto" else args.agent
        attachments: list[dict] | None = None
        if args.image:
            if not getattr(cfg.active, "vision", False):
                msg = ("the current provider/endpoint has no vision support "
                       "(set DEEPSEEK_VISION / OPENAI_VISION / LOCAL_VISION=true)")
                if json_mode:
                    cli._emit_json({"ok": False, "agent": None, "stopped_reason": "error",
                                    "response": "", "error": msg})
                else:
                    console.print(f"[red]--image: {msg}.[/red]")
                return 1
            attachments = []
            for raw in args.image:
                try:
                    part, _note = images.load_image_part(cfg, Path(raw).expanduser().resolve())
                except ToolError as exc:
                    if json_mode:
                        cli._emit_json({"ok": False, "agent": None, "stopped_reason": "error",
                                        "response": "", "error": str(exc)})
                    else:
                        console.print(f"[red]--image: {exc}[/red]")
                    return 1
                attachments.append(part)
        return cli.run_once(prompt, agent_key=key, think=args.think, attachments=attachments)
    cli.default_think = bool(args.think)
    cli.repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
