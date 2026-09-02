"""Comandi in background: dispatch, lettura incrementale, terminazione pulita.

Il loop agentico è sincrono: un `run_command` che dura minuti (una scansione, una
build, una suite di test) blocca il turno e costringe a scegliere tra un timeout
generoso e un'attesa inutile. Qui il comando viene avviato e "messo da parte": il
modello continua a lavorare e legge l'output quando vuole, con `job(action="check")`.

Tre scelte di progetto, tutte per robustezza su Windows E su POSIX:

1. **Un thread lettore, non il polling dei descrittori.** `select()`/`poll()` non
   funzionano sulle pipe di Windows (solo socket), e sbirciarle richiederebbe
   `PeekNamedPipe` via ctypes. Un thread demone che legge a blocchi è identico sui
   due sistemi e non ha rami di piattaforma.
2. **stderr rediretto su stdout.** Un solo stream significa ordine cronologico
   esatto, un solo lettore e — soprattutto — nessun rischio della classe di
   deadlock in cui una pipe si riempie mentre stiamo leggendo l'altra.
3. **Terminazione dell'ALBERO, non del figlio.** Con `shell=True` il figlio diretto
   è `cmd.exe`/`sh` e il processo che interessa (nmap) è un nipote: `terminate()`
   ucciderebbe la shell lasciando vivo il resto. Su POSIX si crea una sessione
   nuova e si segnala il process group; su Windows si usa `taskkill /T`.

Sui processi lasciati indietro: ogni job finito viene *reaped* (POSIX: `poll()`
raccoglie lo zombie, le pipe e il thread vengono chiusi), i job oltre la vita
massima sono terminati alla prima interazione, e `stop_all()` è chiamato su tutte
le uscite di flair — REPL, one-shot ed `atexit` come ultima rete. L'unico caso che
nessun programma può gestire è un SIGKILL su flair stesso.

Limite noto e dichiarato anche al modello (v. prompts): i programmi interattivi a
schermo intero (top, vim, installer con TUI) vedono una pipe invece di un
terminale e si comportano male o si bloccano. Gli strumenti a righe vanno bene.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

log = logging.getLogger("flair.tools.jobs")

# Il controllo di piattaforma esiste in DUE forme, di proposito: questa costante
# per la leggibilità a runtime, e il letterale `sys.platform == "win32"` dentro
# _terminate_tree — perché mypy sa restringere i tipi SOLO su quella forma. Su
# Windows `signal.SIGKILL` e `os.killpg` non esistono nemmeno negli stub, quindi
# senza il letterale il ramo POSIX viene type-checkato e va in errore pur non
# essendo mai eseguito (rosso in CI su Windows, verde su Linux).
_IS_WINDOWS = sys.platform == "win32"
_READ_CHUNK = 4096
_POLL_SLICE = 0.2      # granularità dell'attesa: piccola per restare reattivi a Ctrl-C


def _popen(command: str, cwd: str | None) -> subprocess.Popen:
    """Avvia il comando nella shell di sistema con un solo stream di output e in un
    gruppo di processi PROPRIO, così la terminazione può raggiungere anche i nipoti."""
    kwargs: dict = {
        "shell": True,
        "cwd": cwd,
        "stdin": subprocess.DEVNULL,   # fase 1: nessun stdin (v. modulo docstring)
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "bufsize": 0,                  # byte grezzi: la decodifica è nostra, con errors="replace"
    }
    if _IS_WINDOWS:
        # Gruppo separato: necessario per un eventuale CTRL_BREAK e per non ereditare
        # i segnali della console di flair.
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True   # setsid: abilita os.killpg sull'intero albero
    return subprocess.Popen(command, **kwargs)


def _terminate_tree(proc: subprocess.Popen, grace: float) -> None:
    """Termina il processo E i suoi discendenti, gentilmente e poi a forza.

    Il punto delicato: con `shell=True` il figlio diretto è la shell, e la sua
    morte non dice nulla sul processo che ci interessa. Un nipote che ignora
    SIGTERM sopravviveva proprio così: la shell moriva subito, `wait()` tornava e
    l'escalation non avveniva mai.

    Semantica scelta, deliberatamente semplice: SIGTERM al GRUPPO (creato al lancio
    con start_new_session, di cui il figlio è leader → pgid == pid del figlio),
    attesa del figlio entro la grazia, poi SIGKILL al gruppo SEMPRE — innocuo se
    non è rimasto nessuno. Interrogare il gruppo per sapere se è vuoto sembrava più
    elegante, ma non è affidabile: uno zombie conta come membro, e dove PID 1 non
    raccoglie (container) il gruppo risulterebbe vivo per sempre. Chi ha chiesto
    `stop` vuole che il job finisca, non che finisca con eleganza."""
    # `sys.platform` letterale, non _IS_WINDOWS: è la sola forma su cui mypy
    # restringe, e rende il ramo POSIX irraggiungibile (quindi non type-checkato)
    # quando l'analisi gira su Windows, dove killpg e SIGKILL non esistono.
    if sys.platform == "win32":
        if proc.poll() is None:
            taskkill = shutil.which("taskkill")
            if taskkill:
                # /T = albero, /F = forzato: su Windows non serve escalation.
                subprocess.run([taskkill, "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=grace + 5, check=False)
            else:   # ambiente Windows ridotto: almeno il figlio diretto
                try:
                    proc.kill()
                except OSError:
                    pass
        try:
            proc.wait(timeout=grace + 5)
        except subprocess.TimeoutExpired:
            log.warning("Il processo %s non risponde nemmeno a taskkill /F.", proc.pid)
        return

    pgid = proc.pid
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break                      # gruppo già vuoto: nulla da forzare
        except OSError as exc:
            log.warning("Segnale %s al gruppo %s fallito: %s", sig, pgid, exc)
            try:
                proc.kill()
            except OSError:
                pass
            break
        if sig is signal.SIGTERM:
            try:
                proc.wait(timeout=max(0.0, grace))
            except subprocess.TimeoutExpired:
                pass                   # non è uscito con le buone: si passa a SIGKILL
    try:
        proc.wait(timeout=2.0)         # raccoglie lo zombie del figlio diretto
    except subprocess.TimeoutExpired:
        log.warning("Il processo %s non risponde nemmeno a SIGKILL.", proc.pid)


class Job:
    """Un comando in esecuzione, con il suo output accumulato e un cursore di lettura.

    Il buffer è un ring in CARATTERI con offset assoluti: `total` conta tutto ciò
    che è stato prodotto, `text` conserva solo la coda entro il tetto. Così il
    cursore `read` sa sempre cosa il modello non ha ancora visto e quanto è stato
    scartato — invece di ri-consegnare tutto a ogni check (che sarebbe il modo più
    rapido di riempire il contesto)."""

    def __init__(self, job_id: str, command: str, cwd: str | None, buffer_chars: int) -> None:
        self.id = job_id
        self.command = command
        self.started_at = time.monotonic()
        self.buffer_chars = max(4096, buffer_chars)
        self._lock = threading.Lock()
        self._text = ""
        self._total = 0        # caratteri prodotti in assoluto
        self._read = 0         # offset assoluto già consegnato al modello
        self._dropped = 0      # caratteri usciti dal ring
        self._event = threading.Event()   # segnala output nuovo o fine del processo
        self.stopped_by_user = False
        self.proc = _popen(command, cwd)
        self._reader = threading.Thread(target=self._pump, name=f"flair-job-{job_id}", daemon=True)
        self._reader.start()

    # ── produzione ──────────────────────────────────────────────────────────
    def _pump(self) -> None:
        """Legge lo stream fino alla chiusura. Gira in un thread demone: se flair
        muore male non tiene in vita il processo."""
        stream = self.proc.stdout
        try:
            while stream is not None:
                chunk = stream.read(_READ_CHUNK)
                if not chunk:
                    break
                self._append(chunk.decode("utf-8", errors="replace"))
        except (OSError, ValueError):   # pipe chiusa mentre leggevamo: normale
            pass
        finally:
            self.proc.poll()            # POSIX: raccoglie subito lo zombie se ha finito
            self._event.set()           # sveglia chi era in attesa: non arriverà altro

    def _append(self, text: str) -> None:
        with self._lock:
            self._total += len(text)
            self._text += text
            excess = len(self._text) - self.buffer_chars
            if excess > 0:
                self._text = self._text[excess:]
                self._dropped += excess
        self._event.set()

    # ── stato ───────────────────────────────────────────────────────────────
    @property
    def exit_code(self) -> int | None:
        return self.proc.poll()

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def status(self) -> str:
        code = self.exit_code
        if code is None:
            return "running"
        return f"exited {code}" if not self.stopped_by_user else f"stopped (exit {code})"

    def unread(self) -> int:
        with self._lock:
            return max(0, self._total - self._read)

    # ── consumo ─────────────────────────────────────────────────────────────
    def take_new(self) -> tuple[str, int]:
        """(output non ancora consegnato, caratteri persi dal ring). Avanza il cursore."""
        with self._lock:
            available_from = self._total - len(self._text)
            start = max(self._read, available_from)
            lost = max(0, available_from - self._read)
            out = self._text[start - available_from:] if self._total > start else ""
            self._read = self._total
            self._event.clear()
            return out, lost

    def wait_for_activity(self, timeout: float) -> None:
        """Attende output nuovo o la fine del processo, al massimo `timeout` secondi.
        Attesa a fette brevi per restare interrompibile con Ctrl-C su entrambi i
        sistemi (su Windows una wait lunga non è sempre interrompibile)."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self._event.wait(min(_POLL_SLICE, max(0.0, deadline - time.monotonic()))):
                return
            if self.proc.poll() is not None:
                return

    def stop(self, grace: float) -> None:
        self.stopped_by_user = True
        _terminate_tree(self.proc, grace)
        self.close()

    def release_buffer(self) -> None:
        """Libera l'output di un job CONCLUSO e già letto per intero. Nient'altro
        arriverà e nulla resta da consegnare, quindi tenerlo in RAM è puro spreco:
        col tetto di default sono 256 KB per job, e una sessione lunga ne accumula
        decine. Metadati (id, comando, stato, durata) restano: servono a `list`."""
        if self.running:
            return
        with self._lock:
            if self._total - self._read <= 0:
                self._text = ""

    def close(self) -> None:
        """Rilascia le risorse di un job concluso: pipe chiusa, thread raccolto,
        zombie reaped. Idempotente."""
        try:
            self.proc.poll()
            if self.proc.stdout is not None and not self.proc.stdout.closed:
                self.proc.stdout.close()
        except (OSError, ValueError):
            pass
        if self._reader.is_alive():
            self._reader.join(timeout=1.0)


class BackgroundJobs:
    """Registro dei job della sessione. Condiviso PER RIFERIMENTO con i worker
    paralleli (come la memoria di sessione) e protetto da lock, perché qui non
    basta l'atomicità di list.append: si creano e si rimuovono voci."""

    def __init__(self, max_jobs: int = 8, buffer_chars: int = 262_144,
                 max_lifetime: int = 3600, stop_grace: float = 3.0,
                 keep_finished: int = 5) -> None:
        self.max_jobs = max_jobs
        self.buffer_chars = buffer_chars
        self.max_lifetime = max_lifetime
        self.stop_grace = stop_grace
        # Quanti job CONCLUSI restano nel registro. Un job finito serve ancora un
        # momento (il modello ne legge la coda dopo la fine), ma tenerli tutti fa
        # crescere `list` a ogni sessione lunga — righe di rumore nel contesto a
        # ogni chiamata — e trattiene i buffer. Oltre il tetto, i più vecchi
        # vengono sfrattati e il loro numero resta contato, così `list` lo dichiara
        # invece di far sparire i job in silenzio.
        self.keep_finished = max(0, keep_finished)
        self.evicted = 0
        self._jobs: dict[str, Job] = {}
        self._seq = 0
        self._lock = threading.Lock()
        # Ultima rete contro i processi lasciati indietro: se flair esce per una via
        # che non passa dal cleanup esplicito, ci pensa l'interprete.
        atexit.register(self.stop_all)

    # ── manutenzione ────────────────────────────────────────────────────────
    def reap(self) -> None:
        """Manutenzione, chiamata a ogni interazione — la pulizia non dipende dal
        fatto che il modello si ricordi di fare `stop`. Fa tre cose: chiude i job
        finiti (zombie, pipe, thread), termina quelli oltre la vita massima, e tiene
        il registro alla dimensione voluta liberando i buffer già letti e sfrattando
        i job conclusi più vecchi."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if not job.running:
                job.close()
                job.release_buffer()
            elif self.max_lifetime and job.elapsed > self.max_lifetime:
                log.warning("Job %s oltre la vita massima (%ss): terminato.", job.id, self.max_lifetime)
                job.stop(self.stop_grace)
        with self._lock:
            finished = sorted((j for j in self._jobs.values() if not j.running),
                              key=lambda j: j.started_at)
            for job in finished[:max(0, len(finished) - self.keep_finished)]:
                del self._jobs[job.id]
                self.evicted += 1

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.running]

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get((job_id or "").strip())

    # ── ciclo di vita ───────────────────────────────────────────────────────
    def start(self, command: str, cwd: str | None) -> Job:
        self.reap()
        with self._lock:
            live = sum(1 for j in self._jobs.values() if j.running)
            if live >= self.max_jobs:
                raise RuntimeError(
                    f"too many background jobs ({live}/{self.max_jobs}): stop one with "
                    "job(action=\"stop\", id=...) or raise FLAIR_BG_MAX_JOBS.")
            self._seq += 1
            job_id = f"j{self._seq}"
            job = Job(job_id, command, cwd, self.buffer_chars)
            self._jobs[job_id] = job
            return job

    def stop(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        job.stop(self.stop_grace)
        return job

    def stop_all(self) -> int:
        """Termina TUTTI i job vivi. Ritorna quanti ne ha fermati. Idempotente:
        chiamarla su ogni uscita (REPL, one-shot, atexit) è il punto di questa
        classe — un processo lasciato vivo dopo l'uscita di flair è un bug."""
        stopped = 0
        for job in self.all():
            if job.running:
                job.stop(self.stop_grace)
                stopped += 1
            else:
                job.close()
        return stopped


# ── Implementazione dei tool (condivisa dai due agenti: cambia solo il cwd) ────

_NO_JOBS = ("❌ Background jobs are not available in this session "
            "(no registry attached: unattended/read-only mode).")


def _fmt_duration(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds / 60:.1f}m"


def _job_line(job: Job) -> str:
    unread = job.unread()
    tail = f" | {unread} chars unread" if unread else ""
    return f"{job.id} · {job.status()} · {_fmt_duration(job.elapsed)} · {job.command[:70]}{tail}"


def _collect_new(job: Job, wait: float) -> tuple[str, int]:
    """Output nuovo del job, attendendo al massimo `wait` secondi.

    Continua ad attendere se ciò che è arrivato è solo spazi: l'output di un
    processo si materializza a FRAMMENTI (una singola `print` può arrivare in più
    write, tanto più con lo stdout del figlio non bufferato), e restituire un
    frammento bianco facendo avanzare il cursore vorrebbe dire annunciare "nessun
    output" mentre il comando sta scrivendo — e perdere quel pezzo per sempre."""
    out, lost = "", 0
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        chunk, dropped = job.take_new()
        out += chunk
        lost += dropped
        if out.strip() or not job.running:
            return out, lost
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return out, lost
        job.wait_for_activity(min(0.5, remaining))


def run_background_impl(ctx, command: str, cwd: str | None) -> str:
    """Avvia il comando e ritorna subito. La finestra di grazia serve a distinguere
    «avviato» da «morto all'istante»: senza, un comando con un typo risponderebbe
    "started" e l'errore si scoprirebbe solo al check successivo."""
    registry = getattr(ctx, "jobs", None)
    if registry is None:
        return _NO_JOBS
    try:
        job = registry.start(command, cwd)
    except RuntimeError as exc:
        return f"❌ {exc}"
    except OSError as exc:
        return f"❌ Could not start the command: {exc}"

    first, lost = _collect_new(job, getattr(ctx.cfg, "bg_start_grace", 1.5))
    head = f"✅ job {job.id} started: {command}"
    body = f"\n{first}" if first.strip() else ""
    if lost:
        # Un comando che riversa molto output all'istante può superare il tetto del
        # buffer già durante la finestra di grazia: scartare in silenzio significa
        # far credere al modello di aver visto tutto.
        body += (f"\n[{lost} chars of output were dropped: the buffer keeps the most "
                 f"recent {job.buffer_chars} chars]")
    if job.running:
        return (f"{head}{body}\n[still running — read new output with "
                f'job(action="check", id="{job.id}"); add wait_seconds to block up to '
                f'{getattr(ctx.cfg, "bg_max_wait", 30)}s]')
    return (f"{head}{body}\n[the command already finished with exit code {job.exit_code} "
            f"after {_fmt_duration(job.elapsed)} — it did not need the background]")


def job_impl(ctx, action: str, id: str = "", wait_seconds: int = 0) -> str:  # noqa: A002
    """check / list / stop sui job della sessione."""
    registry = getattr(ctx, "jobs", None)
    if registry is None:
        return _NO_JOBS
    registry.reap()
    act = (action or "").strip().lower()

    if act == "list":
        jobs = registry.all()
        evicted = getattr(registry, "evicted", 0)
        older = (f"\n  (+{evicted} older finished job(s) no longer tracked)" if evicted else "")
        if not jobs:
            return f"No background jobs in this session.{older}"
        # I job vivi per primi: è quello che serve sapere quando si decide se
        # aspettare o passare ad altro.
        ordered = sorted(jobs, key=lambda j: (j.running is False, j.started_at))
        running = sum(1 for j in jobs if j.running)
        body = "\n".join(f"  {_job_line(j)}" for j in ordered)
        return f"{len(jobs)} job(s), {running} still running:\n{body}{older}"

    if act not in ("check", "stop"):
        return ('❌ Unknown action: use "check" (read new output), "list" (all jobs) '
                'or "stop" (terminate one).')

    job = registry.get(id)
    if job is None:
        known = ", ".join(j.id for j in registry.all()) or "none"
        if getattr(registry, "evicted", 0):
            return (f"❌ Job '{id}' is no longer tracked: it had finished and was dropped to keep "
                    f"the list short (the {registry.keep_finished} most recent finished jobs are "
                    f"kept). Currently tracked: {known}.")
        return f"❌ No job with id '{id}'. Existing jobs: {known}."

    if act == "stop":
        was_running = job.running
        registry.stop(id)
        out, lost = job.take_new()
        note = "" if was_running else " (it had already finished)"
        lost_note = f"\n[{lost} chars of earlier output were dropped from the buffer]" if lost else ""
        tail = f"\n{out}" if out.strip() else ""
        return f"🛑 job {job.id} stopped{note}: {job.status()}{lost_note}{tail}"

    # check
    cap = getattr(ctx.cfg, "bg_max_wait", 30)
    wait = max(0, min(int(wait_seconds or 0), cap))
    out, lost = _collect_new(job, wait)
    parts = [_job_line(job)]
    if lost:
        parts.append(f"[{lost} chars of earlier output were dropped: the buffer keeps "
                     f"the most recent {job.buffer_chars} chars]")
    if out.strip():
        # Verbatim, senza strip: un frammento può finire a metà riga e il prossimo
        # continuarla — tagliare il newline finale incollerebbe le due righe.
        parts.append(f"--- new output ---\n{out}")
    elif job.running:
        parts.append("[no new output yet — it is still running]")
    if not job.running:
        parts.append("[job finished: nothing more will arrive]")
    return "\n".join(parts)
