# flair 3.0

An **agentic** AI assistant with two sides — **advanced coding** and **general desktop automation** — on **DeepSeek** *or* **OpenAI**, interchangeable. Runs on **Linux, macOS and Windows**.

It is the professional rewrite of the original project: same idea (an LLM that uses tools to do real things), but with a clean, token-efficient and maintainable architecture.

---

## Why it differs from the previous version

The original project fell into loops and burned tokens (a single analysis consumed ~850k tokens over 28 iterations). The causes, and how they are solved here:

| Original problem | Cause | Fix in flair 3.0 |
|---|---|---|
| Endless "name not found" loops | Indirection `analyze → guess name → read by name` | Direct tools: the model reads files instead of guessing their symbols |
| `cache_hit = 0`, ~20k tokens re-sent every turn | Compression heuristics **mutated** already-sent messages, invalidating the prefix cache | **Append-only** history: the head never changes → cache active on both providers |
| "Frankenstein" architecture (orchestrator + router + gates + budgets...) | Many patches fighting the model | **One** agentic engine, tools with co-located schema, minimal router |
| Drift between dispatch and tool schemas | Two parallel lists to sync by hand | `Tool` = function **+** schema in the same place (`@tool`) |

---

## Architecture

```
flair/
├── config.py            Config + nested ProviderConfig, loaded from .env
│                        (context window/compaction, streaming, logging, per-model pricing)
├── llm/                 LLM layer (provider abstraction)
│   ├── base.py          Types (Usage/LLMResponse/ToolCall), robust arg parsing,
│   │                    OpenAI-compatible provider (streaming, retry on transient
│   │                    errors only, normalized usage, overflow detector)
│   ├── deepseek.py      DeepSeek specifics (V4 thinking via parameter, max_tokens)
│   ├── openai.py        OpenAI specifics (o-series & GPT-5, max_completion_tokens)
│   └── factory.py       create_provider(cfg)
├── core/                Engine, independent of the concrete tools
│   ├── tool.py          Tool, Toolset, ToolContext, ToolError, @tool
│   ├── agent.py         Append-only agent loop + COMPACTION + anti-loop
│   └── router.py        Agent selection: continuation stickiness + LLM (capped) + heuristic fallback
├── tools/
│   ├── fs.py            Filesystem helpers + resilient edit matcher (apply_edit)
│   ├── coding.py        Coding tools (sandboxed to the project root)
│   ├── system.py        Cross-platform desktop tools (whole machine)
│   └── web.py           Web search + page fetch (Tavily / ddgs / DuckDuckGo)
├── agents/
│   ├── coding.py        Builds the coding agent (+ project instructions, web tools)
│   └── general.py       Builds the general agent (+ web_search/web_fetch)
├── prompts/             System prompts (.md) + project-instructions loader
├── session_log.py       JSONL session log + file logging
├── session_store.py     Save/resume conversation state across runs
└── cli.py               CLI + REPL (rich): streaming, diff preview, cost, sessions
```

**One engine, two agents.** `core/agent.py` is generic: it takes a *toolset* and a *prompt*. The two agents differ only in those — no duplicated logic.

- **Coding agent** — `read_file`, `list_directory`, `glob`, `grep` (with optional context lines and a files-only mode), `repo_map`, `edit_file`, `multi_edit`, `write_file`, `move_path`, `run_command`, `explore`, `plan`, `remember`, plus read-only `web_search` / `web_fetch` for information that lives online (library docs, API signatures, error messages). File tools are **sandboxed** to the project root (`--root`): it cannot escape it.
- **General agent** — `open_url`, `open_path`, `open_application`, `search_files`, `list_directory`, `read_file`, `write_file`, `edit_file`, `run_command`, `run_powershell`, `system_info`, `get_datetime`, `clipboard_get/set`, `web_search`, `web_fetch`. Operates on the **whole machine** (that is its purpose: "open the browser", "find a song", "write a report to disk"). It can also **converse**: if no tool is needed, it just answers.

For complex/multi-line PowerShell on Windows, the agent uses `run_powershell`: the script is written to a temporary file, executed with `-File`, and the temp file is **always removed** (success, error, or timeout) — no escaping headaches, no leftovers. Multi-line commands sent through `run_command` are routed the same way internally, instead of through cmd.exe (which breaks on embedded newlines).

**Safety.** Destructive tools (`edit_file`, `multi_edit`, `write_file`, `run_command`, `run_powershell`) ask for confirmation interactively, with a **diff preview**. `--yes` / `FLAIR_AUTO_APPROVE=true` disables it.

**Two providers, one interface.** DeepSeek and OpenAI both speak the OpenAI protocol; the differences (token parameter, reasoning models without `temperature`, cache fields, CoT) are isolated in two minimal subclasses. Adding a third provider = one file.

---

## Installation

Requires Python ≥ 3.10.

```bash
cd Flair
pip install -e .
# optional extras for the general agent (detailed RAM/CPU, portable clipboard):
pip install -e ".[extras]"
```

Then configure your keys:

```bash
cp .env.example .env
# edit .env: pick FLAIR_PROVIDER and set the matching API key
```

---

## Configuration (`.env`)

```env
FLAIR_PROVIDER=deepseek            # or: openai

DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-v4-flash       # fast workhorse for the tool loop
DEEPSEEK_THINK_MODEL=deepseek-v4-pro   # reasoner used for --think
DEEPSEEK_REASONING_EFFORT=max          # thinking depth for --think (ships enabled: measured best)

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini          # fast, non-reasoning, cheap
OPENAI_THINK_MODEL=gpt-5-mini      # reasoning for --think (Chat-Completions-compatible)
# OPENAI_REASONING_EFFORT=medium   # already 'medium' by default with --think

FLAIR_ROOT=.                       # working root for the coding agent
FLAIR_AUTO_APPROVE=false           # confirmation for destructive tools
```

All parameters (token limits, output caps, prices for cost estimation) are in `.env.example`, commented.

> Model names are only **defaults** and can be overridden from `.env` or the CLI.
>
> **DeepSeek** (primary provider): with the **V4** models the thinking mode is a *parameter*, not a separate model name — and it is **enabled by default server-side**, so even the fast loop on `deepseek-v4-flash` reasons at the server's default depth. On `--think` steps Flair makes it explicit on `deepseek-v4-pro` (a genuine reasoner) and sends `DEEPSEEK_REASONING_EFFORT` (ships as `max`: field-measured to be both the cheapest and the most precise setting). Flair follows the **V4 thinking protocol** for tool calls: the reasoning of tool-call turns is passed back to the API in subsequent requests, so the model resumes its chain of thought across steps instead of re-deriving it (providers that don't accept the field get it stripped at request time). Two opt-in deep regimes exist: `FLAIR_THINK_STEPS=all` keeps the reasoner for **every** step of a `--think` turn, and `DEEPSEEK_FAST_REASONING_EFFORT` raises the depth of the fast loop itself. The legacy aliases `deepseek-chat`/`deepseek-reasoner` still work but map to V4-flash (chat = non-thinking, reasoner = thinking) and **retire on 2026-07-24**, so the defaults use the V4 names. Note: because the old `deepseek-reasoner` was just V4-flash in thinking mode, its spend appeared under V4-flash — with `deepseek-v4-pro` as the think model you'll see it as a distinct line.
>
> **OpenAI**: reasoning models (the `o` series and the **GPT-5** family) are detected by name, so Flair omits `temperature` (which they reject) and sets `reasoning_effort` (default `medium` with `--think`). Prefer `gpt-5`/`gpt-5-mini`/`gpt-5.1` or `o3` as the think model: from **gpt-5.4 onward** reasoning + tools is offered only on the Responses API, not the Chat Completions API used here.

---

## Usage

### Interactive (REPL)

```bash
flair                          # auto-routes between coding and general
flair --provider openai
flair --root ~/projects/my-repo
```

REPL commands:

| Command | Effect |
|---|---|
| `/code <task>` | force the coding agent |
| `/do <task>` | force the general agent |
| `/think <task>` | use the thinking model on the first step |
| `/agent` | show the current agent |
| `/provider [name]` | show, or switch provider at runtime (`deepseek`/`openai`) |
| `/model <name>` | switch the fast model at runtime |
| `/think-model <name>` | switch the thinking model at runtime |
| `/compact` | compact the active agent's context now |
| `/cost` | session token/cost summary |
| `/save [name]` | save the session (defaults to the current name) |
| `/load <name>` | resume a saved session |
| `/sessions` | list saved sessions |
| `/reset` | clear the conversation |
| `/root <path>` | change the working root |
| `exit` | quit |

### One-shot

```bash
flair -p "open youtube for me"
flair -p "find the mp3 files in Music"
flair --agent coding -p "explain how authentication works" --root ~/app
flair --think -p "refactor the parsing module to reduce complexity"
flair --yes -p "run the tests and fix the errors"        # no confirmations
flair --no-stream -p "..."                                # disable streaming
flair --log ./logs -p "..."                               # write the session log (JSONL)
flair --session my-work                                   # use/create a session (auto-saved)
flair --continue                                          # resume the last saved session
flair --version                                           # print version
```

You can always invoke it as `python -m flair ...` too.

### Headless / scheduled (cron, systemd, CI)

Flair stays a one-shot command — the OS scheduler (cron, systemd timers, Task Scheduler, CI) handles *when* to run it; Flair handles *what* to do. The flags below make a `-p` run safe and machine-readable for unattended use; the interactive REPL is unchanged.

```bash
# Read-only report to JSON, with a hard cost ceiling, on a schedule:
flair -p "summarize today's changes under ./src" \
      --read-only --json --max-cost 0.10 --root /data >> out.jsonl

echo "audit the auth module" | flair -p - --read-only --quiet   # task from stdin
```

- **`--json`** prints exactly one JSON object on stdout (and nothing else): `ok`, `agent`, `stopped_reason`, `response`, `steps`, `truncated`, `usage`, `cost_usd`, `tools`, `files_changed`. On error or interruption it still emits a JSON object, so the contract is reliable. Human-oriented notes go to stderr.
- **Exit codes** reflect the outcome, not just success: `0` done, `2` max-steps, `3` loop, `4` stopped (something needed approval, or was interrupted), `5` budget exceeded, `1` error, `130` interrupted. A scheduler can branch on these.
- **`--read-only`** disables every destructive tool (write / edit / `run_command` / `run_powershell`) in both agents — the agent can read, search, browse and reason, but cannot change anything. Ideal for monitoring, reporting and audit jobs. (`FLAIR_READ_ONLY=true`.)
- **`--max-cost <usd>`** is a hard ceiling on the **session** cost: when the cumulative spend reaches it, the run stops (`stopped_reason: "budget"`) before the next paid call — no surprise bills. (`FLAIR_MAX_COST`.) Combined with the existing `--max-steps` (60 by default), runaway loops are bounded both ways.
- **`--quiet`** (`-q`) prints only the final answer. **`-p -`** reads the task from stdin.

For unattended runs prefer **stateless** invocations (no `--session`): two scheduled runs sharing one session file would race on it.

---

## Professional features

**Streaming.** The answer (and intermediate text) appears as it arrives. Disable it with `--no-stream` or `FLAIR_STREAM=false`.

**Parallel tool execution.** When the model requests several tools in one turn and they are **all read-only** (reads, searches, `repo_map`, `web_fetch`/`web_search`, `explore`), Flair runs them concurrently instead of one after another — a real latency win when the batch is I/O-bound (e.g. five 200 ms fetches finish in ~200 ms, not ~1 s). Safety is preserved by construction: batches containing any **destructive** tool (write / edit / `run_command`), or a single tool, stay strictly sequential, so the approval gate and write-ordering are never raced. The output is identical to sequential execution — same results, appended in the original tool-call order — only faster; workers run pure tool logic with an isolated context (no shared state, no interleaved output, exact usage accounting). Disable with `FLAIR_PARALLEL_TOOLS=false`; cap the worker count with `FLAIR_PARALLEL_TOOLS_MAX`.

**Session persistence.** Close Flair and pick up exactly where you left off. `--session <name>` uses (or creates) a named session that auto-saves after every turn; `--continue` resumes the most recent one. In the REPL, `/save`, `/load`, and `/sessions` manage them by hand. A session stores the full conversation of **both** agents plus cumulative usage, as JSON under `FLAIR_SESSION_DIR` (default `~/.flair/sessions`). Writes are **atomic** (temp file + rename), so a crash mid-write or two concurrent runs never leave a truncated file. Secrets are never saved — only chat messages.

**Session memory.** Durable facts, orthogonal to the conversation: the agent jots down non-obvious, long-lived knowledge (the project's test command, conventions, constraints, user preferences) with the `remember` tool — one concise line per note. Notes live in the **system prompt** next to the project instructions, i.e. in the stable, **cached** prefix: after the first call they cost cache-hit prices, and an empty memory injects nothing at all. Because compaction and pruning only ever touch the message history, memory is **lossless by construction** — a fact survives any number of compactions verbatim, and `/reset` keeps it too. Notes follow the session: `/save` persists them as a human-editable sidecar (`<name>.memory.md`) next to the session JSON, `/load` restores them. Guardrails are deterministic (no LLM calls): duplicate notes are refused, obvious credential patterns are rejected, and a **hard cap** (`FLAIR_MEMORY_MAX_CHARS`, ~1k tokens by default) refuses further notes instead of silently evicting — pruning is the user's call via `/memory` (list) and `/memory clear`, or by editing the sidecar. Disable everything with `FLAIR_MEMORY=false` (the tool is not even registered).

**Runtime switching.** Change provider or model mid-conversation without restarting: `/provider openai`, `/model <name>`, `/think-model <name>`. Histories are preserved; pricing re-aligns automatically.

**Context indicator + manual compaction.** After each turn the status line shows how full the active agent's context is (e.g. `contesto · coding: 23% (28k/120k)`). `/compact` summarizes older messages on demand to reclaim space (it also happens automatically near the threshold).

**Codebase map.** `repo_map` returns a compact outline of the project — for every source file, its top-level definitions (functions, classes and signatures) — in a single call. The model uses it to orient itself cheaply instead of issuing many `list_directory`/`grep`/`read_file` calls, which both **lowers token usage** on real repositories and improves navigation. It is always generated fresh from the current files (never stale), confined to the project root, and size-capped. Python is parsed with `ast` (accurate); around twenty other languages — JS/TS, Go, Rust, Java, C#, C/C++, Ruby, PHP, Swift, Kotlin, Scala, shell, Lua, Dart, Elixir, and more — are covered with per-language patterns, so it works on virtually any codebase.

**Read-only explorer sub-agent.** `explore` delegates a research question ("where and how is X implemented?", "which files handle Y?") to a **sub-agent with its own isolated context** and a **read-only** toolset (`repo_map`, `list_directory`, `glob`, `grep`, `read_file`, plus web). The sub-agent does the heavy reading in its own conversation and returns only a concise synthesis, so the parent agent's context stays lean — **lowering token usage** on large tasks while adding a focused-investigation capability. It is safe by construction: it cannot edit files or run commands, it cannot recurse (it does not have `explore` itself), it is bounded by `FLAIR_EXPLORER_MAX_STEPS`, and it is confined to the project root. Its token usage is rolled into the session total, so cost stays accurate. If the model never calls it, behaviour is unchanged.

**Explicit plan (`plan`).** For multi-step tasks the model opens with a short, structured TODO list and rewrites it as it goes (`da_fare` / `in_corso` / `fatto`). A visible plan is the standard countermeasure to *flailing* — the real token killer, where a task takes 25 steps instead of 12 — and improves reliability on long tasks. The tool is stateless (each call rewrites the full list), tolerant of model quirks (plain strings, English statuses, JSON-string arrays), capped in size, and shown in full in the CLI. The compaction summarizer is instructed to preserve the current plan and step states.

**Robust tool arguments.** Beyond coercing wrong types from each tool's schema and tolerantly dropping invented keys, a missing **required** argument now yields an *actionable* error instead of an opaque failure: it names what's missing and any ignored keys, and when one looks like the intended parameter it suggests it (e.g. `filename` → `path`). It only flags arguments the function genuinely requires (no default), so there are no false positives. This shortens the recovery when a model omits `path` or uses the wrong key name on a long edit — fewer wasted round-trips.

**Resilient `edit_file` / `multi_edit`.** Matching `old_string` is not purely literal: it cascades through *exact → outer whitespace ignored → line-ending tolerant → indentation tolerant* (re-indenting the new block to the correct level automatically). If the match is not unique it returns a clear error inviting a re-read of the file, instead of failing opaquely. When it uses a fallback it says so (`[match: indentation tolerant]`). `multi_edit` applies several edits to one file in a single, **atomic** call (if any edit fails, the file is left untouched) — fewer round-trips and tokens.

**File creation.** `write_file` creates whole files and intermediate folders; `edit_file` makes targeted changes. The coding agent can therefore both **create** and **modify**.

**Diff preview + "always allow".** Before every destructive operation (when confirmations are on) Flair shows a **colored diff** of what will change (for `edit_file`/`write_file`) or the command (`run_command`). At the `[y]es / [n]o / [a]lways / [s]top` prompt, `a` stops asking **for that tool** for the rest of the session (so a long run of commands isn't interrupted at every step), and `s` (or `Ctrl-C`) **stops the whole agentic flow** and returns control to you — the interruption is recorded in the conversation, so the agent knows exactly where it was stopped and can pick up from there on your next message. If an `edit_file` match would fail, the preview says so up front instead of showing an empty diff.

**Project instructions.** If the root contains an `AGENTS.md` (or `FLAIR.md`, `CLAUDE.md`, `.flair.md`) file, its content is loaded into the coding agent's prompt: conventions, build/test commands, constraints. `/root` reloads it on the fly.

**Web search & fetch (both agents).** The `web_search` tool works out of the box: the **`ddgs`** metasearch library is a bundled dependency, so key-free web search is available right after `pip install -e .` (it handles search engines' anti-bot protections — the most reliable key-free option). If `TAVILY_API_KEY` is set, Tavily is used first (most reliable overall). The tool also falls back to a best-effort DuckDuckGo scrape and the Instant Answer JSON API. Keyless engines can occasionally rate-limit automated requests; if a search returns empty, retry after a few seconds or set a Tavily key. The companion `web_fetch` tool downloads a page and returns its readable text, so the agent can actually **read** a result, not just list it. The general agent uses these for everyday lookups; the coding agent uses them to fill gaps it cannot infer from the project's files (treating the codebase as the source of truth). On error both return a clear message — never an exception.

**Session logging.** With `--log <folder>` (or `FLAIR_LOG_DIR`) every turn is written to `session-<timestamp>.jsonl` (task, response, tools used, usage) and internal events to `flair.log` — useful to analyze where tokens go.

**Model-aware cost tracking + budget warning.** Every request is priced with the rates of the model that actually served it (fast and thinking steps in the same turn are each billed correctly — validated against the provider dashboard) and accumulated into the displayed cost; the price table is overridable via `FLAIR_PRICE_*`, and the `FLAIR_MAX_COST` hard cap brakes on this real spend. Set `FLAIR_COST_WARN=<usd>` to get a one-time warning when a session's estimated cost crosses that threshold.

---

## Token efficiency (without being stingy)

- **Prefix cache** always active (append-only history) → repeated input costs ~1/10 on both providers. The status line shows the cache-hit %.
- **Context compaction**: when the context exceeds a threshold (a fraction of the model window, default 75%), the older part is summarized into **one** message and the run continues with a new, stable prefix. You pay the cache miss **once per compaction**, not every turn — unlike naive agent loops that invalidate the cache every turn. Context size is measured exactly from the API's `prompt_tokens` (no tokenizer to install). Before summarizing, a free deterministic **stage 0** prunes provably-superseded tool outputs — duplicates of identical calls, reads of files later overwritten by a full `write_file`, and partial reads covered by a later full read — replacing them with a short stub (tool-call pairing stays intact, the freshest occurrence always survives). If pruning alone brings the context back under the threshold, the LLM summary is skipped entirely: space is reclaimed **without** trading detail for a summary. Disable with `FLAIR_COMPACT_PRUNE=false`. If the provider still reports an overflow, Flair prunes, compacts aggressively and retries once.
- **Targeted thinking**: with `--think`, the reasoning model is used on the **first** step (planning); the tool loop continues with the fast, cheap model, **inheriting the preserved reasoning chain** (V4 passback) instead of re-deriving it. Opt-in: `FLAIR_THINK_STEPS=all` keeps the reasoner for the whole turn.
- **Targeted tool output**: files read with `offset`/`limit`, grep/commands truncated with a hint to narrow down.
- **Near-free router**: bare continuations ("procedi", "ok", "go ahead"…) stick to the current mode **deterministically, with zero LLM calls** — a contentless message carries no routing signal, and letting a model decide it can switch agent mid-task (different sandbox, prompt and tools) and invalidate the cached prefix. Everything else is decided by one tiny LLM call (response capped to **a few tokens**), with a keyword+sticky heuristic as offline fallback.
- **Resilient edit_file**: no "old_string not found" loops caused by whitespace/indentation (see above) → fewer wasted attempts.
- No aggressive budget: the only brake is the repeated-identical-call detector, which leads to a **clean finish** instead of a loop.

---

## Extending

**Add a tool** — a decorated function; the schema sits right next to it:

```python
from flair.core.tool import ToolContext, tool

@tool(
    "screenshot",
    "Capture a screenshot and save it.",
    {"type": "object",
     "properties": {"path": {"type": "string", "description": "Where to save the PNG."}},
     "required": ["path"]},
    destructive=False,
)
def screenshot(ctx: ToolContext, path: str) -> str:
    ...
    return f"✓ Saved {path}"
```

Then add it to the `TOOLS` list of the right module (`tools/coding.py`, `tools/system.py` or `tools/web.py`). Nothing else to touch: dispatch and schema are automatic.

**Add a provider** — subclass `OpenAICompatProvider` (set `token_param` and `reasoning_regex`) and register it in `llm/factory.py`.

---

## Tests

Offline suite (no network, fake provider) with ~535 assertions covering: robust argument parsing, usage normalization for both providers, the **real provider request path** (parameters sent to the API: `max_tokens` vs `max_completion_tokens`, `temperature` omitted on reasoning models, DeepSeek V4 thinking enabled via parameter, retry on transient errors only), **streaming assembly**, **compaction** and overflow recovery, the resilient `edit_file` matcher and **atomic `multi_edit`**, **actionable missing-argument errors** (naming the missing arg and suggesting the intended one, e.g. `filename`→`path`), **parallel tool execution** (ordered append despite out-of-order completion, exact delegated-usage accounting, destructive batches kept sequential, correct result association, on/off flag), the **`repo_map`** outline across ~two dozen languages, the **read-only `explore` sub-agent** (isolation, read-only toolset, no recursion, leak-proof usage roll-up), the **`plan`** tool and **stage-0 context pruning** (rules, guarantees, summary-skip), web **search** (multi-backend cascade + errors) and **fetch**, **session persistence** (save/resume round-trip and **atomic writes**, at both the store and CLI level), **session memory** (dedup, secret filtering, hard cap, prompt injection only at session boundaries, sidecar round-trip, `/reset` keeping notes, off-flag), **`grep` context/files-only modes** (merged adjacent blocks, match vs context markers, clamped context, coercion) and the root-confined **`move_path`** (deterministic no-overwrite semantics, directory moves, escape attempts blocked), **honest `read_file` headers** (declared range always equals delivered lines, continuation hint on every partial read) and the **mechanical read-inventory appended to compaction summaries** (partial markers, dedup, cap), **runtime provider/model switching**, the context indicator, the router (including deterministic continuation stickiness), **headless execution** (significant exit codes, the `--json` result object, read-only tool filtering, the hard cost budget), and **both** agents on the real tools.

```bash
python tests/test_smoke.py        # direct runner
pytest -q                         # alternatively (dev extra)
```

For lint and type-check (`dev` extra): `ruff check .` and `mypy flair`.

### Eval harness

`tests/evals/` runs end-to-end tasks against a **real** agent and checks the outcome deterministically, reporting pass/fail, steps, tokens and cache-hit per task — so a change can be measured rather than guessed. It needs the same API keys as flair.

```bash
python tests/evals/run_evals.py              # run all tasks (live)
python tests/evals/run_evals.py --list       # list tasks
python tests/evals/run_evals.py --self-test  # exercise the runner with no network
```

> Note: comments, docstrings and test labels are intentionally in Italian (a documented maintainer choice); everything user- and model-facing — system prompts, tool schemas and outputs, the CLI and runtime messages — is in English. The agent always replies in the user's language.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
