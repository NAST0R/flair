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
│   └── router.py        Agent selection: heuristics + sticky + LLM fallback (capped)
├── tools/
│   ├── fs.py            Filesystem helpers + resilient edit matcher (apply_edit)
│   ├── coding.py        Coding tools (sandboxed to the project root)
│   ├── system.py        Cross-platform desktop tools (whole machine)
│   └── web.py           Web search (Tavily / DuckDuckGo)
├── agents/
│   ├── coding.py        Builds the coding agent (+ project instructions)
│   └── general.py       Builds the general agent (+ web_search)
├── prompts/             System prompts (.md) + project-instructions loader
├── session_log.py       JSONL session log + file logging
└── cli.py               CLI + REPL (rich): streaming, diff preview, cost
```

**One engine, two agents.** `core/agent.py` is generic: it takes a *toolset* and a *prompt*. The two agents differ only in those — no duplicated logic.

- **Coding agent** — `read_file`, `list_directory`, `glob`, `grep`, `edit_file`, `write_file`, `run_command`. **Sandboxed** to the project root (`--root`): it cannot escape it.
- **General agent** — `open_url`, `open_path`, `open_application`, `search_files`, `list_directory`, `read_file`, `run_command`, `system_info`, `get_datetime`, `clipboard_get/set`, `web_search`. Operates on the **whole machine** (that is its purpose: "open the browser", "find a song"). It can also **converse**: if no tool is needed, it just answers.

**Safety.** Destructive tools (`edit_file`, `write_file`, `run_command`) ask for confirmation interactively, with a **diff preview**. `--yes` / `FLAIR_AUTO_APPROVE=true` disables it.

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
DEEPSEEK_MODEL=deepseek-v4-flash       # fast workhorse (non-thinking)
DEEPSEEK_THINK_MODEL=deepseek-v4-pro   # reasoner used for --think

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
> **DeepSeek** (primary provider): with the **V4** models the thinking mode is enabled by a *parameter*, not by a separate model name — Flair sends `thinking: {type: enabled}` automatically only on the `--think` step, on `deepseek-v4-pro` (a genuine reasoner). The fast loop stays on `deepseek-v4-flash` (non-thinking). The legacy aliases `deepseek-chat`/`deepseek-reasoner` still work but map to V4-flash (chat = non-thinking, reasoner = thinking) and **retire on 2026-07-24**, so the defaults use the V4 names. Note: because the old `deepseek-reasoner` was just V4-flash in thinking mode, its spend appeared under V4-flash — with `deepseek-v4-pro` as the think model you'll see it as a distinct line.
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
| `/provider` | show the active provider and models |
| `/cost` | session token/cost summary |
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
```

You can always invoke it as `python -m flair ...` too.

---

## Professional features

**Streaming.** The answer (and intermediate text) appears as it arrives. Disable it with `--no-stream` or `FLAIR_STREAM=false`.

**Resilient `edit_file`.** Matching `old_string` is not purely literal: it cascades through *exact → outer whitespace ignored → line-ending tolerant → indentation tolerant* (re-indenting the new block to the correct level automatically). If the match is not unique it returns a clear error inviting a re-read of the file, instead of failing opaquely. When it uses a fallback it says so (`[match: indentation tolerant]`).

**File creation.** `write_file` creates whole files and intermediate folders; `edit_file` makes targeted changes. The coding agent can therefore both **create** and **modify**.

**Diff preview + "always allow".** Before every destructive operation (when confirmations are on) Flair shows a **colored diff** of what will change (for `edit_file`/`write_file`) or the command (`run_command`). At the `[y]es / [n]o / [a]lways` prompt, `a` remembers the operation for the session and stops asking.

**Project instructions.** If the root contains an `AGENTS.md` (or `FLAIR.md`, `CLAUDE.md`, `.flair.md`) file, its content is loaded into the coding agent's prompt: conventions, build/test commands, constraints. `/root` reloads it on the fly.

**Web search (general agent).** The `web_search` tool works out of the box: the **`ddgs`** metasearch library is a bundled dependency, so key-free web search is available right after `pip install -e .` (it handles search engines' anti-bot protections — the most reliable key-free option). If `TAVILY_API_KEY` is set, Tavily is used first (most reliable overall). The tool also falls back to a best-effort DuckDuckGo scrape and the Instant Answer JSON API. Keyless engines can occasionally rate-limit automated requests; if a search returns empty, retry after a few seconds or set a Tavily key. On error it returns a clear message explaining the cause and the fix — never an exception.

**Session logging.** With `--log <folder>` (or `FLAIR_LOG_DIR`) every turn is written to `session-<timestamp>.jsonl` (task, response, tools used, usage) and internal events to `flair.log` — useful to analyze where tokens go.

**Per-model cost estimate.** The displayed cost uses a per-model price table (DeepSeek/OpenAI), overridable via `FLAIR_PRICE_*`.

---

## Token efficiency (without being stingy)

- **Prefix cache** always active (append-only history) → repeated input costs ~1/10 on both providers. The status line shows the cache-hit %.
- **Context compaction**: when the context exceeds a threshold (a fraction of the model window, default 75%), the older part is summarized into **one** message and the run continues with a new, stable prefix. You pay the cache miss **once per compaction**, not every turn — the opposite of the old Flair, which invalidated the cache every turn. Context size is measured exactly from the API's `prompt_tokens` (no tokenizer to install). If the provider still reports an overflow, Flair compacts aggressively and retries once.
- **Targeted thinking**: with `--think`, the reasoning model is used on the **first** step (planning); the tool loop continues with the fast, cheap model.
- **Targeted tool output**: files read with `offset`/`limit`, grep/commands truncated with a hint to narrow down.
- **Near-free router**: it decides by heuristics in most cases; it falls back to the LLM only when truly unsure, with the response capped to **a few tokens**.
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

Offline suite (no network, fake provider) with ~100 assertions covering: robust argument parsing, usage normalization for both providers, the **real provider request path** (parameters sent to the API: `max_tokens` vs `max_completion_tokens`, `temperature` omitted on reasoning models, retry on transient errors only), **streaming assembly**, **compaction** and overflow recovery, the resilient `edit_file` matcher, web search (parser + errors), the router, and **both** agents on the real tools.

```bash
python tests/test_smoke.py        # direct runner
pytest -q                         # alternatively (dev extra)
```

For lint and type-check (`dev` extra): `ruff check .` and `mypy flair`.

> Note: code, runtime messages, system prompts and the CLI are intentionally in Italian; this README and the repository framing are in English.

---

## License

Personal use.
