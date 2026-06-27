# 🤖 Free-Tier AI Orchestrator

> **The idea:** just create free API keys at the AI providers, drop them in, and
> let their **free models and free quotas collaborate** for the best possible
> *free* resolution of complex tasks.

An autonomous, **tool-using terminal agent** — a Claude-Code-style CLI (plus a
graphical web UI) — powered by **free/free-tier AI providers**. You type a goal;
it plans, reads and writes files, runs shell commands, checks its own work, and
keeps going until the goal is done. It picks the best available model for each
job by itself and makes several models **collaborate**.

No manual model picking. No paid plan required. Pure Python standard library
(`rich` is optional, only for nicer output).

### Bring your own free keys
Create a free key at any of these (one is enough; more = more capacity), then put
them in `.env` or your environment:

| Provider | Get a key | Env var |
|----------|-----------|---------|
| OpenRouter (many `:free` models) | https://openrouter.ai/keys | `OPENROUTER_API_KEY` |
| Google Gemini (free tier) | https://aistudio.google.com/apikey | `GEMINI_API_KEY` |
| Hugging Face | https://huggingface.co/settings/tokens | `HUGGINGFACE_API_KEY` |
| Together AI | https://api.together.ai/settings/api-keys | `TOGETHER_API_KEY` |
| Fireworks AI | https://fireworks.ai/account/api-keys | `FIREWORKS_API_KEY` |
| Replicate | https://replicate.com/account/api-tokens | `REPLICATE_API_TOKEN` |

The orchestrator discovers every free/free-tier model across the providers you've
keyed, and routes/rotates across them automatically (rate-limit aware), so the
free quotas add up.

---

## 🖥️ Graphical web UI (`ofo --web`)

```bash
ofo --web            # opens a local web app in your browser
ofo --web --web-port 9000
```

A self-contained, zero-dependency web app (stdlib `http.server` + Server-Sent
Events) that turns the CLI into a visual dashboard — easy to understand, nothing
to install:

- **Live activity stream** — every thought, tool call and observation animates in
  as the agent works, with per-tool icons (📄 read · ✏️ write · ⚡ bash · 🧠 consult
  · 🎚️ master_audio).
- **Approve from the browser** — file writes and shell commands pop an Approve/Deny
  modal; toggle **Auto-approve** for hands-free runs (destructive commands still ask).
- **Live sidebar** — providers, per-role routing, the learned model leaderboard, and
  a usage HUD (requests · cache hits · elapsed · models used).
- The agent operates on the folder you launched it in, sandboxed to it.

It runs on `127.0.0.1` only (not exposed to the network).

## How the agent works

It uses a strict **JSON-action protocol** (ReAct), so it works even with free
models that have no native function-calling:

```
You: "add a /health route to app.py and run the tests"

  [1] inspect the app          → read_file   {"path": "app.py"}
      observation: <file contents>
  [2] add the route            → edit_file   {"path": "app.py", "find": ..., "replace": ...}
      ↳ asks permission, applies the edit
  [3] verify it works          → run_bash    {"command": "pytest -q"}
      ↳ asks permission, observation: 5 passed
  [4] done                     → finish      {"summary": "added /health, tests green"}
      ↳ a *different* model verifies the goal is truly met before finishing
```

Each turn the model emits exactly one action:

```json
{"thought": "...", "tool": "read_file", "args": {"path": "app.py"}}
```

The driver model is **auto-selected** per call and **rotated** on rate-limits or
errors, so one throttled free model never stalls the run. Before the agent is
allowed to `finish`, a second model verifies the work against your goal — that
cross-model check is how the team pushes to actually *complete* the task instead
of stopping early.

---

## What's new in v1.3 — multi-provider free-tier routing

- **More providers.** OpenRouter, OpenAI, Anthropic Claude, Perplexity Sonar,
  Gemini API, Hugging Face Inference Providers, Together AI, Fireworks AI,
  Replicate and local Oobabooga can all be enabled from the CLI.
- **One autonomous router.** All providers feed one health-aware model registry;
  selection, rotation, cooldowns, ensemble diversity and the learned leaderboard
  work across providers.
- **Provider-specific adapters.** OpenAI, Fireworks, Together, Hugging Face and
  Oobabooga use OpenAI-compatible chat completions. Anthropic uses the native
  Messages API, Perplexity uses Sonar chat, Gemini uses `generateContent`, and
  Replicate uses predictions with polling.
- **Safe fallback behavior.** Bad keys, quota limits, 429s and unavailable
  models are isolated to that provider/model where possible, then the CLI rotates
  to another candidate.

## Robust, efficient, self-learning

- **Reliable file edits.** Lenient JSON parsing accepts the literal newlines weak
  models put in multi-line file content (the #1 cause of "invalid action" stalls),
  plus tolerant action parsing (alternate key names / OpenAI-style tool calls).
- **No more stalls.** The loop refuses to re-read unchanged files, detects
  repeating/no-progress behaviour, nudges the model, and — if it still can't
  proceed — returns a useful **partial summary** instead of aborting silently.
- **Cheaper.** A deterministic response cache dedupes identical low-temperature
  calls (verifier/critic), cutting free-tier requests; a per-turn usage meter
  shows requests, cache hits, time and models used.
- **Safer.** Every write/edit is backed up in-session; `/undo` reverts the last one.
- **Self-learning.** A persistent leaderboard (`~/.ofo/leaderboard.json`) tracks
  which models actually succeed and how fast, and biases selection toward them
  across sessions. See it with `--stats` or `/stats`.

## Setup

```bash
# 1. Add at least one provider key.
cp .env.example .env

# Supported env keys:
# OPENROUTER_API_KEY=...
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
# PERPLEXITY_API_KEY=...
# GEMINI_API_KEY=...
# HUGGINGFACE_API_KEY=...
# TOGETHER_API_KEY=...
# FIREWORKS_API_KEY=...
# REPLICATE_API_TOKEN=...
# OOBABOOGA_BASE_URL=http://127.0.0.1:5000/v1  # optional local provider

# 2. (optional) nicer output
pip install rich

# 3. Sanity-check without spending any requests:
./run.sh --self-test
```

Requires Python 3.8+. No other mandatory dependencies.

---

## Install it system-wide (use `ofo` from any folder)

```bash
./install.sh
```

This installs a tiny `ofo` launcher into `~/.local/bin`, copies your keys to a
global config (`~/.config/ofo/.env`, `chmod 600`) so they're found from any
directory, and makes sure `~/.local/bin` is on your `PATH`. Then, from anywhere:

```bash
cd ~/any/project
ofo "add a --json flag to cli.py and update the tests"   # operates on THIS folder
ofo                       # interactive shell here
ofo --list-models
```

- The agent always operates on your **current** directory (sandboxed to it).
- Key precedence: real env vars › `./​.env` (project-local) › `~/.config/ofo/.env`
  (global) › `~/.ofo/.env`. Set `OFO_ENV_FILE=/path/.env` to point somewhere else.
- Remove it again with `./install.sh --uninstall`.
- Prefer pip/pipx? `pipx install .` (or `pip install --user .`) also gives `ofo`;
  with that route, put your keys in `~/.config/ofo/.env` or your environment.

---

## Usage

```bash
./run.sh                                   # interactive agent shell (REPL)
./run.sh "create a Python CLI that renames files by date"   # run once, then exit
./run.sh --auto "scaffold a FastAPI app with a /health route and tests"
./run.sh --godmode "compare SQLite, DuckDB and Postgres for local analytics"
./run.sh --godmode-providers              # show GodMode webapp -> CLI provider mapping
./run.sh --list-models                     # show discovered models + routing
./run.sh --providers openrouter,gemini --list-models
```

Equivalently: `python3 -m orchestrator ...`.

### In the REPL
Type a goal, or use a slash-command:

| Command | Action |
|---------|--------|
| `/help` | list commands |
| `/tools` | show the tools the agent can use |
| `/models` | show selected models + routing |
| `/stats` | show the learned model leaderboard |
| `/godmode PROMPT` | send one prompt to several diverse models and synthesize |
| `/auto` | toggle auto-approve for writes & shell commands |
| `/undo` | revert the agent's last file write/edit |
| `/reset` | clear the conversation |
| `/exit` | quit |

### Tools the agent has
| Tool | Purpose | Needs approval |
|------|---------|----------------|
| `list_dir` / `read_file` | inspect the project | no (read-only) |
| `write_file` | create/overwrite a file | yes |
| `edit_file` | replace exact text in a file | yes |
| `run_bash` | run a shell command | yes (always for destructive ones) |
| `consult_models` | ask a diverse panel of models (default 8; `--consult-cap` / `max_models` to widen) then synthesize | yes |
| `master_audio` | create a TuneCore/streaming-ready WAV master with `ffmpeg` | yes |
| `finish` | declare the goal done | — |

### Useful flags
| Flag | Meaning |
|------|---------|
| `--auto` | auto-approve writes & commands (destructive commands still prompt) |
| `--no-bash` | disable the shell tool entirely |
| `--allow-outside` | permit file access outside the working dir (off by default) |
| `--root DIR` | run the agent in a different directory |
| `--max-steps N` | cap tool-actions per goal (default 25) |
| `--no-verify` | skip the cross-model completion check |
| `--role coder` | bias the driver toward coding models |
| `--max-requests N` | hard cap on total API calls (runaway guard, default 200) |
| `--providers a,b` | use only selected providers for this run |
| `--no-cache` | disable the deterministic response cache |
| `--no-stats` | don't read/write the cross-session model leaderboard |
| `--godmode` | send one prompt to several diverse models in parallel |
| `--godmode-width N` | choose how many models GodMode asks (default 4) |
| `--godmode-no-synthesis` | show the parallel answers without a merged answer |
| `--godmode-providers` | show how GodMode providers map to CLI adapters |
| `-o FILE` | also save the final summary to a file |

Info commands: `--list-models`, `--stats` (learned leaderboard), `--self-test`.

## GodMode-style fanout (`--godmode`)

The project also includes the core idea from
[smol-ai/GodMode](https://github.com/smol-ai/GodMode): one prompt goes to
multiple independent AI systems at the same time so you can compare their
answers. Here it is implemented over the existing provider API/router layer
instead of Electron webviews:

```bash
./run.sh --godmode "Review this migration plan for missing risks" -o review.md
./run.sh --godmode --godmode-width 6 --godmode-no-synthesis "Brainstorm names"
```

GodMode picks diverse healthy models from the current registry, runs them in
parallel, prints every model answer, and by default asks a reasoning model to
synthesize one final answer.

The CLI also maps the GodMode provider universe to API-backed access where that
is technically stable:

| GodMode source | CLI provider | Setup |
|----------------|--------------|-------|
| ChatGPT | `openai` | `OPENAI_API_KEY` |
| Claude / Claude 2 | `anthropic` | `ANTHROPIC_API_KEY` |
| Perplexity / Labs | `perplexity` | `PERPLEXITY_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Together | `together` | `TOGETHER_API_KEY` |
| HuggingChat | `huggingface` | `HUGGINGFACE_API_KEY` |
| Bard/Gemini | `gemini` | `GEMINI_API_KEY` |
| Oobabooga | `oobabooga` | `--providers oobabooga`, optional `OOBABOOGA_BASE_URL` |

Providers that GodMode reaches only through logged-in browser webapps, such as
Bing/Copilot, Poe, Phind, You.com, Pi and demo sites, are shown by
`--godmode-providers` as `webapp-only`. The CLI does not automate those browser
sessions because their DOM/login flows are brittle and not API-stable.

---

## Safety (the "safe autonomous" part)

- **Sandboxed files**: the agent can only touch the working directory unless you
  pass `--allow-outside`. Path traversal (`../`) is blocked.
- **Permission prompts**: every file write, edit and shell command asks first
  (like a coding assistant). `--auto` skips the routine ones, but **obviously
  destructive commands** (`rm -rf`, `mkfs`, `dd`, piping curl into a shell, …)
  *always* prompt, even with `--auto`.
- **Audio mastering**: `master_audio` uses local `ffmpeg`/`ffprobe` to create a
  44.1 kHz, 16-bit stereo WAV by default, with two-pass loudness normalization
  for TuneCore/streaming release prep. Install with `sudo apt install ffmpeg`
  if the tool reports missing binaries.
- **Model collaboration**: if you ask for all available models to collaborate,
  the agent can call `consult_models`. It fans out across the currently healthy
  model registry and is bounded by `--max-requests`, reserving budget for a
  synthesized consensus.
- **API key** is read from the environment / `.env`, never logged.
- **Rate-limit aware**: respects `Retry-After`, backs off with jitter, and
  rotates to other models instead of hammering one quota pool.
- **Free-tier aware**: OpenRouter models are filtered for zero prompt/completion
  price. Gemini defaults to Flash-family `generateContent` models. OpenAI,
  Anthropic, Perplexity, Fireworks, Together, Hugging Face and Replicate are
  treated as API/credit-backed providers when you supply a key; check each
  provider dashboard because the CLI cannot see whether credits have been
  exhausted or billing is enabled.
- **Hard budgets**: a global request cap (`--max-requests`), a per-goal step
  limit, and a per-command timeout make runaway impossible.
- **Bounded output**: tool results are size-capped so a huge file or command
  output can't blow up context or the terminal.
- **Honest reporting**: it won't claim "done" until a second model agrees the
  goal is met; otherwise it says so.

---

## Pipeline mode (`--pipeline`)

For pure content generation (specs, docs, analysis) where you want maximum
quality rather than tool use, the original multi-model pipeline is still here:

```bash
./run.sh --pipeline "Write a detailed migration plan from REST to gRPC" -o plan.md
./run.sh --pipeline --dry-run "..."     # show the plan only
```

It runs: **plan → (ensemble of diverse models + judge) → critic → integrate →
verify + refine**, and prints a critic-assessed confidence score.

---

## Project layout

```
orchestrator/
  config.py        settings, constants, .env loader
  client.py        std-lib multi-provider client/adapters
  registry.py      free/free-tier model discovery, scoring, health, selection
  router.py        per-role model selection + rotation (shared)
  godmode.py       GodMode-style parallel multi-model fanout + synthesis
  tools.py         sandboxed file/shell tools with permission gating
  agent.py         the interactive ReAct tool-use loop  ← the CLI "like Claude Code"
  repl.py          interactive shell + slash-commands
  agents.py        prompts + robust JSON parsers (planner/critic/judge/verifier)
  orchestrator.py  batch pipeline: plan → execute → aggregate → verify
  ui.py            terminal UI (rich if available, plain fallback)
  cli.py           argument parsing & main()
  selftest.py      offline tests (run with --self-test)
```

Tip: base URLs are overridable with `OPENROUTER_BASE_URL`, `OPENAI_BASE_URL`,
`ANTHROPIC_BASE_URL`, `PERPLEXITY_BASE_URL`, `FIREWORKS_BASE_URL`,
`TOGETHER_BASE_URL`, `HUGGINGFACE_BASE_URL`, `GEMINI_BASE_URL`,
`REPLICATE_BASE_URL` and `OOBABOOGA_BASE_URL`.

You can also set `OFO_PROVIDERS=openrouter,openai,anthropic,...` or pass
`--providers`.
If a provider's model-list endpoint is unavailable, `PROVIDER_MODELS` can add
manual fallbacks, for example `TOGETHER_MODELS=model|131072|Display Name`.

> The curated quality table in `registry.py` is only a *prior*; live provider
> catalogues and the learned leaderboard decide what the CLI tries first.
