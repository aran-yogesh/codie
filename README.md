# codie

A self-improving coding agent. Gets faster and makes fewer mistakes over time by learning from every session.

Built on [LangGraph](https://github.com/langchain-ai/langgraph) + [Mem0](https://github.com/mem0ai/mem0) + Claude.

## How it works

```
SESSION START
  │
  ├── Ticker: consolidate stale memories (if due)
  │
BEFORE EACH TASK
  │
  ├── Recall: search Mem0 → MemRL re-rank → inject into system prompt
  │
AGENT RUNS (tool loop)
  │
  ├── Claude + 7 tools: bash, read, write, edit, glob, grep, recall
  │
AFTER EACH TASK
  │
  ├── Extract: Haiku extracts facts from compressed transcript
  ├── Retro-extract: on clean success, extracts "what made this work?"
  ├── MemRL update: reward signal updates Q-values of recalled memories
  └── Corrections: user "don't/stop/wrong" saved immediately
```

The agent learns repo facts, user preferences, and reusable patterns. Memories that lead to successful outcomes get ranked higher (MemRL). Stale memories decay and get consolidated.

## Install

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [Anthropic API key](https://console.anthropic.com/)
- A [Mem0 API key](https://app.mem0.ai/) (optional — agent works without it, just no memory)

### Setup

```bash
git clone <repo-url> && cd codie

# Create .env
cp .env.example .env
# Edit .env and add your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   MEM0_API_KEY=m0-...          (optional)

# Install dependencies
uv sync
```

### Global install (use `codie` from anywhere)

```bash
uv tool install .

# Now you can run from any directory:
codie
```

### Or use with `~/.codie/.env` for global API keys

```bash
mkdir -p ~/.codie
cp .env ~/.codie/.env
```

codie loads `~/.codie/.env` first, then `./env` in the current directory (local overrides global).

## Usage

### 1. Start the server (one terminal)

```bash
cd codie
uv run langgraph dev --port 2025
```

The LangGraph server handles the agent loop, tool execution, and thread persistence.

### 2. Run codie (another terminal)

```bash
# If installed globally:
codie

# Or run directly:
cd codie && uv run python -m agent.cli
```

```
┌─ codie — self-improving coding agent (memory on) ─┐
│  /quit     — exit                                  │
│  /clear    — clear screen                          │
│  /memories — show stored memories                  │
└────────────────────────────────────────────────────┘
/Users/you/my-project

> fix the auth bug in login.py

● recalled 3 memories (MemRL ranked)
● Recall(auth login middleware)
  ⎿ Auth middleware is in src/auth/middleware.py
● Read(src/auth/middleware.py)
  ⎿ Read 45 lines from src/auth/middleware.py
● Edit(src/auth/middleware.py)
  ⎿ Replaced 1 occurrence in src/auth/middleware.py
● Bash(pytest tests/test_auth.py -v)
  ⎿ 3 passed in 0.2s

Fixed missing token validation in middleware.py:42.

● utility update: reward=1.0 for 3 memories
● learned 2 facts
● retro-learned 1 pattern
```

### Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit |
| `/clear` | Clear screen |
| `/memories` | Show all stored memories |

## Architecture

```
codie/
├── agent/
│   ├── cli.py       — Interactive REPL, memory integration
│   ├── memory.py    — Mem0 wrapper, MemRL, extraction, ticker
│   ├── prompt.py    — Dynamic system prompt (context + memories)
│   ├── server.py    — LangGraph agent factory
│   └── tools.py     — bash, read, write, edit, glob, grep
├── benchmark.py     — Benchmark suite (measures improvement over time)
├── tests/
├── langgraph.json   — LangGraph server config
└── pyproject.toml
```

### Memory system

**Three memory types:**
- `repo_fact` — specific to a repository (test commands, file locations, frameworks)
- `user_pref` — user preferences (applies everywhere)
- `pattern` — reusable strategies (what worked, what failed)

**MemRL (Reinforcement Learning for Memory):**

Memories aren't just stored — they're scored. Each memory has a Q-value that tracks how useful it is:

```
Recall → Agent runs → Outcome (reward)
                         │
                         ▼
              Q_new = Q_old + α(R - Q_old)

Reward:
  1.0  = clean success (no errors)
  0.5  = success with errors along the way
  0.0  = no response
 -0.5  = user corrected the agent
```

Next time, memories are re-ranked by composite score: `(1-λ) × similarity + λ × Q-value`. Useful memories surface higher. Unhelpful ones sink.

**Ticker (consolidation):**

Runs on session start if 24h+ or 5+ sessions since last tick:
- Merges duplicate memories
- Resolves contradictions
- Promotes repeated facts into patterns
- Deletes stale memories

**Smart gates:**
- Skip recall on trivial tasks (< 4 words, greetings)
- Skip learning on simple sessions (< 3 tool calls, no errors)
- Compressed transcripts for extraction (~80% fewer tokens)

### Tools

| Tool | Description |
|------|-------------|
| `bash` | Run shell commands (tests, git, packages) |
| `read` | Read files with line numbers and ranges |
| `write` | Create or overwrite files |
| `edit` | Edit files via exact string replacement |
| `glob` | Find files by pattern |
| `grep` | Search file contents by regex |
| `recall` | Search memory from past sessions |

## Benchmark

Run the benchmark to measure improvement:

```bash
# Start server first, then:
uv run python benchmark.py
```

Run it twice — the second run should show fewer tool calls and faster completion because the agent remembers what it learned in run 1.

## Configuration

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `MEM0_API_KEY` | No | Mem0 API key (memory disabled without it) |
| `LANGGRAPH_API_URL` | No | Server URL (default: `http://localhost:2025`) |
| `LLM_MODEL_ID` | No | Model ID (default: `claude-sonnet-4-20250514`) |

### State files

codie stores local state in `~/.codie/`:

```
~/.codie/
├── .env              — Global API keys
├── state.json        — Ticker state (last tick time, session count)
└── q_values.json     — MemRL Q-values for each memory
```

## License

MIT
