# codie

A self-improving CLI coding agent that learns from every session. Built on LangGraph + Claude.

Unlike traditional coding agents that start fresh every time, codie **remembers** your preferences, project patterns, and past mistakes — and applies them automatically in future sessions.

## Benchmark

Tested across 5 independent coding tasks, 3 coding standards scored per task:

| Metric | No Memory | With Memory | Delta |
|--------|-----------|-------------|-------|
| **Overall** | **27%** (4/15) | **100%** (15/15) | **+73%** |
| Type hints | 0/5 | 5/5 | +5 |
| Docstrings | 4/5 | 5/5 | +1 |
| Input validation | 0/5 | 5/5 | +5 |

The agent learns coding preferences in one session and applies them perfectly across 5 separate sessions — without being reminded.

```
uv run python3 benchmark.py    # Reproduce these results
```

## Install

```bash
git clone <repo-url> && cd codie
uv sync
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Usage

**Terminal 1 — Start the server:**
```bash
uv run langgraph dev --port 2025 --no-reload --no-browser
```

**Terminal 2 — Run codie (from any directory):**
```bash
codie
# or
uv run codie
```

### Commands

| Command | What it does |
|---------|-------------|
| `/init` | Explore the codebase and write initial project memories |
| `/remember` | Reflect on the conversation and save key learnings |
| `/memory` | Display current memory blocks |
| `/clear` | Clear screen |
| `/quit` | Save session learnings and exit |

## How Memory Works

codie's memory system lets the agent manage its own memory through tool calls. Memory is stored as local JSON files — no external databases, no vector stores, no APIs.

### Architecture

```
User (CLI) --> LangGraph Server --> Claude (ReAct Loop) --> 12 Tools
                                         |
                                    .codie/ (disk)
```

### Memory Layers

```
.codie/
  memory/
    persona.json     # Agent identity and approach
    user.json        # Your preferences and coding style
    project.json     # Codebase context and conventions
  archival/          # Long-term searchable facts
  lessons/           # Mistakes and corrections
  skills/            # Reusable multi-step procedures
  sessions/          # Past session summaries
```

**Core memory blocks** (persona, user, project) are injected into the system prompt at the start of every session. The agent always sees them. Max 2000 chars each with automatic compaction.

**Archival memory** is long-term storage the agent searches on demand. Good for facts that don't need to be in every prompt.

**Skills** are saved procedures (as markdown files) the agent can look up when solving similar problems.

### The Learning Loop

```
Session starts --> Memory blocks loaded into prompt
                   Agent sees what it learned before

User gives task --> Agent works (informed by memory)
                    Agent calls memory tools to save new knowledge

/remember       --> Agent reflects, distills key learnings

/quit           --> Post-session reflection saves lessons

Next session    --> Agent is smarter
```

### Memory Tools

The agent has 6 memory tools it controls:

| Tool | Purpose |
|------|---------|
| `core_memory_append` | Add to a persistent block (persona/user/project) |
| `core_memory_replace` | Edit text inside a block |
| `archival_memory_insert` | Save to long-term searchable storage |
| `archival_memory_search` | Keyword search over archived memories |
| `skill_save` | Save a reusable procedure |
| `skill_search` | Find saved procedures |

The agent decides **when** to use these — it's not forced. When a user states a preference, corrects a mistake, or when the agent discovers a pattern, it saves it.

### 6 Coding Tools

| Tool | Purpose |
|------|---------|
| `bash` | Run shell commands (tests, git, packages) |
| `read` | Read files with line numbers |
| `write` | Create or overwrite files |
| `edit` | Exact string replacement in files |
| `glob` | Find files by pattern |
| `grep` | Search file contents by regex |

## Trade-offs

**What works well:**
- Cross-session preference recall — the agent reliably applies learned coding standards
- Project context — after `/init`, the agent understands your codebase without re-exploring
- Skill reuse — saved procedures make repeated workflows faster
- Zero external dependencies — everything is local JSON files

**What doesn't work well:**
- Keyword-only search — archival memory uses word overlap, not semantic similarity. "deployment containers" won't match "deploy docker"
- No automatic learning from mistakes — the post-task reflection hook can't do blocking I/O in the async server context. Learning happens through explicit tool calls, `/remember`, and `/quit`
- Memory can go stale — if the codebase changes, saved project context may become wrong. Use `/init` to refresh
- Block size limit — 2000 chars per core block. Compaction removes duplicates and trims old content, but complex projects may hit limits

**What could be better with Mem0:**
- Semantic search over memories instead of keyword matching
- Automatic fact extraction from conversations
- Smarter memory ranking and relevance scoring
- Cross-project memory sharing

## Project Structure

```
agent/
  cli.py             # CLI client + commands
  server.py          # LangGraph agent factory + reflection hook
  prompt.py          # System prompt + memory injection
  tools.py           # 6 coding tools + 6 memory tools
  memory.py          # Storage layer (.codie/ CRUD)
  memory_tools.py    # Memory tool definitions
  reflection.py      # Post-task reflection logic
benchmark.py         # Style adherence benchmark
langgraph.json       # LangGraph deployment config
pyproject.toml       # Python project config
```

## Design Principles

- **Local-first** — all memory lives as JSON files on disk. No external databases, no cloud APIs, no vendor lock-in.
- **Agent-controlled** — the agent decides what to remember, not a separate system. Memory tools are just tools like `bash` or `read`.
- **Lightweight** — ~400 lines of memory code total. No complex infrastructure.
- **Transparent** — run `/memory` to see exactly what the agent knows. Edit `.codie/` files directly if needed.
