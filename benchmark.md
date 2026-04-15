# Benchmark: Style Adherence

## What We Measure

Can the agent **learn coding preferences in one session and apply them in future sessions** without being reminded?

This is the core value proposition of a self-improving agent — it should get better at following your standards over time, not start from scratch every conversation.

## Setup

**3 coding standards** the agent must learn:
1. Always add type hints on every parameter and return type
2. Always include a docstring on every function
3. Always validate inputs and raise ValueError for invalid arguments

**5 coding tasks** (each in a separate session):
1. Reverse a linked list
2. Find the second largest number in a list
3. Convert celsius to fahrenheit
4. Count vowels in a string
5. Merge two sorted lists

**Scoring:** Each task is checked for all 3 standards. 1 point per standard met. Max score = 5 tasks x 3 standards = 15 points.

## How It Works

### Run 1: Without Memory (Baseline)

- All memory wiped (no `.codie/` directory)
- Agent gets no preferences, no history
- Each task runs in a **new session** (new thread)
- We score whatever the agent writes by default

### Run 2: With Memory

- All memory wiped for a clean start
- **Teaching phase:** Tell the agent the 3 standards and ask it to save them to memory using `core_memory_append`
- Verify the standards are saved to `.codie/memory/user.json`
- Each task runs in a **new session** (new thread) — the agent has zero conversation context from teaching
- The only way the agent knows the preferences is through its **persistent memory block** loaded into the system prompt

### Why Separate Sessions Matter

Each task runs in a completely new LangGraph thread. There is no shared conversation history between the teaching phase and the test tasks. The agent cannot "cheat" by looking at earlier messages — it can only know the user's preferences if it reads them from its memory block.

## Results

| Metric | No Memory | With Memory | Delta |
|--------|-----------|-------------|-------|
| **Overall** | **27%** (4/15) | **100%** (15/15) | **+73%** |
| Type hints | 0/5 | 5/5 | +5 |
| Docstrings | 4/5 | 5/5 | +1 |
| Input validation | 0/5 | 5/5 | +5 |

### What The Numbers Mean

- **27% baseline:** Without memory, the agent only writes docstrings by default (Claude's habit). It never adds type hints or input validation unprompted.
- **100% with memory:** After learning the 3 standards once, the agent applied all 3 perfectly across every task in every separate session.
- **+73% improvement:** This is entirely driven by persistent memory. The agent's behavior changed because of what it learned.

### Breakdown By Standard

**Type hints (0/5 -> 5/5):** The biggest gain. Without memory, Claude writes bare `def fibonacci(n):`. With memory, it writes `def fibonacci(n: int) -> int:` every time. The agent would never do this unprompted — it only does it because it remembers the user's standard.

**Docstrings (4/5 -> 5/5):** Claude already writes docstrings most of the time. Memory made it consistent (100% instead of 80%).

**Input validation (0/5 -> 5/5):** Without memory, functions have no input validation. With memory, every function includes `raise ValueError` for invalid inputs. Again — the agent would never add this unprompted.

## Use Cases This Proves

### 1. Coding Style Enforcement
Teams have standards — type hints, docstrings, error handling patterns, naming conventions. Instead of repeating these every session, tell the agent once. It remembers forever.

### 2. User Personalization
Different developers have different preferences. One prefers pytest, another prefers unittest. One wants verbose logging, another wants minimal output. The agent adapts to each user.

### 3. Cross-Session Consistency
Without memory, every session is a coin flip — the agent might or might not follow your style. With memory, it's deterministic. Same standards applied every time.

### 4. Onboarding Reduction
New team members spend time learning project conventions. A self-improving agent learns them once and applies them immediately — no repeated explanations needed.

## Reproduce

```bash
# Terminal 1: Start the server
uv run langgraph dev --port 2025 --no-reload --no-browser

# Terminal 2: Run the benchmark
uv run python3 benchmark.py
```

Takes about 5-10 minutes. Results saved to `benchmark_results.json`.
