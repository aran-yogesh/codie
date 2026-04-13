"""Mem0-backed memory — recall, learn, tick. Falls back to no-op if not configured."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time

from anthropic import Anthropic

logger = logging.getLogger(__name__)

try:
    from mem0 import MemoryClient

    _HAS_MEM0 = True
except ImportError:
    _HAS_MEM0 = False
    MemoryClient = None

_client = None
_llm = None

EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

# ─── State + MemRL constants ─────────────────────────────────────────

STATE_DIR = os.path.expanduser("~/.codie")
STATE_FILE = os.path.join(STATE_DIR, "state.json")

Q_INIT = 0.5          # Initial Q-value for new memories
Q_ALPHA = 0.1         # Learning rate for Q updates
Q_LAMBDA = 0.5        # Balance: similarity vs utility in retrieval
Q_FILE = os.path.join(STATE_DIR, "q_values.json")


def _read_q_values() -> dict:
    """Read Q-value store from disk. Maps memory_id → {q, used, helped}."""
    try:
        with open(Q_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_q_values(qv: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(Q_FILE, "w") as f:
        json.dump(qv, f)


def _get_q(memory_id: str) -> float:
    """Get Q-value for a memory. Returns Q_INIT if not tracked yet."""
    qv = _read_q_values()
    entry = qv.get(memory_id, {})
    return entry.get("q", Q_INIT)


def _read_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_state(state: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ─── Repo identification ────────────────────────────────────────────


def get_repo_id(cwd: str) -> str:
    """Get a stable repo identifier from git remote or path."""
    try:
        r = subprocess.run(
            "git remote get-url origin 2>/dev/null",
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return cwd


# ─── Clients ────────────────────────────────────────────────────────


def _get_client():
    global _client
    if not _HAS_MEM0:
        return None
    if _client is None:
        api_key = os.environ.get("MEM0_API_KEY", "")
        if not api_key:
            return None
        _client = MemoryClient(api_key=api_key)
    return _client


def _get_llm():
    global _llm
    if _llm is None:
        _llm = Anthropic()
    return _llm


def is_enabled() -> bool:
    return _get_client() is not None


# ─── Smart gates ────────────────────────────────────────────────────


def should_recall(task: str) -> bool:
    """Only recall for non-trivial tasks."""
    if len(task.split()) < 4:
        return False
    trivial = ["hello", "hi", "hey", "thanks", "what is", "explain", "help"]
    lower = task.lower()
    return not any(lower.startswith(t) for t in trivial)


def should_learn(messages: list) -> bool:
    """Only learn when something interesting happened."""
    tool_calls = 0
    has_errors = False

    for msg in messages:
        msg_type = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        if msg_type == "ai":
            tc = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
            tool_calls += len(tc or [])
            if isinstance(content, list):
                tool_calls += sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")

        elif msg_type == "tool":
            text = _extract_text(content)
            if any(e in text for e in ["[error", "[file not found", "[old_string not found", "[exit code:", "[timed out"]):
                has_errors = True

    return tool_calls >= 3 or has_errors


# ─── Correction detection ───────────────────────────────────────────

CORRECTION_SIGNALS = [
    "don't", "dont", "stop", "wrong", "no,", "no ",
    "actually", "instead", "never", "please use", "always use",
]


def detect_correction(user_input: str) -> str | None:
    """Check if user input is a correction. Returns the text or None."""
    lower = user_input.lower()
    for signal in CORRECTION_SIGNALS:
        if signal in lower:
            return user_input[:300]
    return None


def learn_correction(correction: str, repo_id: str):
    """Save a user correction to Mem0 immediately."""
    client = _get_client()
    if client is None:
        return
    try:
        client.add(
            f"User correction: {correction}",
            user_id="codie",
            metadata={"type": "user_pref", "repo": repo_id, "source": "correction"},
        )
    except Exception as e:
        logger.warning("Failed to store correction: %s", e)


# ─── RECALL (before task) ───────────────────────────────────────────


def recall(query: str, repo_id: str = "", limit: int = 5) -> tuple[list[str], list[str]]:
    """Two-Phase Retrieval (MemRL).

    Phase A: Semantic search via Mem0 → top candidates by similarity.
    Phase B: Re-rank by composite score = (1-λ)×similarity + λ×Q-value.

    Returns: (memory_texts, memory_ids) — IDs needed for utility update later.
    """
    client = _get_client()
    if client is None:
        return [], []

    try:
        # Phase A: semantic recall — get more candidates than we need
        k1 = limit * 2
        results = client.search(
            query,
            user_id="codie",
            limit=k1,
            filters={"AND": [{"user_id": "codie"}]},
        )
        items = results.get("results", []) if isinstance(results, dict) else results

        # Filter by repo
        candidates = []
        for m in items:
            if not isinstance(m, dict):
                continue
            text = m.get("memory", str(m))
            mem_id = m.get("id", "")
            meta = m.get("metadata", {}) or {}
            mem_repo = meta.get("repo", "")
            mem_type = meta.get("type", "")
            score = m.get("score", 0.5)  # similarity score from Mem0

            # Strict repo scoping:
            # - Same repo → always include
            # - user_pref → include (global, applies everywhere)
            # - pattern → include (reusable strategy)
            # - Different repo + repo_fact → exclude (not relevant here)
            if mem_repo and mem_repo != repo_id and mem_type not in ("user_pref", "pattern"):
                continue

            candidates.append({
                "text": text,
                "id": mem_id,
                "similarity": score,
                "q_value": _get_q(mem_id) if mem_id else Q_INIT,
            })

        if not candidates:
            return [], []

        # Phase B: re-rank by composite score
        for c in candidates:
            c["composite"] = (1 - Q_LAMBDA) * c["similarity"] + Q_LAMBDA * c["q_value"]

        candidates.sort(key=lambda c: c["composite"], reverse=True)
        top = candidates[:limit]

        return [c["text"] for c in top], [c["id"] for c in top if c["id"]]

    except Exception as e:
        logger.warning("Memory recall failed: %s", e)
        return [], []


def format_for_system_prompt(memories: list[str]) -> str:
    """Format memories as system prompt instructions (not user message)."""
    if not memories:
        return ""
    lines = ["\n# What I Know From Past Sessions"]
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. {m}")
    return "\n".join(lines)


# Keep old name for backwards compat
format_for_injection = format_for_system_prompt


# ─── LEARN (after task) ─────────────────────────────────────────────


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def _compress_transcript(messages: list) -> str:
    """Compressed transcript — only important messages, saves ~80% tokens."""
    important = []
    first_human = True

    for msg in messages:
        msg_type = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        text = _extract_text(content)
        if not text.strip():
            continue

        if msg_type == "human" and first_human:
            important.append(f"TASK: {text[:300]}")
            first_human = False
        elif msg_type == "human":
            important.append(f"USER: {text[:200]}")
        elif msg_type == "tool":
            if any(e in text for e in ["[error", "[file not found", "[exit code:", "[timed out"]):
                important.append(f"TOOL ERROR: {text[:200]}")
        elif msg_type == "ai":
            tc = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
            if tc:
                names = [t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "") for t in tc]
                important.append(f"AGENT USED: {', '.join(names)}")

    # Add final AI response
    for msg in reversed(messages):
        msg_type = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
        if msg_type == "ai":
            text = _extract_text(msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", ""))
            if text.strip():
                important.append(f"RESULT: {text[:300]}")
                break

    return "\n".join(important)


EXTRACTION_PROMPT = """\
You are a memory extraction agent for a coding agent called "codie".
Given a compressed transcript, extract facts worth remembering for future sessions.

Classify each fact:
- repo_fact: specific to this repository (test commands, file locations, frameworks)
- user_pref: how the user wants things done (applies everywhere)
- pattern: a reusable strategy that worked (or anti-pattern that failed)

Rules:
- One short sentence per fact.
- Only extract what's useful in a FUTURE session.
- Skip obvious things and temporary state.
- Max 5 facts. Empty list if nothing worth remembering.

Return ONLY a JSON array: [{"type": "repo_fact", "text": "..."}, ...]
"""


def extract_facts(messages: list) -> list[dict]:
    """Use Haiku to extract structured facts from the conversation."""
    transcript = _compress_transcript(messages)
    if not transcript.strip():
        return []

    try:
        llm = _get_llm()
        response = llm.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=500,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": transcript}],
        )
        text = response.content[0].text.strip()

        # Handle markdown code blocks
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)

        facts = json.loads(text)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, dict) and "text" in f][:5]
        return []
    except Exception as e:
        logger.warning("Fact extraction failed: %s", e)
        return []


RETRO_EXTRACTION_PROMPT = """\
You are a memory extraction agent reviewing a SUCCESSFUL coding task.
This task completed with no errors. Your job is to extract facts that explain WHY it succeeded — what context, patterns, or decisions made it work.

Focus on:
- **shortcuts**: What did the agent already know that saved time?
- **decisions**: What approach was chosen and why was it the right one?
- **reusable patterns**: What strategy here would work for similar tasks?
- **pitfalls avoided**: What could have gone wrong but didn't?

Do NOT repeat facts that are obvious from the task description.
Do NOT extract what was done — extract what made it work.

Max 3 facts. Empty list if nothing insightful.

Return ONLY a JSON array: [{"type": "pattern", "text": "..."}, ...]
"""


def extract_retro_facts(messages: list) -> list[dict]:
    """Retroactive extraction — runs on successful tasks to catch what the first pass missed.
    Asks 'what made this succeed?' instead of 'what happened?'"""
    transcript = _compress_transcript(messages)
    if not transcript.strip():
        return []

    try:
        llm = _get_llm()
        response = llm.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=400,
            system=RETRO_EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": f"This task succeeded cleanly.\n\n{transcript}"}],
        )
        text = response.content[0].text.strip()

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)

        facts = json.loads(text)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, dict) and "text" in f][:3]
        return []
    except Exception as e:
        logger.warning("Retro extraction failed: %s", e)
        return []


def learn(facts: list[dict], repo_id: str = "") -> int:
    """Store extracted facts in Mem0 with metadata. Returns count stored."""
    client = _get_client()
    if client is None:
        return 0

    stored = 0
    for fact in facts:
        try:
            text = fact.get("text", str(fact)) if isinstance(fact, dict) else str(fact)
            fact_type = fact.get("type", "repo_fact") if isinstance(fact, dict) else "repo_fact"
            client.add(
                text,
                user_id="codie",
                metadata={"type": fact_type, "repo": repo_id, "source": "extracted"},
            )
            stored += 1
        except Exception as e:
            logger.warning("Failed to store fact: %s", e)

    return stored


# ─── MemRL: REWARD + UTILITY UPDATE ───────────────────────────────────


def compute_reward(messages: list) -> float:
    """Compute reward from task outcome.

    1.0  = clean success (no errors)
    0.5  = success with some errors along the way
    0.0  = failed (no meaningful response)
    -0.5 = user corrected the agent
    """
    has_response = False
    tool_errors = 0
    tool_calls = 0
    has_correction = False

    for msg in messages:
        msg_type = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        text = _extract_text(content)

        if msg_type == "ai" and text.strip():
            has_response = True
            tc = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
            tool_calls += len(tc or [])

        elif msg_type == "tool":
            if any(e in text for e in ["[error", "[file not found", "[old_string not found", "[exit code:", "[timed out"]):
                tool_errors += 1

        elif msg_type == "human":
            if detect_correction(text):
                has_correction = True

    if has_correction:
        return -0.5
    if not has_response:
        return 0.0
    if tool_errors > 0:
        return 0.5
    return 1.0


def update_utility(memory_ids: list[str], reward: float):
    """MemRL utility update: Q_new = Q_old + α(R - Q_old) for each recalled memory."""
    if not memory_ids:
        return

    qv = _read_q_values()

    for mid in memory_ids:
        if not mid:
            continue
        entry = qv.get(mid, {"q": Q_INIT, "used": 0, "helped": 0})
        old_q = entry["q"]
        new_q = old_q + Q_ALPHA * (reward - old_q)
        entry["q"] = round(new_q, 4)
        entry["used"] = entry.get("used", 0) + 1
        if reward > 0.5:
            entry["helped"] = entry.get("helped", 0) + 1
        qv[mid] = entry

    _write_q_values(qv)


# ─── LIST ────────────────────────────────────────────────────────────


def get_all(user_id: str = "codie") -> list[str]:
    """Get all stored memories."""
    try:
        client = _get_client()
        if client is None:
            return []
        results = client.get_all(user_id=user_id, filters={"AND": [{"user_id": user_id}]})
        items = results.get("results", []) if isinstance(results, dict) else results
        return [m.get("memory", str(m)) if isinstance(m, dict) else str(m) for m in items]
    except Exception as e:
        logger.warning("Memory get_all failed: %s", e)
        return []


# ─── TICKER (consolidation) ─────────────────────────────────────────

TICK_PROMPT = """\
Review these memories stored by a coding agent and clean them up.

Current memories:
{memories}

Tasks:
1. MERGE duplicates into single, better-worded memories
2. RESOLVE contradictions (keep newer/more specific)
3. PROMOTE repeated facts into reusable patterns
4. FLAG stale/useless memories for deletion

Return JSON only:
{{"keep": ["text to keep"], "add": ["new merged memories"], "delete": ["exact text to remove"], "summary": "one sentence"}}
"""


def should_tick() -> bool:
    """Check if it's time to run the ticker."""
    state = _read_state()
    last_tick = state.get("last_tick", 0)
    session_count = state.get("sessions_since_tick", 0)
    hours_since = (time.time() - last_tick) / 3600
    return hours_since >= 24 or session_count >= 5


def bump_session_count():
    """Increment session counter (called on each session start)."""
    state = _read_state()
    state["sessions_since_tick"] = state.get("sessions_since_tick", 0) + 1
    _write_state(state)


def tick() -> str | None:
    """Run memory consolidation. Returns summary or None."""
    client = _get_client()
    if client is None:
        return None

    try:
        results = client.get_all(user_id="codie", filters={"AND": [{"user_id": "codie"}]})
        items = results.get("results", []) if isinstance(results, dict) else results
        if len(items) < 3:
            _mark_ticked()
            return None

        memories_text = "\n".join(
            f"- {m.get('memory', str(m))}" if isinstance(m, dict) else f"- {m}"
            for m in items
        )

        llm = _get_llm()
        response = llm.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=1000,
            system="You are a memory consolidation agent. Return valid JSON only.",
            messages=[{"role": "user", "content": TICK_PROMPT.format(memories=memories_text)}],
        )
        text = response.content[0].text.strip()

        # Handle markdown code blocks
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        result = json.loads(text)

        to_delete = result.get("delete", [])
        if to_delete:
            for item in items:
                mem_text = item.get("memory", "") if isinstance(item, dict) else str(item)
                mem_id = item.get("id", "") if isinstance(item, dict) else ""
                if mem_text in to_delete and mem_id:
                    try:
                        client.delete(mem_id)
                    except Exception:
                        pass

        for mem_text in result.get("add", []):
            try:
                client.add(mem_text, user_id="codie", metadata={"source": "consolidated"})
            except Exception:
                pass

        _mark_ticked()
        return result.get("summary", "consolidated")

    except Exception as e:
        logger.warning("Tick failed: %s", e)
        _mark_ticked()
        return None


def _mark_ticked():
    state = _read_state()
    state["last_tick"] = time.time()
    state["sessions_since_tick"] = 0
    _write_state(state)
