"""Storage layer for codie's self-improving memory system.

Manages .codie/ directory with memory blocks, archival memory,
lessons, skills, and session summaries. All JSON-based, local files.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone

VALID_LABELS = ("persona", "user", "project")
DEFAULT_MAX_CHARS = 2000

DEFAULT_PERSONA = "I am codie, a coding agent. I learn and improve across sessions."


# ---------------------------------------------------------------------------
# Directory / config
# ---------------------------------------------------------------------------

def _codie_dir(cwd: str) -> str:
    return os.path.join(cwd, ".codie")


def ensure_codie_dir(cwd: str) -> str:
    """Create .codie/ and all subdirs if they don't exist. Idempotent."""
    base = _codie_dir(cwd)
    for sub in ("memory", "archival", "lessons", "skills", "sessions"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    # config.json
    config_path = os.path.join(base, "config.json")
    if not os.path.exists(config_path):
        _atomic_write(config_path, {
            "agent_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    # Default memory blocks
    for label in VALID_LABELS:
        bp = _block_path(cwd, label)
        if not os.path.exists(bp):
            default = DEFAULT_PERSONA if label == "persona" else ""
            _atomic_write(bp, {
                "label": label,
                "content": default,
                "max_chars": DEFAULT_MAX_CHARS,
            })

    # Empty archival + lessons
    for name in ("archival/entries.json", "lessons/entries.json"):
        p = os.path.join(base, name)
        if not os.path.exists(p):
            _atomic_write(p, [])

    return base


def get_agent_id(cwd: str) -> str:
    config_path = os.path.join(_codie_dir(cwd), "config.json")
    try:
        with open(config_path) as f:
            return json.load(f).get("agent_id", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

def _atomic_write(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


# ---------------------------------------------------------------------------
# Memory block CRUD
# ---------------------------------------------------------------------------

def _block_path(cwd: str, label: str) -> str:
    return os.path.join(_codie_dir(cwd), "memory", f"{label}.json")


def load_block(cwd: str, label: str) -> dict:
    """Load a single memory block. Returns {"label", "content", "max_chars"}."""
    data = _read_json(_block_path(cwd, label), {})
    return {
        "label": data.get("label", label),
        "content": data.get("content", ""),
        "max_chars": data.get("max_chars", DEFAULT_MAX_CHARS),
    }


def save_block(cwd: str, label: str, content: str, max_chars: int = DEFAULT_MAX_CHARS) -> None:
    _atomic_write(_block_path(cwd, label), {
        "label": label,
        "content": content,
        "max_chars": max_chars,
    })


def load_all_blocks(cwd: str) -> list[dict]:
    blocks = []
    for label in VALID_LABELS:
        blocks.append(load_block(cwd, label))
    return blocks


def append_to_block(cwd: str, label: str, content: str) -> str:
    """Append text to a block. Returns new content or error string."""
    if label not in VALID_LABELS:
        return f"[error: invalid label '{label}'. Must be one of: {', '.join(VALID_LABELS)}]"

    block = load_block(cwd, label)
    existing = block["content"]
    max_chars = block["max_chars"]

    # Add newline separator if block has content
    new_content = (existing + "\n" + content) if existing else content

    if len(new_content) > max_chars:
        # Try compaction first
        new_content = _compact_text(new_content, max_chars)

    if len(new_content) > max_chars:
        return f"[error: block '{label}' would exceed {max_chars} char limit ({len(new_content)} chars). Use core_memory_replace to update existing content first.]"

    save_block(cwd, label, new_content, max_chars)
    return new_content


def replace_in_block(cwd: str, label: str, old: str, new: str) -> str:
    """Replace text in a block. Returns new content or error string."""
    if label not in VALID_LABELS:
        return f"[error: invalid label '{label}'. Must be one of: {', '.join(VALID_LABELS)}]"

    block = load_block(cwd, label)
    content = block["content"]

    count = content.count(old)
    if count == 0:
        return f"[old_content not found in '{label}' block]"
    if count > 1:
        return f"[found {count} matches — must be unique. Add more context.]"

    new_content = content.replace(old, new, 1)

    if len(new_content) > block["max_chars"]:
        return f"[error: replacement would exceed {block['max_chars']} char limit]"

    save_block(cwd, label, new_content, block["max_chars"])
    return new_content


def compact_block(cwd: str, label: str) -> str:
    """Compact a block by deduplicating and trimming. Returns compacted content."""
    block = load_block(cwd, label)
    compacted = _compact_text(block["content"], block["max_chars"])
    save_block(cwd, label, compacted, block["max_chars"])
    return compacted


def _compact_text(text: str, max_chars: int) -> str:
    """Remove duplicate lines and trim oldest content if over limit."""
    lines = text.split("\n")

    # Remove exact duplicate lines (keep last occurrence)
    seen = set()
    deduped = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(line)
    deduped.reverse()

    result = "\n".join(deduped)

    # If still over limit, trim oldest lines (from top)
    while len(result) > max_chars and "\n" in result:
        result = result[result.index("\n") + 1:]

    # Last resort: hard truncate
    if len(result) > max_chars:
        result = result[:max_chars - 12] + "\n[compacted]"

    return result


# ---------------------------------------------------------------------------
# Archival memory
# ---------------------------------------------------------------------------

def _archival_path(cwd: str) -> str:
    return os.path.join(_codie_dir(cwd), "archival", "entries.json")


def archival_insert(cwd: str, content: str, source: str = "agent") -> dict:
    entries = _read_json(_archival_path(cwd), [])
    entry = {
        "id": str(uuid.uuid4()),
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    entries.append(entry)
    _atomic_write(_archival_path(cwd), entries)
    return entry


def archival_search(cwd: str, query: str, limit: int = 5) -> list[dict]:
    """Keyword search over archival entries. Score by word overlap."""
    entries = _read_json(_archival_path(cwd), [])
    if not entries:
        return []

    query_words = set(re.findall(r'\w+', query.lower()))
    if not query_words:
        return entries[:limit]

    scored = []
    for entry in entries:
        content_words = set(re.findall(r'\w+', entry.get("content", "").lower()))
        overlap = len(query_words & content_words)
        if overlap > 0:
            scored.append((overlap, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------

def _lessons_path(cwd: str) -> str:
    return os.path.join(_codie_dir(cwd), "lessons", "entries.json")


def add_lesson(cwd: str, content: str, verified: bool = False) -> dict:
    lessons = _read_json(_lessons_path(cwd), [])

    # Skip duplicates
    for existing in lessons:
        if existing.get("content") == content:
            return existing

    entry = {
        "id": str(uuid.uuid4()),
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verified": verified,
    }
    lessons.append(entry)
    _atomic_write(_lessons_path(cwd), lessons)
    return entry


def load_lessons(cwd: str) -> list[dict]:
    return _read_json(_lessons_path(cwd), [])


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def _skills_dir(cwd: str) -> str:
    return os.path.join(_codie_dir(cwd), "skills")


def save_skill(cwd: str, name: str, description: str, steps: str) -> str:
    """Save a skill as a .md file. Returns the file path."""
    # Sanitize name for filename
    safe_name = re.sub(r'[^\w\-]', '-', name.lower()).strip('-')
    if not safe_name:
        safe_name = "unnamed-skill"

    path = os.path.join(_skills_dir(cwd), f"{safe_name}.md")
    content = f"# {name}\n\n{description}\n\n## Steps\n\n{steps}\n"

    with open(path, "w") as f:
        f.write(content)

    return path


def search_skills(cwd: str, query: str) -> list[dict]:
    """Search skills by keyword matching on name + description."""
    skills = load_skills(cwd)
    if not skills or not query:
        return skills

    query_words = set(re.findall(r'\w+', query.lower()))
    scored = []
    for skill in skills:
        text = f"{skill['name']} {skill['description']}".lower()
        text_words = set(re.findall(r'\w+', text))
        overlap = len(query_words & text_words)
        if overlap > 0:
            scored.append((overlap, skill))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]


def load_skills(cwd: str) -> list[dict]:
    """Load all skills metadata from .md files."""
    skills_dir = _skills_dir(cwd)
    skills = []

    if not os.path.isdir(skills_dir):
        return skills

    for fname in sorted(os.listdir(skills_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(skills_dir, fname)
        try:
            with open(path) as f:
                content = f.read()
            # Parse name from first # heading
            lines = content.split("\n")
            name = lines[0].lstrip("# ").strip() if lines else fname[:-3]
            # Description is the text between title and ## Steps
            desc_lines = []
            for line in lines[1:]:
                if line.startswith("## "):
                    break
                if line.strip():
                    desc_lines.append(line.strip())
            description = " ".join(desc_lines)

            skills.append({
                "name": name,
                "description": description,
                "path": path,
            })
        except Exception:
            continue

    return skills


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _sessions_dir(cwd: str) -> str:
    return os.path.join(_codie_dir(cwd), "sessions")


def save_session_summary(cwd: str, session_id: str, summary: str) -> None:
    path = os.path.join(_sessions_dir(cwd), f"{session_id}.json")
    _atomic_write(path, {
        "session_id": session_id,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def load_recent_sessions(cwd: str, limit: int = 5) -> list[dict]:
    sdir = _sessions_dir(cwd)
    if not os.path.isdir(sdir):
        return []

    sessions = []
    for fname in sorted(os.listdir(sdir), reverse=True):
        if not fname.endswith(".json"):
            continue
        data = _read_json(os.path.join(sdir, fname), None)
        if data:
            sessions.append(data)
        if len(sessions) >= limit:
            break

    return sessions
