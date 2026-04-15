"""Memory tools — core_memory, archival, skills. Agent calls these to manage its own memory."""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.config import get_config

from agent.memory import (
    ensure_codie_dir,
    append_to_block,
    replace_in_block,
    load_block,
    archival_insert,
    archival_search,
    save_skill as _save_skill,
    search_skills as _search_skills,
    VALID_LABELS,
)


def _get_cwd() -> str:
    config = get_config()
    return config.get("configurable", {}).get("cwd", "/tmp")


@tool
def core_memory_append(label: str, content: str) -> str:
    """Append content to a core memory block. Memory blocks persist across sessions and are always visible in your system prompt. Use this to save important context about the user, project, or yourself.

    Args:
        label: Memory block label — one of 'persona', 'user', 'project'.
        content: Text to append to the block (added on a new line).
    """
    cwd = _get_cwd()
    ensure_codie_dir(cwd)
    result = append_to_block(cwd, label, content)
    if result.startswith("[error"):
        return result
    return f"Updated '{label}' block. Current content ({len(result)} chars):\n{result}"


@tool
def core_memory_replace(label: str, old_content: str, new_content: str) -> str:
    """Replace text within a core memory block. The old_content must match exactly. Use this to update or correct information in your persistent memory.

    Args:
        label: Memory block label — one of 'persona', 'user', 'project'.
        old_content: Exact text to find in the block.
        new_content: Replacement text.
    """
    cwd = _get_cwd()
    ensure_codie_dir(cwd)
    result = replace_in_block(cwd, label, old_content, new_content)
    if result.startswith("["):
        return result
    return f"Replaced in '{label}' block. Current content ({len(result)} chars):\n{result}"


@tool
def archival_memory_insert(content: str) -> str:
    """Save information to long-term archival memory. Use for facts, lessons, patterns, and anything worth remembering that doesn't fit in core memory blocks. Archival memories are searchable but not always visible in your prompt.

    Args:
        content: The information to store.
    """
    cwd = _get_cwd()
    ensure_codie_dir(cwd)
    entry = archival_insert(cwd, content)
    return f"Saved to archival memory (id: {entry['id'][:8]})"


@tool
def archival_memory_search(query: str) -> str:
    """Search long-term archival memory by keywords. Returns relevant stored memories.

    Args:
        query: Keywords to search for.
    """
    cwd = _get_cwd()
    ensure_codie_dir(cwd)
    results = archival_search(cwd, query)
    if not results:
        return f"No archival memories matching '{query}'"

    lines = []
    for r in results:
        ts = r.get("timestamp", "")[:10]
        lines.append(f"[{ts}] {r['content']}")
    return "\n".join(lines)


@tool
def skill_save(name: str, description: str, steps: str) -> str:
    """Save a reusable skill (workflow/procedure) for future reference. Skills are searchable and listed in your prompt on future sessions.

    Args:
        name: Short skill name (e.g. 'add-api-endpoint', 'run-migrations').
        description: What the skill does and when to use it.
        steps: Step-by-step instructions (markdown format).
    """
    cwd = _get_cwd()
    ensure_codie_dir(cwd)
    path = _save_skill(cwd, name, description, steps)
    return f"Skill '{name}' saved to {path}"


@tool
def skill_search(query: str) -> str:
    """Search saved skills by keywords. Returns matching skill names and descriptions.

    Args:
        query: Keywords to search for.
    """
    cwd = _get_cwd()
    ensure_codie_dir(cwd)
    results = _search_skills(cwd, query)
    if not results:
        return f"No skills matching '{query}'"

    lines = []
    for s in results:
        lines.append(f"**{s['name']}**: {s['description']}")
    return "\n".join(lines)


MEMORY_TOOLS = [
    core_memory_append,
    core_memory_replace,
    archival_memory_insert,
    archival_memory_search,
    skill_save,
    skill_search,
]
