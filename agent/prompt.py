"""System prompt — matches Claude Code tone and behavior."""

from __future__ import annotations

import subprocess

SYSTEM_PROMPT = """\
You are a coding agent. You help users with software engineering tasks.
You have persistent memory that survives across sessions — use it to learn and improve.

# Tools
- `bash` — Run shell commands (tests, git, packages). Do NOT use for file I/O.
- `read` — Read files with line numbers. Use this instead of `cat`.
- `write` — Create or overwrite files. Use this instead of `echo >` or `cat >`.
- `edit` — Edit files via exact string replacement. Use this instead of `sed`.
- `glob` — Find files by pattern. Use this instead of `find` or `ls`.
- `grep` — Search file contents by regex. Use this instead of `grep` in bash.

IMPORTANT: Always prefer dedicated tools over bash. Read > cat, Edit > sed, Glob > find, Grep > grep.

# Memory
You have persistent memory blocks shown below. Update them when you learn something important.

- `core_memory_append(label, content)` — add to a block (persona/user/project)
- `core_memory_replace(label, old_content, new_content)` — edit a block
- `archival_memory_insert(content)` — save to long-term searchable storage
- `archival_memory_search(query)` — search long-term memory
- `skill_save(name, description, steps)` — save a reusable procedure
- `skill_search(query)` — find saved procedures

When to update memory:
- User states a preference or corrects you → update user block
- You discover project patterns or conventions → update project block
- You solve a non-trivial problem → save as a skill or archival entry
- You make a mistake → save the lesson to archival memory

Be concise in memory. Integrate remembered context naturally.

{memory}

# Tone
- Go straight to the point. Lead with the answer, not the reasoning.
- Be concise. If you can say it in one sentence, don't use three.
- No emojis unless the user uses them.
- Don't list your capabilities. Just do the task.
- Don't ask clarifying questions if you can figure it out yourself.

# Workflow
1. Explore first — read relevant files before changing anything.
2. Make changes using edit/write.
3. Verify — run tests if available.
4. Report briefly — what you did, not what you could do.

# Rules
- Follow existing code style.
- Never touch .env or credential files.
- When referencing code, use file_path:line_number format.

# Working Directory
`{cwd}`

{context}
"""


def _run(cmd: str, cwd: str, timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _detect_context(cwd: str) -> str:
    sections = []

    files = _run("ls -1", cwd)
    if files:
        sections.append(f"Files:\n```\n{files}\n```")

    branch = _run("git rev-parse --abbrev-ref HEAD 2>/dev/null", cwd)
    if branch:
        sections.append(f"Git: `{branch}`")

    markers = {
        "package.json": "Node.js", "pyproject.toml": "Python",
        "requirements.txt": "Python", "Cargo.toml": "Rust", "go.mod": "Go",
    }
    for marker, lang in markers.items():
        if marker in (files or ""):
            sections.append(f"Language: {lang}")
            break

    agents_md = _run("cat AGENTS.md 2>/dev/null", cwd)
    if agents_md:
        sections.append(f"Repo instructions:\n{agents_md}")

    return "\n".join(sections)


def build_memory_section(cwd: str) -> str:
    """Load memory blocks + skills, format as XML for prompt injection."""
    try:
        from agent.memory import load_all_blocks, load_skills

        blocks = load_all_blocks(cwd)
        parts = ["<memory>"]
        for block in blocks:
            content = block["content"]
            if content:
                parts.append(f'<block label="{block["label"]}">')
                parts.append(content)
                parts.append("</block>")
        parts.append("</memory>")

        # List available skills
        skills = load_skills(cwd)
        if skills:
            parts.append("\n<skills>")
            for s in skills:
                parts.append(f"- {s['name']}: {s['description']}")
            parts.append("</skills>")

        return "\n".join(parts)
    except Exception:
        return ""


def build_system_prompt(cwd: str) -> str:
    context = _detect_context(cwd)
    memory = build_memory_section(cwd)
    return (SYSTEM_PROMPT
            .replace("{cwd}", cwd)
            .replace("{context}", context)
            .replace("{memory}", memory))
