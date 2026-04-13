"""System prompt — dynamic context + memory injection."""

from __future__ import annotations

import asyncio
import os
import subprocess


SYSTEM_PROMPT = """\
You are a coding agent. You help users with software engineering tasks.

# Tools
- `bash` — Run shell commands (tests, git, packages). Do NOT use for file I/O.
- `read` — Read files with line numbers. Use this instead of `cat`.
- `write` — Create or overwrite files. Use this instead of `echo >` or `cat >`.
- `edit` — Edit files via exact string replacement. Use this instead of `sed`.
- `glob` — Find files by pattern. Use this instead of `find` or `ls`.
- `grep` — Search file contents by regex. Use this instead of `grep` in bash.
- `recall` — Search your memory from past sessions. Use BEFORE exploring. You might already know the answer.

IMPORTANT: Always prefer dedicated tools over bash. Read > cat, Edit > sed, Glob > find, Grep > grep.
IMPORTANT: Before exploring a codebase, call `recall` first to check if you already know the project structure, file locations, or user preferences from past sessions.

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
{memories}
"""


def _run_sync(cmd: str, cwd: str, timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _detect_context_sync(cwd: str) -> str:
    sections = []

    files = _run_sync("ls -1", cwd)
    if files:
        sections.append(f"Files:\n```\n{files}\n```")

    branch = _run_sync("git rev-parse --abbrev-ref HEAD 2>/dev/null", cwd)
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

    # Check for AGENTS.md
    agents_path = os.path.join(cwd, "AGENTS.md")
    if os.path.isfile(agents_path):
        try:
            with open(agents_path) as f:
                agents_md = f.read().strip()
            if agents_md:
                sections.append(f"Repo instructions:\n{agents_md}")
        except Exception:
            pass

    return "\n".join(sections)


async def build_system_prompt(cwd: str, memories: str = "") -> str:
    """Build system prompt with context + memories. Runs blocking I/O in a thread."""
    context = await asyncio.to_thread(_detect_context_sync, cwd)
    return (
        SYSTEM_PROMPT
        .replace("{cwd}", cwd)
        .replace("{context}", context)
        .replace("{memories}", memories)
    )
