"""System prompt — matches Claude Code tone and behavior."""

from __future__ import annotations

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

IMPORTANT: Always prefer dedicated tools over bash. Read > cat, Edit > sed, Glob > find, Grep > grep.

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


def build_system_prompt(cwd: str) -> str:
    context = _detect_context(cwd)
    return SYSTEM_PROMPT.replace("{cwd}", cwd).replace("{context}", context)
