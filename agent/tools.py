"""Agent tools — Bash, Read, Write, Edit, Glob, Grep."""

from __future__ import annotations

import fnmatch
import os
import subprocess

from langchain_core.tools import tool
from langgraph.config import get_config


def _get_cwd() -> str:
    config = get_config()
    return config.get("configurable", {}).get("cwd", "/tmp")


def _resolve(path: str, cwd: str) -> str:
    return path if os.path.isabs(path) else os.path.join(cwd, path)


@tool
def bash(command: str) -> str:
    """Execute a shell command. Use for running tests, git, installing packages, etc. Do NOT use for reading or editing files — use Read/Edit/Write instead."""
    cwd = _get_cwd()
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=120,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[timed out after 120s]"
    except Exception as e:
        return f"[error: {e}]"


@tool
def read(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read a file with line numbers. Use start_line/end_line to read specific ranges (1-indexed). end_line=0 reads to end.

    Args:
        path: File path relative to working directory or absolute.
        start_line: First line to read (1-indexed).
        end_line: Last line to read (0 = to end).
    """
    cwd = _get_cwd()
    full = _resolve(path, cwd)
    try:
        with open(full) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"[file not found: {path}]"
    except Exception as e:
        return f"[error: {e}]"

    total = len(lines)
    s = max(1, start_line) - 1
    e = total if end_line <= 0 else min(end_line, total)
    selected = lines[s:e]
    numbered = [f"{i + s + 1:4d} | {line}" for i, line in enumerate(selected)]
    return f"[{path}] lines {s + 1}-{e} of {total}\n" + "".join(numbered)


@tool
def write(path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Args:
        path: File path relative to working directory or absolute.
        content: The full file content to write.
    """
    cwd = _get_cwd()
    full = _resolve(path, cwd)

    # Don't touch sensitive files
    basename = os.path.basename(full)
    if basename in (".env", ".env.local", "credentials.json", "secrets.yaml"):
        return f"[refused: won't write to {basename}]"

    try:
        os.makedirs(os.path.dirname(full), exist_ok=True) if os.path.dirname(full) else None
        with open(full, "w") as f:
            f.write(content)
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Wrote {line_count} lines to {path}"
    except Exception as e:
        return f"[error writing {path}: {e}]"


@tool
def edit(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing old_string with new_string. The old_string must match exactly (including whitespace). Must be unique in the file.

    Args:
        path: File path relative to working directory or absolute.
        old_string: Exact text to find.
        new_string: Replacement text.
    """
    cwd = _get_cwd()
    full = _resolve(path, cwd)
    try:
        with open(full) as f:
            text = f.read()
    except FileNotFoundError:
        return f"[file not found: {path}]"
    except Exception as e:
        return f"[error: {e}]"

    count = text.count(old_string)
    if count == 0:
        return f"[old_string not found in {path}]"
    if count > 1:
        return f"[found {count} matches — must be unique. Add more surrounding context.]"

    new_text = text.replace(old_string, new_string, 1)
    try:
        with open(full, "w") as f:
            f.write(new_text)
    except Exception as e:
        return f"[error writing {path}: {e}]"
    return f"Replaced 1 occurrence in {path}"


@tool
def glob(pattern: str, path: str = "") -> str:
    """Find files matching a glob pattern. Returns matching file paths sorted by modification time.

    Args:
        pattern: Glob pattern like '**/*.py', 'src/**/*.ts', '*.json'.
        path: Directory to search in (default: working directory).
    """
    cwd = _get_cwd()
    search_dir = _resolve(path, cwd) if path else cwd

    matches = []
    for root, dirs, files in os.walk(search_dir):
        # Skip hidden dirs and common noise
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git", "venv", ".venv")]
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, cwd)
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(name, pattern):
                try:
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    mtime = 0
                matches.append((rel_path, mtime))

    # Sort by modification time (newest first)
    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return f"No files matching '{pattern}'"

    result = [m[0] for m in matches[:50]]
    output = "\n".join(result)
    if len(matches) > 50:
        output += f"\n... and {len(matches) - 50} more"
    return output


@tool
def grep(pattern: str, path: str = "", include: str = "") -> str:
    """Search file contents for a regex pattern. Returns matching lines with file paths and line numbers.

    Args:
        pattern: Regex pattern to search for.
        path: File or directory to search (default: working directory).
        include: Glob to filter files, e.g. '*.py', '*.ts'.
    """
    cwd = _get_cwd()
    search_path = _resolve(path, cwd) if path else cwd

    cmd = f"grep -rn --include='*' -E {_shell_quote(pattern)} {_shell_quote(search_path)}"
    if include:
        cmd = f"grep -rn --include={_shell_quote(include)} -E {_shell_quote(pattern)} {_shell_quote(search_path)}"

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip()
        if not output:
            return f"No matches for '{pattern}'"

        # Make paths relative
        lines = output.split("\n")
        rel_lines = []
        for line in lines[:50]:
            rel_lines.append(line.replace(cwd + "/", ""))
        output = "\n".join(rel_lines)
        if len(lines) > 50:
            output += f"\n... and {len(lines) - 50} more matches"
        return output
    except subprocess.TimeoutExpired:
        return "[search timed out]"
    except Exception as e:
        return f"[error: {e}]"


def _shell_quote(s: str) -> str:
    """Simple shell quoting."""
    return "'" + s.replace("'", "'\\''") + "'"


from agent.memory_tools import MEMORY_TOOLS

ALL_TOOLS = [bash, read, write, edit, glob, grep] + MEMORY_TOOLS
