"""Tests for agent tools."""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock

# Mock langgraph before importing tools
sys.modules.setdefault("langgraph", MagicMock())
sys.modules.setdefault("langgraph.config", MagicMock())
sys.modules.setdefault("langchain_core", MagicMock())
sys.modules.setdefault("langchain_core.tools", MagicMock())

# Now import — but we need the real tool functions, not mocked ones.
# Re-import after patching the decorator to be a passthrough.
import importlib
from langchain_core.tools import tool
tool.side_effect = lambda fn: fn  # Make @tool a no-op decorator

import agent.tools as tools


class TestReadFile:
    def test_read_full_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = tools.read.invoke({"path": "test.txt"})
        assert "line1" in result
        assert "line2" in result
        assert "lines 1-3 of 3" in result

    def test_read_line_range(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        result = tools.read.invoke({"path": "test.txt", "start_line": 2, "end_line": 4})
        assert "lines 2-4 of 5" in result
        assert "b" in result
        assert "d" in result

    def test_read_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        result = tools.read.invoke({"path": "nope.txt"})
        assert "file not found" in result


class TestWriteFile:
    def test_write_new_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        result = tools.write.invoke({"path": "out.txt", "content": "hello\nworld\n"})
        assert "Wrote 2 lines" in result
        assert (tmp_path / "out.txt").read_text() == "hello\nworld\n"

    def test_write_creates_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        result = tools.write.invoke({"path": "sub/dir/file.txt", "content": "nested"})
        assert "Wrote" in result
        assert (tmp_path / "sub" / "dir" / "file.txt").exists()

    def test_refuses_env_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        result = tools.write.invoke({"path": ".env", "content": "SECRET=bad"})
        assert "refused" in result


class TestEditFile:
    def test_edit_replaces(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        f = tmp_path / "code.py"
        f.write_text("def add(a, b):\n    return a + b\n")
        result = tools.edit.invoke({
            "path": "code.py",
            "old_string": "return a + b",
            "new_string": "return a + b  # fixed",
        })
        assert "Replaced 1 occurrence" in result
        assert "# fixed" in f.read_text()

    def test_edit_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        f = tmp_path / "code.py"
        f.write_text("hello world")
        result = tools.edit.invoke({
            "path": "code.py",
            "old_string": "xyz",
            "new_string": "abc",
        })
        assert "not found" in result

    def test_edit_multiple_matches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        f = tmp_path / "code.py"
        f.write_text("foo\nfoo\nbar\n")
        result = tools.edit.invoke({
            "path": "code.py",
            "old_string": "foo",
            "new_string": "baz",
        })
        assert "found 2 matches" in result


class TestGlob:
    def test_finds_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = tools.glob.invoke({"pattern": "*.py"})
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_no_matches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        result = tools.glob.invoke({"pattern": "*.rs"})
        assert "No files" in result


class TestGrep:
    def test_finds_pattern(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        (tmp_path / "code.py").write_text("def hello():\n    pass\n")
        result = tools.grep.invoke({"pattern": "def hello", "path": str(tmp_path)})
        assert "def hello" in result

    def test_no_matches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        (tmp_path / "code.py").write_text("nothing here")
        result = tools.grep.invoke({"pattern": "foobar", "path": str(tmp_path)})
        assert "No matches" in result


class TestBash:
    def test_runs_command(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        result = tools.bash.invoke({"command": "echo hello"})
        assert "hello" in result

    def test_captures_stderr(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_get_cwd", lambda: str(tmp_path))
        result = tools.bash.invoke({"command": "ls /nonexistent 2>&1"})
        assert "No such file" in result or "exit code" in result
