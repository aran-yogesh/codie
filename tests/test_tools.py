"""Tests for agent/tools.py — coding tools."""

import os

import pytest

from agent.tools import bash, read, write, edit, glob, grep


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with a sample file."""
    sample = tmp_path / "hello.py"
    sample.write_text("def greet(name):\n    return f'Hello, {name}!'\n")
    return str(tmp_path)


class TestBash:
    def test_echo(self):
        result = bash.invoke({"command": "echo hello"})
        assert "hello" in result

    def test_exit_code(self):
        result = bash.invoke({"command": "exit 1"})
        assert "[exit code: 1]" in result


class TestRead:
    def test_read_file(self, workspace):
        result = read.invoke({"path": os.path.join(workspace, "hello.py")})
        assert "def greet" in result

    def test_read_not_found(self):
        result = read.invoke({"path": "/tmp/nonexistent_file_xyz.py"})
        assert "[file not found" in result


class TestWrite:
    def test_write_file(self, workspace):
        path = os.path.join(workspace, "new.py")
        result = write.invoke({"path": path, "content": "x = 1\n"})
        assert "Wrote" in result
        assert os.path.isfile(path)

    def test_refuses_env(self, workspace):
        path = os.path.join(workspace, ".env")
        result = write.invoke({"path": path, "content": "SECRET=abc"})
        assert "[refused" in result


class TestEdit:
    def test_replace(self, workspace):
        path = os.path.join(workspace, "hello.py")
        result = edit.invoke({"path": path, "old_string": "def greet(name):", "new_string": "def greet(user):"})
        assert "Replaced" in result

    def test_not_found(self, workspace):
        path = os.path.join(workspace, "hello.py")
        result = edit.invoke({"path": path, "old_string": "nonexistent", "new_string": "x"})
        assert "[old_string not found" in result


class TestGlob:
    def test_find_py(self, workspace):
        result = glob.invoke({"pattern": "*.py", "path": workspace})
        assert "hello.py" in result

    def test_no_match(self, workspace):
        result = glob.invoke({"pattern": "*.rs", "path": workspace})
        assert "No files" in result


class TestGrep:
    def test_find_pattern(self, workspace):
        result = grep.invoke({"pattern": "def greet", "path": workspace})
        assert "greet" in result

    def test_no_match(self, workspace):
        result = grep.invoke({"pattern": "nonexistent_pattern", "path": workspace})
        assert "No matches" in result
