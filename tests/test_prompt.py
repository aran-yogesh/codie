"""Tests for prompt module."""

from __future__ import annotations

import asyncio

from agent.prompt import build_system_prompt


def _run(coro):
    return asyncio.run(coro)


class TestBuildSystemPrompt:
    def test_includes_cwd(self, tmp_path):
        prompt = _run(build_system_prompt(str(tmp_path)))
        assert str(tmp_path) in prompt

    def test_includes_tool_instructions(self, tmp_path):
        prompt = _run(build_system_prompt(str(tmp_path)))
        assert "bash" in prompt.lower()
        assert "read" in prompt.lower()
        assert "edit" in prompt.lower()

    def test_safe_with_braces(self):
        prompt = _run(build_system_prompt("/path/{with}/braces"))
        assert "/path/{with}/braces" in prompt

    def test_detects_git(self, tmp_path):
        import subprocess
        subprocess.run("git init", shell=True, cwd=str(tmp_path), capture_output=True)
        prompt = _run(build_system_prompt(str(tmp_path)))
        assert "Git" in prompt or "main" in prompt or "master" in prompt

    def test_detects_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
        prompt = _run(build_system_prompt(str(tmp_path)))
        assert "Python" in prompt
