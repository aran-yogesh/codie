"""Tests for agent/prompt.py — system prompt builder."""

import os

import pytest

from agent.prompt import build_system_prompt, build_memory_section, _detect_context
from agent.memory import ensure_codie_dir, save_block, save_skill


@pytest.fixture
def cwd(tmp_path):
    ensure_codie_dir(str(tmp_path))
    return str(tmp_path)


class TestBuildSystemPrompt:
    def test_includes_cwd(self, cwd):
        prompt = build_system_prompt(cwd)
        assert cwd in prompt

    def test_includes_memory_section(self, cwd):
        prompt = build_system_prompt(cwd)
        assert "<memory>" in prompt

    def test_includes_tool_instructions(self, cwd):
        prompt = build_system_prompt(cwd)
        assert "core_memory_append" in prompt
        assert "archival_memory_search" in prompt


class TestBuildMemorySection:
    def test_empty_blocks(self, cwd):
        section = build_memory_section(cwd)
        assert "<memory>" in section
        assert "persona" in section

    def test_block_content_included(self, cwd):
        save_block(cwd, "user", "prefers tabs over spaces")
        section = build_memory_section(cwd)
        assert "prefers tabs over spaces" in section

    def test_skills_listed(self, cwd):
        save_skill(cwd, "deploy", "Deploy the app", "1. Build\n2. Push")
        section = build_memory_section(cwd)
        assert "<skills>" in section
        assert "deploy" in section

    def test_no_codie_dir(self, tmp_path):
        # No .codie/ exists — should not crash
        section = build_memory_section(str(tmp_path))
        # Either empty or gracefully handles missing dir
        assert isinstance(section, str)


class TestDetectContext:
    def test_detects_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
        context = _detect_context(str(tmp_path))
        assert "Python" in context

    def test_detects_node(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name":"test"}')
        context = _detect_context(str(tmp_path))
        assert "Node.js" in context

    def test_reads_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Always use pytest")
        context = _detect_context(str(tmp_path))
        assert "Always use pytest" in context
