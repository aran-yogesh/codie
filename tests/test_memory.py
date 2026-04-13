"""Tests for memory module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock external deps
sys.modules.setdefault("mem0", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from agent.memory import (
    _compress_transcript,
    format_for_system_prompt,
    is_enabled,
)


class TestIsEnabled:
    def test_disabled_without_api_key(self, monkeypatch):
        monkeypatch.setenv("MEM0_API_KEY", "")
        import agent.memory as m
        m._client = None
        assert m.is_enabled() is False

    def test_disabled_without_mem0(self):
        import agent.memory as m
        old = m._HAS_MEM0
        m._HAS_MEM0 = False
        m._client = None
        assert m.is_enabled() is False
        m._HAS_MEM0 = old


class TestFormatForSystemPrompt:
    def test_empty(self):
        assert format_for_system_prompt([]) == ""

    def test_single_memory(self):
        result = format_for_system_prompt(["user prefers pytest"])
        assert "Past Sessions" in result
        assert "user prefers pytest" in result

    def test_multiple_memories(self):
        result = format_for_system_prompt(["fact one", "fact two", "fact three"])
        assert "1. fact one" in result
        assert "2. fact two" in result
        assert "3. fact three" in result


class TestCompressTranscript:
    def test_first_human_is_task(self):
        messages = [
            {"type": "human", "content": "fix the bug"},
        ]
        transcript = _compress_transcript(messages)
        assert "TASK: fix the bug" in transcript

    def test_second_human_is_user(self):
        messages = [
            {"type": "human", "content": "fix the bug"},
            {"type": "ai", "content": "done", "tool_calls": []},
            {"type": "human", "content": "now add tests"},
        ]
        transcript = _compress_transcript(messages)
        assert "TASK: fix the bug" in transcript
        assert "USER: now add tests" in transcript

    def test_tool_errors_included(self):
        messages = [
            {"type": "human", "content": "read config"},
            {"type": "tool", "content": "[file not found: config.yaml]"},
        ]
        transcript = _compress_transcript(messages)
        assert "TOOL ERROR" in transcript

    def test_normal_tool_output_excluded(self):
        messages = [
            {"type": "human", "content": "read file"},
            {"type": "tool", "content": "line 1\nline 2\nline 3"},
        ]
        transcript = _compress_transcript(messages)
        assert "line 1" not in transcript

    def test_ai_tool_calls_shown(self):
        messages = [
            {"type": "human", "content": "fix it"},
            {"type": "ai", "content": "Let me look.", "tool_calls": [
                {"name": "read"}, {"name": "edit"},
            ]},
        ]
        transcript = _compress_transcript(messages)
        assert "AGENT USED: read, edit" in transcript

    def test_skips_empty(self):
        messages = [
            {"type": "human", "content": ""},
            {"type": "ai", "content": "hello", "tool_calls": []},
        ]
        transcript = _compress_transcript(messages)
        assert "TASK" not in transcript

    def test_final_result_included(self):
        messages = [
            {"type": "human", "content": "do something"},
            {"type": "ai", "content": "Here's what I did.", "tool_calls": []},
        ]
        transcript = _compress_transcript(messages)
        assert "RESULT: Here's what I did." in transcript
