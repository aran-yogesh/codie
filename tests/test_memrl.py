"""Tests for MemRL — Q-values, two-phase retrieval, utility update, reward computation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Mock external deps
sys.modules.setdefault("mem0", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

import agent.memory as mem


# ─── Q-VALUE STORE ───────────────────────────────────────────────────


class TestQValueStore:
    def test_read_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mem, "Q_FILE", str(tmp_path / "q.json"))
        assert mem._read_q_values() == {}

    def test_write_and_read(self, tmp_path, monkeypatch):
        qfile = str(tmp_path / "q.json")
        monkeypatch.setattr(mem, "Q_FILE", qfile)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        data = {"mem-123": {"q": 0.8, "used": 3, "helped": 2}}
        mem._write_q_values(data)

        result = mem._read_q_values()
        assert result["mem-123"]["q"] == 0.8
        assert result["mem-123"]["used"] == 3

    def test_get_q_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mem, "Q_FILE", str(tmp_path / "q.json"))
        assert mem._get_q("nonexistent") == mem.Q_INIT

    def test_get_q_existing(self, tmp_path, monkeypatch):
        qfile = str(tmp_path / "q.json")
        monkeypatch.setattr(mem, "Q_FILE", qfile)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        mem._write_q_values({"mem-abc": {"q": 0.9, "used": 5, "helped": 4}})
        assert mem._get_q("mem-abc") == 0.9


# ─── REWARD COMPUTATION ──────────────────────────────────────────────


class TestComputeReward:
    def test_clean_success(self):
        messages = [
            {"type": "human", "content": "fix the bug"},
            {"type": "ai", "content": "Fixed it.", "tool_calls": [{"name": "edit"}]},
        ]
        assert mem.compute_reward(messages) == 1.0

    def test_success_with_errors(self):
        messages = [
            {"type": "human", "content": "fix the bug"},
            {"type": "tool", "content": "[error: something broke]"},
            {"type": "ai", "content": "Fixed it after retrying.", "tool_calls": []},
        ]
        assert mem.compute_reward(messages) == 0.5

    def test_no_response(self):
        messages = [
            {"type": "human", "content": "fix the bug"},
        ]
        assert mem.compute_reward(messages) == 0.0

    def test_user_correction(self):
        messages = [
            {"type": "human", "content": "fix the bug"},
            {"type": "ai", "content": "Done."},
            {"type": "human", "content": "no, don't do it that way"},
        ]
        assert mem.compute_reward(messages) == -0.5

    def test_file_not_found_counts_as_error(self):
        messages = [
            {"type": "human", "content": "read the config"},
            {"type": "tool", "content": "[file not found: config.yaml]"},
            {"type": "ai", "content": "Config not found, created one."},
        ]
        assert mem.compute_reward(messages) == 0.5

    def test_exit_code_counts_as_error(self):
        messages = [
            {"type": "human", "content": "run tests"},
            {"type": "tool", "content": "FAILED\n[exit code: 1]"},
            {"type": "ai", "content": "Tests failed, fixing."},
        ]
        assert mem.compute_reward(messages) == 0.5


# ─── UTILITY UPDATE ──────────────────────────────────────────────────


class TestUpdateUtility:
    def test_success_increases_q(self, tmp_path, monkeypatch):
        qfile = str(tmp_path / "q.json")
        monkeypatch.setattr(mem, "Q_FILE", qfile)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        # Start at Q_INIT (0.5)
        mem._write_q_values({})
        mem.update_utility(["mem-1", "mem-2"], reward=1.0)

        qv = mem._read_q_values()
        # Q_new = 0.5 + 0.1 * (1.0 - 0.5) = 0.55
        assert qv["mem-1"]["q"] == 0.55
        assert qv["mem-2"]["q"] == 0.55
        assert qv["mem-1"]["used"] == 1
        assert qv["mem-1"]["helped"] == 1

    def test_failure_decreases_q(self, tmp_path, monkeypatch):
        qfile = str(tmp_path / "q.json")
        monkeypatch.setattr(mem, "Q_FILE", qfile)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        mem._write_q_values({"mem-1": {"q": 0.8, "used": 5, "helped": 4}})
        mem.update_utility(["mem-1"], reward=0.0)

        qv = mem._read_q_values()
        # Q_new = 0.8 + 0.1 * (0.0 - 0.8) = 0.72
        assert qv["mem-1"]["q"] == 0.72
        assert qv["mem-1"]["used"] == 6
        assert qv["mem-1"]["helped"] == 4  # not incremented

    def test_correction_penalizes(self, tmp_path, monkeypatch):
        qfile = str(tmp_path / "q.json")
        monkeypatch.setattr(mem, "Q_FILE", qfile)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        mem._write_q_values({"mem-1": {"q": 0.5, "used": 2, "helped": 1}})
        mem.update_utility(["mem-1"], reward=-0.5)

        qv = mem._read_q_values()
        # Q_new = 0.5 + 0.1 * (-0.5 - 0.5) = 0.4
        assert qv["mem-1"]["q"] == 0.4

    def test_repeated_success_converges_high(self, tmp_path, monkeypatch):
        qfile = str(tmp_path / "q.json")
        monkeypatch.setattr(mem, "Q_FILE", qfile)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        mem._write_q_values({})
        # Simulate 20 successes
        for _ in range(20):
            mem.update_utility(["mem-1"], reward=1.0)

        qv = mem._read_q_values()
        assert qv["mem-1"]["q"] > 0.85  # Should converge toward 1.0

    def test_repeated_failure_converges_low(self, tmp_path, monkeypatch):
        qfile = str(tmp_path / "q.json")
        monkeypatch.setattr(mem, "Q_FILE", qfile)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        mem._write_q_values({})
        # Simulate 20 failures
        for _ in range(20):
            mem.update_utility(["mem-1"], reward=0.0)

        qv = mem._read_q_values()
        assert qv["mem-1"]["q"] < 0.15  # Should converge toward 0.0

    def test_empty_ids_does_nothing(self, tmp_path, monkeypatch):
        qfile = str(tmp_path / "q.json")
        monkeypatch.setattr(mem, "Q_FILE", qfile)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        mem._write_q_values({})
        mem.update_utility([], reward=1.0)
        assert mem._read_q_values() == {}


# ─── SMART GATES ─────────────────────────────────────────────────────


class TestSmartGates:
    def test_should_recall_trivial(self):
        assert mem.should_recall("hey") is False
        assert mem.should_recall("hi") is False
        assert mem.should_recall("thanks") is False

    def test_should_recall_real_task(self):
        assert mem.should_recall("fix the authentication bug in middleware") is True
        assert mem.should_recall("create a calculator program") is True

    def test_should_recall_short(self):
        assert mem.should_recall("yo") is False
        assert mem.should_recall("what") is False

    def test_should_learn_trivial(self):
        messages = [
            {"type": "ai", "content": "Hello!", "tool_calls": []},
        ]
        assert mem.should_learn(messages) is False

    def test_should_learn_with_tools(self):
        messages = [
            {"type": "ai", "content": "", "tool_calls": [{"name": "read"}, {"name": "edit"}, {"name": "bash"}]},
        ]
        assert mem.should_learn(messages) is True

    def test_should_learn_with_errors(self):
        messages = [
            {"type": "tool", "content": "[error: something failed]"},
        ]
        assert mem.should_learn(messages) is True


# ─── CORRECTION DETECTION ────────────────────────────────────────────


class TestCorrectionDetection:
    def test_detects_dont(self):
        assert mem.detect_correction("don't use mocks in tests") is not None

    def test_detects_wrong(self):
        assert mem.detect_correction("that's wrong, fix it") is not None

    def test_detects_instead(self):
        assert mem.detect_correction("use pytest instead") is not None

    def test_ignores_normal(self):
        assert mem.detect_correction("create a new file") is None

    def test_truncates_long(self):
        result = mem.detect_correction("don't " + "x" * 500)
        assert result is not None
        assert len(result) <= 300


# ─── FORMAT ──────────────────────────────────────────────────────────


class TestFormatForSystemPrompt:
    def test_empty(self):
        assert mem.format_for_system_prompt([]) == ""

    def test_formats_memories(self):
        result = mem.format_for_system_prompt(["fact one", "fact two"])
        assert "Past Sessions" in result
        assert "1. fact one" in result
        assert "2. fact two" in result


# ─── TICKER ──────────────────────────────────────────────────────────


class TestTicker:
    def test_should_tick_fresh_install(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mem, "STATE_FILE", str(tmp_path / "state.json"))
        # No state file = never ticked = should tick
        assert mem.should_tick() is True

    def test_should_not_tick_recent(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "state.json")
        monkeypatch.setattr(mem, "STATE_FILE", state_file)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        import time
        mem._write_state({"last_tick": time.time(), "sessions_since_tick": 1})
        assert mem.should_tick() is False

    def test_should_tick_after_5_sessions(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "state.json")
        monkeypatch.setattr(mem, "STATE_FILE", state_file)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        import time
        mem._write_state({"last_tick": time.time(), "sessions_since_tick": 5})
        assert mem.should_tick() is True

    def test_bump_session_count(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "state.json")
        monkeypatch.setattr(mem, "STATE_FILE", state_file)
        monkeypatch.setattr(mem, "STATE_DIR", str(tmp_path))

        mem._write_state({"sessions_since_tick": 2})
        mem.bump_session_count()
        state = mem._read_state()
        assert state["sessions_since_tick"] == 3
