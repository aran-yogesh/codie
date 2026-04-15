"""Tests for agent/memory.py — storage layer."""

import os
import shutil

import pytest

from agent.memory import (
    ensure_codie_dir,
    get_agent_id,
    load_block,
    save_block,
    load_all_blocks,
    append_to_block,
    replace_in_block,
    compact_block,
    archival_insert,
    archival_search,
    add_lesson,
    load_lessons,
    save_skill,
    search_skills,
    load_skills,
    save_session_summary,
    load_recent_sessions,
    VALID_LABELS,
)


@pytest.fixture
def cwd(tmp_path):
    """Provide a temp directory with .codie/ initialized."""
    ensure_codie_dir(str(tmp_path))
    return str(tmp_path)


# --- Directory / config ---

class TestInit:
    def test_creates_directory_structure(self, cwd):
        base = os.path.join(cwd, ".codie")
        assert os.path.isdir(base)
        assert os.path.isdir(os.path.join(base, "memory"))
        assert os.path.isdir(os.path.join(base, "archival"))
        assert os.path.isdir(os.path.join(base, "lessons"))
        assert os.path.isdir(os.path.join(base, "skills"))
        assert os.path.isdir(os.path.join(base, "sessions"))

    def test_creates_config(self, cwd):
        assert os.path.isfile(os.path.join(cwd, ".codie", "config.json"))

    def test_agent_id_is_stable(self, cwd):
        id1 = get_agent_id(cwd)
        id2 = get_agent_id(cwd)
        assert id1 == id2
        assert id1 != "unknown"

    def test_idempotent(self, cwd):
        id1 = get_agent_id(cwd)
        ensure_codie_dir(cwd)
        id2 = get_agent_id(cwd)
        assert id1 == id2

    def test_default_blocks_created(self, cwd):
        for label in VALID_LABELS:
            path = os.path.join(cwd, ".codie", "memory", f"{label}.json")
            assert os.path.isfile(path)

    def test_persona_has_default_content(self, cwd):
        block = load_block(cwd, "persona")
        assert "codie" in block["content"].lower()


# --- Memory block CRUD ---

class TestBlocks:
    def test_load_all_blocks(self, cwd):
        blocks = load_all_blocks(cwd)
        assert len(blocks) == 3
        labels = [b["label"] for b in blocks]
        assert labels == ["persona", "user", "project"]

    def test_save_and_load(self, cwd):
        save_block(cwd, "user", "prefers pytest")
        block = load_block(cwd, "user")
        assert block["content"] == "prefers pytest"
        assert block["max_chars"] == 2000

    def test_append(self, cwd):
        result = append_to_block(cwd, "user", "likes type hints")
        assert "type hints" in result
        result = append_to_block(cwd, "user", "uses black formatter")
        assert "type hints" in result
        assert "black formatter" in result

    def test_append_newline_separator(self, cwd):
        append_to_block(cwd, "user", "line one")
        result = append_to_block(cwd, "user", "line two")
        assert "\n" in result

    def test_replace(self, cwd):
        append_to_block(cwd, "user", "prefers pytest")
        result = replace_in_block(cwd, "user", "pytest", "unittest")
        assert "unittest" in result
        assert "pytest" not in result

    def test_replace_not_found(self, cwd):
        append_to_block(cwd, "user", "prefers pytest")
        result = replace_in_block(cwd, "user", "nonexistent", "something")
        assert "[old_content not found" in result

    def test_replace_multiple_matches(self, cwd):
        append_to_block(cwd, "user", "test test")
        result = replace_in_block(cwd, "user", "test", "foo")
        assert "[found 2 matches" in result

    def test_invalid_label_append(self, cwd):
        result = append_to_block(cwd, "invalid", "test")
        assert "[error" in result

    def test_invalid_label_replace(self, cwd):
        result = replace_in_block(cwd, "invalid", "a", "b")
        assert "[error" in result

    def test_max_chars_enforced(self, cwd):
        save_block(cwd, "user", "", max_chars=50)
        result = append_to_block(cwd, "user", "a" * 60)
        # Should either compact or error
        block = load_block(cwd, "user")
        assert len(block["content"]) <= 50 or "[error" in result

    def test_replace_exceeds_limit(self, cwd):
        save_block(cwd, "user", "short", max_chars=20)
        result = replace_in_block(cwd, "user", "short", "a" * 30)
        assert "[error" in result


# --- Compaction ---

class TestCompaction:
    def test_removes_duplicate_lines(self, cwd):
        save_block(cwd, "project", "line one\nline two\nline one\nline three")
        result = compact_block(cwd, "project")
        assert result.count("line one") == 1
        assert "line two" in result
        assert "line three" in result

    def test_stays_within_limit(self, cwd):
        save_block(cwd, "project", "", max_chars=100)
        for i in range(20):
            append_to_block(cwd, "project", f"info line {i}")
        block = load_block(cwd, "project")
        assert len(block["content"]) <= 100


# --- Archival ---

class TestArchival:
    def test_insert_and_search(self, cwd):
        archival_insert(cwd, "FastAPI uses Pydantic for validation")
        archival_insert(cwd, "Database uses PostgreSQL with SQLAlchemy")
        archival_insert(cwd, "Tests run with pytest")

        results = archival_search(cwd, "database PostgreSQL")
        assert len(results) > 0
        assert any("PostgreSQL" in r["content"] for r in results)

    def test_insert_returns_entry(self, cwd):
        entry = archival_insert(cwd, "some fact")
        assert "id" in entry
        assert "timestamp" in entry
        assert entry["content"] == "some fact"

    def test_search_no_results(self, cwd):
        results = archival_search(cwd, "nonexistent query")
        assert results == []

    def test_search_empty_store(self, cwd):
        results = archival_search(cwd, "anything")
        assert results == []

    def test_search_respects_limit(self, cwd):
        for i in range(10):
            archival_insert(cwd, f"fact number {i} about python")
        results = archival_search(cwd, "python", limit=3)
        assert len(results) == 3


# --- Lessons ---

class TestLessons:
    def test_add_and_load(self, cwd):
        add_lesson(cwd, "always run tests before commit", verified=True)
        lessons = load_lessons(cwd)
        assert len(lessons) == 1
        assert lessons[0]["content"] == "always run tests before commit"
        assert lessons[0]["verified"] is True

    def test_dedup(self, cwd):
        l1 = add_lesson(cwd, "same lesson")
        l2 = add_lesson(cwd, "same lesson")
        assert l1["id"] == l2["id"]
        assert len(load_lessons(cwd)) == 1

    def test_different_lessons(self, cwd):
        add_lesson(cwd, "lesson one")
        add_lesson(cwd, "lesson two")
        assert len(load_lessons(cwd)) == 2


# --- Skills ---

class TestSkills:
    def test_save_and_search(self, cwd):
        save_skill(cwd, "add-endpoint", "Add a new API endpoint",
                   "1. Create route\n2. Add schema\n3. Write tests")
        results = search_skills(cwd, "endpoint API")
        assert len(results) > 0
        assert results[0]["name"] == "add-endpoint"

    def test_save_creates_file(self, cwd):
        path = save_skill(cwd, "deploy", "Deploy the app", "1. Build\n2. Push")
        assert os.path.isfile(path)
        assert path.endswith(".md")

    def test_load_all_skills(self, cwd):
        save_skill(cwd, "skill-a", "First skill", "steps")
        save_skill(cwd, "skill-b", "Second skill", "steps")
        skills = load_skills(cwd)
        assert len(skills) == 2

    def test_search_no_results(self, cwd):
        results = search_skills(cwd, "nonexistent")
        assert results == []

    def test_name_sanitization(self, cwd):
        path = save_skill(cwd, "My Skill! @#$", "desc", "steps")
        assert os.path.isfile(path)
        assert " " not in os.path.basename(path)


# --- Sessions ---

class TestSessions:
    def test_save_and_load(self, cwd):
        save_session_summary(cwd, "s1", "Fixed auth bug")
        save_session_summary(cwd, "s2", "Added endpoints")
        sessions = load_recent_sessions(cwd, limit=5)
        assert len(sessions) == 2

    def test_limit(self, cwd):
        for i in range(10):
            save_session_summary(cwd, f"s{i}", f"Session {i}")
        sessions = load_recent_sessions(cwd, limit=3)
        assert len(sessions) == 3

    def test_empty(self, cwd):
        sessions = load_recent_sessions(cwd)
        assert sessions == []
