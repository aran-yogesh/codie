"""Reflection and learning. Used by /quit and /remember commands."""

from __future__ import annotations


def _reflect_and_learn(cwd: str, messages: list) -> None:
    """Scan messages for errors and user corrections, save as lessons."""
    from agent.memory import add_lesson, ensure_codie_dir

    ensure_codie_dir(cwd)

    tool_errors = []
    corrections = []

    for msg in messages:
        if isinstance(msg, dict):
            msg_type = msg.get("type", "")
            content = msg.get("content", "")
        else:
            msg_type = getattr(msg, "type", "")
            content = getattr(msg, "content", "")

        # Normalize content to string
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )

        # Detect tool errors
        if msg_type == "tool":
            error_markers = ("[error", "[file not found", "[old_string not found",
                             "[timed out", "[exit code:", "[refused")
            if any(marker in content for marker in error_markers):
                tool_errors.append(content[:200])

        # Detect user corrections
        if msg_type in ("human", "user"):
            correction_words = ("actually", "no,", "no ", "wrong", "instead",
                                "should be", "not that", "don't", "stop", "fix")
            content_lower = content.lower().strip()
            if any(content_lower.startswith(w) or f" {w}" in content_lower
                   for w in correction_words):
                corrections.append(content[:200])

    # Save last few errors as lessons
    for error in tool_errors[-3:]:
        add_lesson(cwd, f"Tool error encountered: {error}", verified=False)

    # Save corrections as verified lessons (user explicitly told us)
    for correction in corrections[-2:]:
        add_lesson(cwd, f"User correction: {correction}", verified=True)


def build_remember_prompt() -> str:
    """Build the prompt for /remember command."""
    return (
        "Review this conversation and extract key learnings. For each learning:\n"
        "1. Decide if it belongs in a core memory block (persona/user/project) or archival memory\n"
        "2. Use the appropriate memory tool to save it\n"
        "3. Be concise — distill to the essence\n\n"
        "Look for:\n"
        "- User preferences or corrections\n"
        "- Project patterns, conventions, or gotchas\n"
        "- Successful approaches worth remembering\n"
        "- Mistakes to avoid in the future\n\n"
        "After updating memory, briefly confirm what you saved."
    )
