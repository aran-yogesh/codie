"""CLI client — type `codie` in any directory. Talks to the LangGraph server."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent import memory

console = Console()

SERVER_URL = os.environ.get("LANGGRAPH_API_URL", "http://localhost:2025")


def print_banner():
    mem_status = "[green]memory on[/green]" if memory.is_enabled() else "[dim]memory off[/dim]"
    console.print(
        Panel(
            f"[bold]codie[/bold] — self-improving coding agent ({mem_status})\n"
            "  [dim]/quit[/dim]     — exit\n"
            "  [dim]/clear[/dim]    — clear screen\n"
            "  [dim]/memories[/dim] — show stored memories",
            title="codie",
            border_style="blue",
        )
    )


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def _tool_header(name: str, inputs: dict) -> str:
    if name == "bash":
        cmd = inputs.get("command", "")
        display = cmd if len(cmd) <= 80 else cmd[:77] + "..."
        return f"[yellow]●[/yellow] [bold]Bash[/bold]({display})"
    elif name == "read":
        return f"[yellow]●[/yellow] [bold]Read[/bold]({inputs.get('path', '')})"
    elif name == "write":
        return f"[yellow]●[/yellow] [bold]Write[/bold]({inputs.get('path', '')})"
    elif name == "edit":
        return f"[yellow]●[/yellow] [bold]Edit[/bold]({inputs.get('path', '')})"
    elif name == "glob":
        return f"[yellow]●[/yellow] [bold]Glob[/bold]({inputs.get('pattern', '')})"
    elif name == "grep":
        p = inputs.get("pattern", "")
        inc = inputs.get("include", "")
        return f"[yellow]●[/yellow] [bold]Grep[/bold]({p}{', ' + inc if inc else ''})"
    elif name == "recall":
        q = inputs.get("query", "")
        display = q if len(q) <= 60 else q[:57] + "..."
        return f"[yellow]●[/yellow] [bold]Recall[/bold]({display})"
    return f"[yellow]●[/yellow] [bold]{name}[/bold]"


def _tool_result(name: str, inputs: dict, result: str):
    lines = result.strip().split("\n")
    if name == "bash":
        for line in lines[:5]:
            console.print(f"  ⎿ [dim]{line}[/dim]")
        if len(lines) > 5:
            console.print(f"  ⎿ [dim]... ({len(lines) - 5} more lines)[/dim]")
    elif name == "read":
        console.print(f"  ⎿ [dim]Read {max(0, len(lines) - 1)} lines from {inputs.get('path', '')}[/dim]")
    elif name == "write":
        console.print(f"  ⎿ [dim]{result}[/dim]")
    elif name == "edit":
        console.print(f"  ⎿ [dim]{result}[/dim]")
    elif name == "glob":
        console.print(f"  ⎿ [dim]Found {len(lines)} files[/dim]")
    elif name == "grep":
        console.print(f"  ⎿ [dim]{len(lines)} matches[/dim]")
    elif name == "recall":
        for line in lines[:3]:
            console.print(f"  ⎿ [dim]{line}[/dim]")
        if len(lines) > 3:
            console.print(f"  ⎿ [dim]... ({len(lines) - 3} more)[/dim]")
    else:
        for line in lines[:3]:
            console.print(f"  ⎿ [dim]{line}[/dim]")


def _print_steps(messages: list) -> str:
    """Walk messages, print tool steps, return final AI text."""
    pending = {}
    final_text = ""

    for msg in messages:
        if isinstance(msg, dict):
            msg_type = msg.get("type", "")
            content = msg.get("content", "")
        else:
            msg_type = getattr(msg, "type", "")
            content = getattr(msg, "content", "")

        if msg_type == "ai":
            tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
            for tc in (tool_calls or []):
                if isinstance(tc, dict):
                    tid, tname, tinputs = tc.get("id", ""), tc.get("name", ""), tc.get("args", {})
                else:
                    tid, tname, tinputs = getattr(tc, "id", ""), getattr(tc, "name", ""), getattr(tc, "args", {})
                if tid not in pending:
                    pending[tid] = (tname, tinputs)
                    console.print(_tool_header(tname, tinputs))

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tid = block.get("id", "")
                        if tid not in pending:
                            tname = block.get("name", "")
                            tinputs = block.get("input", {})
                            pending[tid] = (tname, tinputs)
                            console.print(_tool_header(tname, tinputs))

            text = _extract_text(content)
            if text.strip():
                final_text = text

        elif msg_type == "tool":
            tcid = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
            result = _extract_text(content)
            if tcid in pending:
                tname, tinputs = pending[tcid]
                _tool_result(tname, tinputs, result)

    return final_text


# ─── MEMORY INTEGRATION ─────────────────────────────────────────────


def _recall(task: str, repo_id: str) -> tuple[str, str, list[str]]:
    """BEFORE task: two-phase retrieval. Returns (task, memories_text, memory_ids)."""
    if not memory.is_enabled() or not memory.should_recall(task):
        return task, "", []

    memories, memory_ids = memory.recall(task, repo_id=repo_id)
    if memories:
        console.print(f"[dim]● recalled {len(memories)} memories (MemRL ranked)[/dim]")
        return task, memory.format_for_system_prompt(memories), memory_ids

    return task, "", []


def _check_correction(user_input: str, repo_id: str):
    """Check if user input is a correction — save immediately, don't wait for end of task."""
    if not memory.is_enabled():
        return
    correction = memory.detect_correction(user_input)
    if correction:
        memory.learn_correction(correction, repo_id)
        console.print("[dim]● saved correction[/dim]")


def _learn(messages: list, repo_id: str, recalled_ids: list[str] | None = None):
    """AFTER task: extract facts + MemRL utility update + retroactive extraction."""
    if not memory.is_enabled():
        return

    # MemRL: compute reward and update Q-values of recalled memories
    reward = 1.0
    if recalled_ids:
        reward = memory.compute_reward(messages)
        memory.update_utility(recalled_ids, reward)
        console.print(f"[dim]● utility update: reward={reward} for {len(recalled_ids)} memories[/dim]")
    else:
        reward = memory.compute_reward(messages)

    if not memory.should_learn(messages):
        return

    # Standard extraction — "what happened?"
    facts = memory.extract_facts(messages)
    if facts:
        stored = memory.learn(facts, repo_id=repo_id)
        console.print(f"[dim]● learned {stored} facts[/dim]")

    # Retroactive extraction — "what made this succeed?" (only on clean success)
    if reward == 1.0:
        retro_facts = memory.extract_retro_facts(messages)
        if retro_facts:
            retro_stored = memory.learn(retro_facts, repo_id=repo_id)
            console.print(f"[dim]● retro-learned {retro_stored} patterns[/dim]")


def _maybe_tick():
    """Run ticker on session start if needed."""
    if not memory.is_enabled():
        return
    if memory.should_tick():
        console.print("[dim]● consolidating memories...[/dim]")
        summary = memory.tick()
        if summary:
            console.print(f"[dim]● {summary}[/dim]")


# ─── SESSION ─────────────────────────────────────────────────────────


async def session(cwd: str):
    """One event loop, one thread for the entire session."""
    from langgraph_sdk import get_client

    client = get_client(url=SERVER_URL)
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    repo_id = memory.get_repo_id(cwd)
    printed_msg_count = 0

    while True:
        try:
            task = await asyncio.to_thread(console.input, "[bold blue]>[/bold blue] ")
            task = task.strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not task:
            continue
        if task == "/quit":
            console.print("[dim]Bye![/dim]")
            break
        if task == "/clear":
            console.clear()
            print_banner()
            continue
        if task == "/memories":
            all_mems = memory.get_all()
            if not all_mems:
                console.print("[dim]No memories stored yet.[/dim]")
            else:
                console.print(f"\n[bold]Memories ({len(all_mems)}):[/bold]")
                for i, m in enumerate(all_mems, 1):
                    console.print(f"  {i}. {m}")
            console.print()
            continue

        console.print()

        # ── Check for corrections (save immediately) ──
        _check_correction(task, repo_id)

        # ── RECALL: two-phase retrieval (MemRL) ──
        task, memories_text, recalled_ids = _recall(task, repo_id)

        try:
            # ── WORK ──
            status = console.status("[bold blue]...[/bold blue]", spinner="dots")
            status.start()

            run = await client.runs.create(
                thread_id,
                "agent",
                input={"messages": [{"role": "user", "content": task}]},
                config={"configurable": {"cwd": cwd, "memories": memories_text}},
            )
            await client.runs.join(thread_id, run["run_id"])
            status.stop()

            state = await client.threads.get_state(thread_id)
            messages = state.get("values", {}).get("messages", [])

            new_messages = messages[printed_msg_count:]
            printed_msg_count = len(messages)

            response = _print_steps(new_messages)

            if response:
                console.print()
                console.print(Markdown(response))
                console.print()

                # ── LEARN + MemRL UTILITY UPDATE ──
                _learn(new_messages, repo_id, recalled_ids=recalled_ids)
            else:
                console.print("[dim]No response.[/dim]")

        except Exception as e:
            console.print(f"[red]{e}[/red]")
            console.print("[dim]Is the server running? uv run langgraph dev --port 2025[/dim]\n")


def main():
    # Load .env from: ~/.codie/.env (global) → cwd/.env (local override)
    home_env = os.path.expanduser("~/.codie/.env")
    load_dotenv(home_env)
    load_dotenv()  # also load from cwd
    print_banner()
    cwd = os.getcwd()
    console.print(f"[dim]{cwd}[/dim]\n")

    # Ticker: consolidate memories on session start if due
    _maybe_tick()
    memory.bump_session_count()

    try:
        asyncio.run(session(cwd))
    except KeyboardInterrupt:
        console.print("\n[dim]Bye![/dim]")


if __name__ == "__main__":
    main()
