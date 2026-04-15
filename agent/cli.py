"""CLI client — type `codie` in any directory. Talks to the LangGraph server."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

SERVER_URL = os.environ.get("LANGGRAPH_API_URL", "http://localhost:2025")


def print_banner():
    console.print(
        Panel(
            "[bold]codie[/bold] — self-improving coding agent\n"
            "  [dim]/init[/dim]     — explore codebase and initialize memory\n"
            "  [dim]/remember[/dim] — reflect and save learnings to memory\n"
            "  [dim]/memory[/dim]   — show current memory blocks\n"
            "  [dim]/clear[/dim]    — clear screen\n"
            "  [dim]/quit[/dim]     — exit",
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
    elif name == "core_memory_append":
        return f"[magenta]●[/magenta] [bold]Memory Append[/bold]({inputs.get('label', '')})"
    elif name == "core_memory_replace":
        return f"[magenta]●[/magenta] [bold]Memory Replace[/bold]({inputs.get('label', '')})"
    elif name == "archival_memory_insert":
        return f"[magenta]●[/magenta] [bold]Archival Insert[/bold]"
    elif name == "archival_memory_search":
        return f"[magenta]●[/magenta] [bold]Archival Search[/bold]({inputs.get('query', '')})"
    elif name == "skill_save":
        return f"[magenta]●[/magenta] [bold]Skill Save[/bold]({inputs.get('name', '')})"
    elif name == "skill_search":
        return f"[magenta]●[/magenta] [bold]Skill Search[/bold]({inputs.get('query', '')})"
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


async def session(cwd: str):
    """Run the entire interactive session on one event loop, one thread."""
    from langgraph_sdk import get_client

    client = get_client(url=SERVER_URL)
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Track which messages we've already printed steps for
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
            # Post-session reflection
            try:
                from agent.reflection import _reflect_and_learn
                state = await client.threads.get_state(thread_id)
                messages = state.get("values", {}).get("messages", [])
                if messages:
                    _reflect_and_learn(cwd, messages)
                    console.print("[dim]Session learnings saved.[/dim]")
            except Exception:
                pass
            console.print("[dim]Bye![/dim]")
            break
        if task == "/clear":
            console.clear()
            print_banner()
            continue
        if task == "/memory":
            try:
                from agent.memory import load_all_blocks, ensure_codie_dir
                ensure_codie_dir(cwd)
                blocks = load_all_blocks(cwd)
                for b in blocks:
                    console.print(Panel(
                        b["content"] or "[dim]empty[/dim]",
                        title=f"[bold]{b['label']}[/bold] ({len(b['content'])}/{b['max_chars']} chars)",
                        border_style="magenta",
                    ))
            except Exception as e:
                console.print(f"[red]{e}[/red]")
            continue
        if task == "/init":
            task = (
                "Explore this codebase and build your initial memory:\n"
                "1. Use glob('**/*') to see the full structure\n"
                "2. Read README, config files (pyproject.toml, package.json, etc.), and key source files\n"
                "3. Use core_memory_replace on the 'project' block to write a concise summary of:\n"
                "   - Project purpose and structure\n"
                "   - Key technologies and dependencies\n"
                "   - Architecture patterns\n"
                "   - Important conventions or gotchas\n"
                "4. Update your 'persona' block with how you should approach this specific codebase\n"
                "Be thorough but concise. This memory will help you in future sessions."
            )
            console.print("[dim]Initializing memory... (exploring codebase)[/dim]")
        if task.startswith("/remember"):
            from agent.reflection import build_remember_prompt
            extra = task[len("/remember"):].strip()
            task = build_remember_prompt()
            if extra:
                task += f"\n\nThe user specifically wants you to remember: {extra}"
            console.print("[dim]Reflecting on conversation...[/dim]")

        console.print()
        try:
            run = await client.runs.create(
                thread_id,
                "agent",
                input={"messages": [{"role": "user", "content": task}]},
                config={"configurable": {"cwd": cwd}},
            )
            await client.runs.join(thread_id, run["run_id"])

            state = await client.threads.get_state(thread_id)
            messages = state.get("values", {}).get("messages", [])

            # Only print steps for NEW messages since last turn
            new_messages = messages[printed_msg_count:]
            printed_msg_count = len(messages)

            response = _print_steps(new_messages)

            if response:
                console.print()
                console.print(Markdown(response))
                console.print()
            else:
                console.print("[dim]No response.[/dim]")

        except Exception as e:
            console.print(f"[red]{e}[/red]")
            console.print("[dim]Is the server running? uv run langgraph dev --port 2025[/dim]\n")


def main():
    load_dotenv()
    print_banner()
    cwd = os.getcwd()
    console.print(f"[dim]{cwd}[/dim]\n")

    try:
        asyncio.run(session(cwd))
    except KeyboardInterrupt:
        console.print("\n[dim]Bye![/dim]")


if __name__ == "__main__":
    main()
