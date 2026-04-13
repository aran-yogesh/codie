"""Benchmark for codie — measures tool calls, wrong attempts, time, and tokens."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from agent import memory

SERVER_URL = os.environ.get("LANGGRAPH_API_URL", "http://localhost:2025")

# Benchmark tasks — each has a task description and a way to verify success
TASKS = [
    {
        "name": "create_file",
        "task": "Create a file called hello.py that prints 'Hello, World!'",
        "verify_cmd": "python3 hello.py",
        "expected_output": "Hello, World!",
    },
    {
        "name": "read_and_summarize",
        "task": "Read the file hello.py and tell me what it does in one sentence.",
        "verify": lambda response: "hello" in response.lower() or "print" in response.lower(),
    },
    {
        "name": "edit_file",
        "task": "Edit hello.py to also print 'Goodbye, World!' on a second line.",
        "verify_cmd": "python3 hello.py",
        "expected_output": "Hello, World!\nGoodbye, World!",
    },
    {
        "name": "create_project",
        "task": "Create a Python file called calc.py with add, subtract, multiply, divide functions. Then create test_calc.py with tests for each. Run the tests.",
        "verify_cmd": "python3 -m pytest test_calc.py -v 2>&1 | tail -1",
        "expected_contains": "passed",
    },
    {
        "name": "find_and_fix",
        "task": "There's a bug in calc.py — the divide function doesn't handle division by zero. Fix it to return 'Error: division by zero' instead of crashing.",
        "verify_cmd": "python3 -c \"from calc import divide; print(divide(1, 0))\"",
        "expected_output": "Error: division by zero",
    },
    {
        "name": "glob_and_grep",
        "task": "Find all .py files in this directory and tell me which ones contain the word 'def'.",
        "verify": lambda response: "calc.py" in response and "hello.py" not in response,
    },
]


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


def _count_metrics(messages: list) -> dict:
    """Count tool calls, errors, and token usage from messages."""
    tool_calls = 0
    tool_errors = 0
    tools_used = {}

    for msg in messages:
        if isinstance(msg, dict):
            msg_type = msg.get("type", "")
            content = msg.get("content", "")
        else:
            msg_type = getattr(msg, "type", "")
            content = getattr(msg, "content", "")

        if msg_type == "ai":
            tc_list = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
            for tc in (tc_list or []):
                tool_calls += 1
                name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                tools_used[name] = tools_used.get(name, 0) + 1

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_calls += 1
                        name = block.get("name", "")
                        tools_used[name] = tools_used.get(name, 0) + 1

        elif msg_type == "tool":
            text = _extract_text(content)
            if any(err in text for err in ["[error", "[file not found", "[old_string not found", "[exit code:"]):
                tool_errors += 1

    # Token usage from response_metadata
    input_tokens = 0
    output_tokens = 0
    for msg in messages:
        meta = msg.get("response_metadata", {}) if isinstance(msg, dict) else getattr(msg, "response_metadata", {})
        if meta:
            usage = meta.get("usage", {})
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    return {
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "tools_used": tools_used,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


async def run_benchmark():
    import subprocess
    from langgraph_sdk import get_client

    client = get_client(url=SERVER_URL)

    # Setup: create a temp workspace
    workspace = "/tmp/codie-benchmark"
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace)

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    results = []
    total_start = time.time()
    msg_offset = 0

    mem_status = "ON" if memory.is_enabled() else "OFF"
    print(f"\n{'='*70}")
    print(f"  CODIE BENCHMARK — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workspace: {workspace}")
    print(f"  Memory: {mem_status}")
    print(f"{'='*70}\n")

    for i, task_def in enumerate(TASKS, 1):
        name = task_def["name"]
        task = task_def["task"]

        print(f"[{i}/{len(TASKS)}] {name}")
        print(f"  Task: {task[:80]}{'...' if len(task) > 80 else ''}")

        # RECALL: inject memories before task
        use_memory = memory.is_enabled()
        enriched_task = task
        if use_memory:
            memories = memory.recall(task)
            if memories:
                print(f"  ● recalled {len(memories)} memories")
                enriched_task = memory.format_for_injection(memories) + "\n" + task

        start = time.time()

        try:
            run = await client.runs.create(
                thread_id,
                "agent",
                input={"messages": [{"role": "user", "content": enriched_task}]},
                config={"configurable": {"cwd": workspace}},
            )
            run_id = run["run_id"]

            # Poll until run completes (join can return early)
            for _ in range(120):  # 2 min max
                run_state = await client.runs.get(thread_id, run_id)
                status = run_state.get("status", "")
                # debug removed
                if status in ("success", "error", "interrupted"):
                    if status == "error":
                        print(f"  [error] {run_state}")
                    break
                await asyncio.sleep(1)
            else:
                print(f"  ⚠ Run timed out after 120s")

            state = await client.threads.get_state(thread_id)
            messages = state.get("values", {}).get("messages", [])
            new_messages = messages[msg_offset:]
            msg_offset = len(messages)

            elapsed = time.time() - start
            metrics = _count_metrics(new_messages)

            # Get final response
            response = ""
            for msg in reversed(new_messages):
                mt = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
                if mt == "ai":
                    response = _extract_text(
                        msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    )
                    if response.strip():
                        break

            # Verify
            passed = False
            verify_detail = ""

            if "verify_cmd" in task_def:
                try:
                    r = subprocess.run(
                        task_def["verify_cmd"], shell=True, cwd=workspace,
                        capture_output=True, text=True, timeout=10,
                    )
                    actual = r.stdout.strip()
                    if "expected_output" in task_def:
                        passed = actual == task_def["expected_output"]
                        verify_detail = f"expected='{task_def['expected_output']}' got='{actual}'"
                    elif "expected_contains" in task_def:
                        passed = task_def["expected_contains"] in actual
                        verify_detail = f"looking for '{task_def['expected_contains']}' in '{actual}'"
                except Exception as e:
                    verify_detail = f"verify error: {e}"

            elif "verify" in task_def:
                try:
                    passed = task_def["verify"](response)
                    verify_detail = "response check"
                except Exception as e:
                    verify_detail = f"verify error: {e}"

            status = "PASS" if passed else "FAIL"
            result = {
                "name": name,
                "status": status,
                "time_s": round(elapsed, 2),
                "tool_calls": metrics["tool_calls"],
                "tool_errors": metrics["tool_errors"],
                "tools_used": metrics["tools_used"],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "total_tokens": metrics["total_tokens"],
                "verify": verify_detail,
            }
            results.append(result)

            # LEARN: extract facts and store in Mem0
            if use_memory and new_messages:
                facts = memory.extract_facts(new_messages)
                if facts:
                    stored = memory.learn(facts)
                    print(f"  ● learned {stored} facts")

            status_color = "✅" if passed else "❌"
            print(f"  {status_color} {status} | {elapsed:.1f}s | {metrics['tool_calls']} calls | {metrics['tool_errors']} errors | {metrics['total_tokens']} tokens")
            if metrics["tools_used"]:
                print(f"  Tools: {metrics['tools_used']}")
            if not passed:
                print(f"  Verify: {verify_detail}")

        except Exception as e:
            elapsed = time.time() - start
            results.append({
                "name": name,
                "status": "ERROR",
                "time_s": round(elapsed, 2),
                "error": str(e),
                "tool_calls": 0,
                "tool_errors": 0,
                "tools_used": {},
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            })
            print(f"  ❌ ERROR | {elapsed:.1f}s | {e}")

        print()

    total_elapsed = time.time() - total_start

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    total_tool_calls = sum(r["tool_calls"] for r in results)
    total_tool_errors = sum(r["tool_errors"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)

    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Tasks:       {passed}/{len(TASKS)} passed, {failed} failed, {errors} errors")
    print(f"  Time:        {total_elapsed:.1f}s total")
    print(f"  Tool calls:  {total_tool_calls} total, {total_tool_errors} errors ({(total_tool_errors/max(total_tool_calls,1)*100):.0f}% error rate)")
    print(f"  Tokens:      {total_tokens:,} total")
    print(f"{'='*70}\n")

    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tasks": len(TASKS),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total_time_s": round(total_elapsed, 2),
            "total_tool_calls": total_tool_calls,
            "total_tool_errors": total_tool_errors,
            "total_tokens": total_tokens,
        },
        "tasks": results,
    }

    report_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to: {report_path}")

    # Cleanup
    shutil.rmtree(workspace, ignore_errors=True)


def main():
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
