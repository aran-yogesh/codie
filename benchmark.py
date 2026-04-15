"""Benchmark: Memory ON vs OFF — Style Adherence.

Measures whether the agent follows learned coding preferences across sessions.

Setup:
  1. WITHOUT MEMORY: Run 5 coding tasks on a fresh agent (no preferences taught)
  2. WITH MEMORY: Teach 3 preferences, then run the same 5 tasks in NEW sessions
  3. Score each task: does the code follow the 3 preferences?
  4. Compare scores

Usage:
  uv run langgraph dev --port 2025 --no-reload
  uv run python3 benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import datetime

CWD = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("LANGGRAPH_API_URL", "http://localhost:2025")

PREFERENCES = [
    ("type hints", lambda code: "-> " in code and ": " in code.split("->")[0] if "->" in code else False),
    ("docstrings", lambda code: '"""' in code or "'''" in code),
    ("raises ValueError", lambda code: "raise ValueError" in code or "raise TypeError" in code or "ValueError" in code),
]

TASKS = [
    "Write a Python function that reverses a linked list. Reply with code inline, do NOT use write tool.",
    "Write a Python function that finds the second largest number in a list. Reply with code inline, do NOT use write tool.",
    "Write a Python function that converts celsius to fahrenheit. Reply with code inline, do NOT use write tool.",
    "Write a Python function that counts vowels in a string. Reply with code inline, do NOT use write tool.",
    "Write a Python function that merges two sorted lists into one sorted list. Reply with code inline, do NOT use write tool.",
]

TEACHING_MSG = (
    "I have strict coding standards. Save ALL of these to your user memory using core_memory_append:\n"
    "1. ALWAYS add type hints on every parameter and return type\n"
    "2. ALWAYS include a docstring on every function\n"
    "3. ALWAYS validate inputs and raise ValueError for invalid arguments\n"
    "These are non-negotiable. Save them now."
)


async def ask(client, tid, msg, cwd, timeout=120):
    r = await client.runs.create(
        tid, "agent",
        input={"messages": [{"role": "user", "content": msg}]},
        config={"configurable": {"cwd": cwd}},
    )
    try:
        await asyncio.wait_for(client.runs.join(tid, r["run_id"]), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    state = await client.threads.get_state(tid)
    msgs = state.get("values", {}).get("messages", [])
    all_text = []
    for m in msgs:
        t = m.get("type") if isinstance(m, dict) else getattr(m, "type", "")
        if t == "ai":
            c = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            if isinstance(c, list):
                c = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
            if c.strip():
                all_text.append(c)
    return all_text[-1] if all_text else ""


def score_response(code: str) -> dict:
    results = {}
    for name, checker in PREFERENCES:
        try:
            results[name] = 1 if checker(code) else 0
        except Exception:
            results[name] = 0
    return results


async def run_benchmark(client, label: str, cwd: str, teach: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if teach:
        t = await client.threads.create()
        print(f"  Teaching preferences...")
        await ask(client, t["thread_id"], TEACHING_MSG, cwd)
        from agent.memory import load_block
        user = load_block(cwd, "user")
        print(f"  User memory: \"{user['content'][:100]}...\"")

    all_scores = []
    task_details = []
    for i, task in enumerate(TASKS):
        # Each task in a NEW thread (separate session)
        t = await client.threads.create()
        tid = t["thread_id"]
        print(f"\n  Task {i+1}: {task[:60]}...")
        resp = await ask(client, tid, task, cwd)
        scores = score_response(resp)
        all_scores.append(scores)
        total = sum(scores.values())
        print(f"    Score: {total}/{len(PREFERENCES)} — {scores}")
        task_details.append({"task": task[:60], "scores": scores, "total": total})

    # Aggregate
    agg = {name: 0 for name, _ in PREFERENCES}
    for s in all_scores:
        for k, v in s.items():
            agg[k] += v

    total_possible = len(TASKS) * len(PREFERENCES)
    total_got = sum(agg.values())
    pct = (total_got / total_possible) * 100

    print(f"\n  {'─'*40}")
    print(f"  TOTAL: {total_got}/{total_possible} ({pct:.0f}%)")
    for name, count in agg.items():
        print(f"    {name}: {count}/{len(TASKS)}")

    return {
        "label": label,
        "total": total_got,
        "max": total_possible,
        "pct": pct,
        "detail": agg,
        "tasks": task_details,
    }


async def main():
    from langgraph_sdk import get_client
    client = get_client(url=URL)

    print(f"\n  CODIE BENCHMARK — Style Adherence")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- RUN 1: WITHOUT MEMORY ---
    codie_dir = os.path.join(CWD, ".codie")
    if os.path.exists(codie_dir):
        shutil.rmtree(codie_dir)
    print("\n  Cleared all memory.")

    no_mem = await run_benchmark(client, "WITHOUT MEMORY (baseline)", CWD, teach=False)

    # --- RUN 2: WITH MEMORY ---
    if os.path.exists(codie_dir):
        shutil.rmtree(codie_dir)

    with_mem = await run_benchmark(client, "WITH MEMORY (learned preferences)", CWD, teach=True)

    # --- COMPARISON ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<25} {'No Memory':>12} {'With Memory':>12} {'Delta':>8}")
    print(f"  {'─'*57}")
    print(f"  {'Overall Score':<25} {no_mem['pct']:>11.0f}% {with_mem['pct']:>11.0f}% {with_mem['pct']-no_mem['pct']:>+7.0f}%")
    for name, _ in PREFERENCES:
        n = no_mem["detail"][name]
        w = with_mem["detail"][name]
        print(f"  {name:<25} {n:>9}/{len(TASKS)} {w:>9}/{len(TASKS)} {w-n:>+7}")
    print(f"\n  Total: {no_mem['total']}/{no_mem['max']} → {with_mem['total']}/{with_mem['max']}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "without_memory": no_mem,
        "with_memory": with_mem,
    }
    results_path = os.path.join(CWD, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
