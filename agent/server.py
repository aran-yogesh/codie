"""Agent factory for LangGraph server."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from agent.prompt import build_system_prompt
from agent.tools import ALL_TOOLS

MODEL = os.environ.get("LLM_MODEL_ID", "claude-sonnet-4-20250514")


async def get_agent(config: RunnableConfig):
    """Create the coding agent. Entry point for langgraph.json."""
    cwd = config.get("configurable", {}).get("cwd", "/tmp")

    model = ChatAnthropic(
        model=MODEL,
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    system_prompt = build_system_prompt(cwd)

    return create_react_agent(model, ALL_TOOLS, prompt=system_prompt)
