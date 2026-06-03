from __future__ import annotations

import uuid
from typing import Any

from ..core.agent import BaseAgent
from ..core.state import ExecutionState
from ..core.task import Plan, Task, TaskStatus
from ..llm.base import LLM

PLANNER_SYSTEM_PROMPT = """You are a planning agent. Given a user goal, decompose it into a sequence of specific, actionable tasks.

Each task must have:
- A clear description of what to do
- Its dependencies (list of task IDs it depends on)
- The type of agent best suited (research, code, finance, write, analyze, critic, reviewer)

Available agent types:
- research: for web research, fact-finding, gathering information
- code: for writing, running, and debugging code
- finance: for financial data analysis and stock information
- write: for creating written content, reports, summaries
- analyze: for data analysis and interpretation
- critic: for reviewing and critiquing outputs from other agents
- reviewer: for final validation and approval of completed work

Respond ONLY with a JSON array of tasks in this format:
[
  {
    "id": "t1",
    "description": "Collect company info on Nvidia",
    "agent": "research",
    "deps": []
  },
  {
    "id": "t2",
    "description": "Gather latest news about Nvidia",
    "agent": "research",
    "deps": []
  },
  {
    "id": "t3",
    "description": "Analyze stock metrics for Nvidia",
    "agent": "finance",
    "deps": ["t1", "t2"]
  },
  {
    "id": "t4",
    "description": "Review the financial analysis for accuracy",
    "agent": "critic",
    "deps": ["t3"]
  },
  {
    "id": "t5",
    "description": "Final validation of the investment report",
    "agent": "reviewer",
    "deps": ["t4"]
  },
  {
    "id": "t6",
    "description": "Generate investment report",
    "agent": "write",
    "deps": ["t5"]
  }
]

Dependencies mean: a task should only run AFTER all its dependencies are complete.
Use deps: [] for tasks that can start immediately."""


class PlannerAgent(BaseAgent):
    def __init__(self, llm: LLM, name: str = "planner", description: str = "Decomposes goals into task plans") -> None:
        super().__init__(name=name, description=description)
        self._llm = llm

    async def run(self, task: Task, state: ExecutionState | None = None) -> str:
        plan = await self.plan(task.description)
        return plan.model_dump_json(indent=2)

    async def plan(self, goal: str) -> Plan:
        content, tool_calls = await self._llm.generate(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": goal}],
        )

        import json
        json_str = _extract_json(content)
        tasks_data = json.loads(json_str)
        if isinstance(tasks_data, dict):
            tasks_data = tasks_data.get("tasks", [tasks_data])

        tasks = []
        for td in tasks_data:
            tasks.append(Task(
                id=td.get("id", f"t_{uuid.uuid4().hex[:6]}"),
                description=td["description"],
                agent=td.get("agent"),
                deps=td.get("deps", []),
                status=TaskStatus.PENDING,
            ))

        return Plan(goal=goal, tasks=tasks)


def _extract_json(text: str) -> str:
    start_brace = text.find("{")
    start_bracket = text.find("[")
    if start_brace == -1 and start_bracket == -1:
        return text
    if start_bracket == -1 or (start_brace != -1 and start_brace < start_bracket):
        start, open_ch, close_ch = start_brace, "{", "}"
    else:
        start, open_ch, close_ch = start_bracket, "[", "]"
    if start == -1:
        return text
    depth = 0
    end = start
    for i, ch in enumerate(text[start:], start):
        if ch == open_ch or (open_ch == "{" and ch == "["):
            depth += 1
        elif ch == close_ch or (close_ch == "}" and ch == "]"):
            depth -= 1
        if depth == 0:
            end = i + 1
            break
    return text[start:end]
