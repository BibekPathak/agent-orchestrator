from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.state import ExecutionState
from ..core.task import Task
from ..llm.base import LLM

SYNTHESIZER_SYSTEM_PROMPT = """You are a synthesizer agent. Given a user's original goal and the results from multiple completed tasks, combine them into a coherent, well-structured final response.

Be thorough and include all relevant findings from each task result."""


class SynthesizerAgent(BaseAgent):
    def __init__(self, llm: LLM, name: str = "synthesizer", description: str = "Combines task results into a final response") -> None:
        super().__init__(name=name, description=description)
        self._llm = llm

    async def synthesize(self, goal: str, completed: list[Task]) -> str:
        results_str = "\n\n".join(
            f"Task: {t.description}\nResult: {t.result}"
            for t in completed if t.result
        )
        prompt = f"Original goal: {goal}\n\nTask results:\n{results_str}\n\nCombine these into a final response."
        content, _ = await self._llm.generate(
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return content

    async def run(self, task: Task, state: ExecutionState | None = None) -> str:
        if state is None:
            return ""
        return await self.synthesize(state.goal, state.completed_tasks)
