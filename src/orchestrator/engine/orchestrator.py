from __future__ import annotations

import uuid
from typing import Any

from ..agents.planner import PlannerAgent
from ..agents.router import RouterAgent
from ..agents.synthesizer import SynthesizerAgent
from ..core.agent import BaseAgent
from ..core.state import ExecutionState
from ..core.task import Plan, Task, TaskStatus
from ..llm.base import LLM
from ..memory.short_term import ShortTermMemory
from .retry import RetryPolicy, with_retry


class AgentOrchestrator:
    def __init__(
        self,
        llm: LLM,
        planner: PlannerAgent | None = None,
        router: RouterAgent | None = None,
        synthesizer: SynthesizerAgent | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._llm = llm
        self._planner = planner or PlannerAgent(llm)
        self._router = router or RouterAgent(llm)
        self._synthesizer = synthesizer or SynthesizerAgent(llm)
        self._agents: dict[str, BaseAgent] = {}
        self._memory = ShortTermMemory()
        self._retry_policy = retry_policy or RetryPolicy()

        self._agents[self._planner.name] = self._planner
        self._agents[self._router.name] = self._router
        self._agents[self._synthesizer.name] = self._synthesizer
        self._router.register_agent(self._planner)
        self._router.register_agent(self._synthesizer)

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        self._router.register_agent(agent)

    def register_agents(self, agents: list[BaseAgent]) -> None:
        for a in agents:
            self.register_agent(a)

    @property
    def memory(self) -> ShortTermMemory:
        return self._memory

    async def execute(self, goal: str, session_id: str | None = None) -> str:
        session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        state = ExecutionState(session_id=session_id, goal=goal)

        plan = await self._planner.plan(goal)
        for t in plan.tasks:
            state.add_task(t)

        await self._memory.save(f"{session_id}:plan", plan.model_dump(mode="json"))

        for wave in plan.topological_waves():
            results = await self._execute_wave(wave, state)
            state.context.update(results)

        result = await self._synthesizer.synthesize(goal, state.completed_tasks)
        await self._memory.save(f"{session_id}:result", result)
        return result

    async def _execute_wave(
        self,
        wave: list[Task],
        state: ExecutionState,
    ) -> dict[str, Any]:
        import asyncio

        async def run_task(task: Task) -> tuple[str, Any]:
            agent_name = await self._router.select(task)
            if agent_name is None or agent_name not in self._agents:
                state.update_task(task.id, status=TaskStatus.FAILED, error=f"No agent found")
                return task.id, None

            agent = self._agents[agent_name]
            state.update_task(task.id, status=TaskStatus.RUNNING, agent=agent_name)

            try:
                result = await with_retry(
                    agent.run,
                    task=task,
                    state=state,
                    policy=self._retry_policy,
                )
                state.update_task(task.id, status=TaskStatus.COMPLETED, result=result)
                return task.id, result
            except Exception as e:
                state.update_task(task.id, status=TaskStatus.FAILED, error=str(e))
                return task.id, None

        tasks = [run_task(t) for t in wave]
        completed = await asyncio.gather(*tasks)
        return dict(completed)
