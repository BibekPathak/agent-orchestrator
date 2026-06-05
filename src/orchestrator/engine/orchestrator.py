from __future__ import annotations

import uuid
from typing import Any

from ..agents.planner import PlannerAgent
from ..agents.router import RouterAgent
from ..agents.synthesizer import SynthesizerAgent
from ..agents.research import ResearchAgent
from ..agents.finance import FinanceAgent
from ..agents.coding import CodingAgent
from ..agents.critic import CriticAgent
from ..agents.reviewer import ReviewerAgent
from ..core.agent import BaseAgent
from ..core.state import ExecutionState
from ..core.task import Plan, TaskNode, TaskStatus
from ..llm.base import LLM
from ..memory.short_term import ShortTermMemory
from ..memory.long_term import ChromaMemory
from ..observability import trace_span
from ..tools import DelegateTaskTool
from ..events import EventBus
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
        self._short_term_memory = ShortTermMemory()
        self._long_term_memory = ChromaMemory()
        self._retry_policy = retry_policy or RetryPolicy()
        self._event_bus = EventBus()

        self._agents[self._planner.name] = self._planner
        self._agents[self._router.name] = self._router
        self._agents[self._synthesizer.name] = self._synthesizer
        self._router.register_agent(self._planner)
        self._router.register_agent(self._synthesizer)
        # Register default agents
        research_agent = ResearchAgent(llm)
        finance_agent = FinanceAgent(llm)
        coding_agent = CodingAgent(llm)
        critic_agent = CriticAgent(llm)
        reviewer_agent = ReviewerAgent(llm)
        self.register_agent(research_agent)
        self.register_agent(finance_agent)
        self.register_agent(coding_agent)
        self.register_agent(critic_agent)
        self.register_agent(reviewer_agent)
        self._wire_delegate_tools()

    def _wire_delegate_tools(self) -> None:
        """Wire up DelegateTaskTool instances with orchestrator reference."""
        for agent in self._agents.values():
            for tool in getattr(agent, 'tools', []):
                if isinstance(tool, DelegateTaskTool):
                    tool.set_orchestrator(self)

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        self._router.register_agent(agent)
        
        from ..events.types import AgentRegisteredEvent
        self._event_bus.publish(AgentRegisteredEvent(
            agent_name=agent.name,
            agent_description=agent.description,
        ))

    def register_agents(self, agents: list[BaseAgent]) -> None:
        for a in agents:
            self.register_agent(a)

    @property
    def memory(self) -> ShortTermMemory:
        return self._short_term_memory

    @property
    def long_term_memory(self) -> ChromaMemory:
        return self._long_term_memory

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    async def execute(self, goal: str, session_id: str | None = None) -> str:
        session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        state = ExecutionState(session_id=session_id, goal=goal)

        from ..events.types import SessionStartedEvent, SessionCompletedEvent, SessionFailedEvent, WaveStartedEvent, WaveCompletedEvent
        
        await self._event_bus.publish(SessionStartedEvent(session_id=session_id, goal=goal))

        try:
            async with trace_span("plan") as span:
                if span:
                    span.set_attribute("session_id", session_id)
                    span.set_attribute("goal", goal)
                plan = await self._planner.plan(goal)
                for t in plan.tasks:
                    state.add_task(t)
                await self._short_term_memory.save(f"{session_id}:plan", plan.model_dump(mode="json"))
                await self._long_term_memory.save(f"plan:{session_id}", plan.model_dump(mode="json"))
                
                dag = plan.to_dag()
                if span:
                    span.set_attribute("dag_nodes", str(len(dag.get("nodes", []))))
                    span.set_attribute("dag_edges", str(len(dag.get("edges", []))))

            wave_num = 0
            for wave in plan.topological_waves():
                wave_num += 1
                await self._event_bus.publish(WaveStartedEvent(
                    wave_number=wave_num,
                    task_ids=[t.id for t in wave],
                    session_id=session_id,
                ))
                
                async with trace_span("execute_wave") as span:
                    if span:
                        span.set_attribute("session_id", session_id)
                        span.set_attribute("wave_tasks", str([t.id for t in wave]))
                    results = await self._execute_wave(wave, state)
                    state.context.update(results)
                
                await self._event_bus.publish(WaveCompletedEvent(
                    wave_number=wave_num,
                    task_ids=[t.id for t in wave],
                    session_id=session_id,
                ))

            async with trace_span("synthesize") as span:
                if span:
                    span.set_attribute("session_id", session_id)
                result = await self._synthesizer.synthesize(goal, state.completed_tasks)
                await self._short_term_memory.save(f"{session_id}:result", result)
                await self._long_term_memory.save(f"result:{session_id}", result)
            
            await self._event_bus.publish(SessionCompletedEvent(
                session_id=session_id,
                goal=goal,
                result=result,
            ))
            
            return result
            
        except Exception as e:
            await self._event_bus.publish(SessionFailedEvent(
                session_id=session_id,
                goal=goal,
                error=str(e),
            ))
            raise
    
    async def approve_task(self, task_id: str, approved_by: str) -> bool:
        """Approve a task that requires human approval."""
        task = self._short_term_memory.load(f"session_:task:{task_id}")
        if task:
            from ..core.task import TaskNode, TaskStatus
            from ..events.types import TaskApprovedEvent
            task_obj = TaskNode.model_validate_json(task)
            if task_obj.requires_approval:
                task_obj.approved_by = approved_by
                task_obj.status = TaskStatus.COMPLETED
                await self._short_term_memory.save(f"session_:task:{task_id}", task_obj.model_dump_json())
                await self._event_bus.publish(TaskApprovedEvent(
                    task_id=task_id,
                    approved_by=approved_by,
                    session_id=task_obj.session_id if hasattr(task_obj, 'session_id') else None,
                ))
                return True
        return False
    
    def get_dag(self, goal: str) -> dict[str, Any]:
        """Get the DAG representation of a goal (plans without executing)."""
        from ..llm import LLMConfig, create_llm
        
        llm = create_llm(LLMConfig(provider=self._llm.config.provider, model=self._llm.config.model))
        planner = PlannerAgent(llm=llm)
        
        import asyncio
        import concurrent.futures
        
        def _plan():
            async def plan_only():
                return await planner.plan(goal)
            return asyncio.run(plan_only())
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_plan)
            plan = future.result()
        
        return plan.to_dag()

    async def _execute_wave(
        self,
        wave: list[TaskNode],
        state: ExecutionState,
    ) -> dict[str, Any]:
        import asyncio
        from ..core.task import TaskStatus
        from ..events.types import TaskStartedEvent, TaskCompletedEvent, TaskFailedEvent, TaskApprovalRequiredEvent

        async def run_task(task: TaskNode) -> tuple[str, Any]:
            if task.requires_approval and task.status != TaskStatus.COMPLETED:
                state.update_task(task.id, status=TaskStatus.WAITING_APPROVAL, 
                                error="Requires human approval")
                await self._event_bus.publish(TaskApprovalRequiredEvent(
                    task_id=task.id,
                    task_description=task.description,
                    session_id=state.session_id,
                ))
                return task.id, None
                
            agent_name = await self._router.select(task)
            if agent_name is None or agent_name not in self._agents:
                state.update_task(task.id, status=TaskStatus.FAILED, error=f"No agent found")
                await self._event_bus.publish(TaskFailedEvent(
                    task_id=task.id,
                    task_description=task.description,
                    agent=agent_name or "unknown",
                    error="No agent found",
                    session_id=state.session_id,
                ))
                return task.id, None

            agent = self._agents[agent_name]
            state.update_task(task.id, status=TaskStatus.RUNNING, agent=agent_name)
            
            await self._event_bus.publish(TaskStartedEvent(
                task_id=task.id,
                task_description=task.description,
                agent=agent_name,
                session_id=state.session_id,
            ))

            try:
                async with trace_span("run_task") as span:
                    if span:
                        span.set_attribute("task_id", task.id)
                        span.set_attribute("task_description", task.description)
                        span.set_attribute("agent", agent_name)
                        span.set_attribute("requires_approval", str(task.requires_approval))
                    result = await with_retry(
                        agent.run,
                        task=task,
                        state=state,
                        policy=self._retry_policy,
                    )
                    if span:
                        span.set_attribute("status", "completed")
                    state.update_task(task.id, status=TaskStatus.COMPLETED, result=result)
                    
                    await self._event_bus.publish(TaskCompletedEvent(
                        task_id=task.id,
                        task_description=task.description,
                        agent=agent_name,
                        result=result,
                        session_id=state.session_id,
                    ))
                    
                    return task.id, result
            except Exception as e:
                state.update_task(task.id, status=TaskStatus.FAILED, error=str(e))
                await self._event_bus.publish(TaskFailedEvent(
                    task_id=task.id,
                    task_description=task.description,
                    agent=agent_name,
                    error=str(e),
                    session_id=state.session_id,
                ))
                return task.id, None

        tasks = [run_task(t) for t in wave]
        completed = await asyncio.gather(*tasks)
        return dict(completed)
