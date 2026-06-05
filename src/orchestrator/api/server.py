from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..engine.orchestrator import AgentOrchestrator
from ..llm.base import LLMConfig
from ..llm import create_llm
from .schemas import (
    AgentRegisterRequest,
    ExecuteRequest,
    ExecuteResponse,
    StatusResponse,
)


class AppState:
    def __init__(self) -> None:
        llm = create_llm(LLMConfig())
        self.orchestrator = AgentOrchestrator(llm=llm)
        self.sessions: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state = AppState()
    yield


app = FastAPI(
    title="AI Agent Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "AI Agent Orchestrator",
        "version": "0.1.0",
        "endpoints": {
            "POST /run": "Execute a goal",
            "POST /dag": "Generate DAG for a goal (no execution)",
            "GET /dag/{session_id}": "Get DAG for existing session",
            "GET /events": "Get event history",
            "GET /events/types": "Get available event types",
            "DELETE /events": "Clear event history",
            "GET /marketplace/agents": "List marketplace agents",
            "GET /marketplace/agents/{name}": "Get agent details",
            "GET /marketplace/capabilities": "Get available capabilities",
            "GET /marketplace/capabilities/{cap}/agents": "Get agents by capability",
            "GET /marketplace/stats": "Get marketplace stats",
            "GET /cost/summary": "Get cost summary",
            "GET /cost/session/{session_id}": "Get session cost",
            "GET /cost/agent/{agent_name}": "Get agent cost",
            "GET /cost/models": "Get costs by model",
            "GET /cost/pricing": "Get pricing models",
            "DELETE /cost": "Clear cost data",
            "GET /evaluation/summary": "Get evaluation summary",
            "GET /evaluation/agents": "Get all agent evaluations",
            "GET /evaluation/agents/{name}": "Get agent evaluation",
            "GET /evaluation/leaderboard": "Get agent leaderboard",
            "GET /evaluation/sessions/{id}": "Get session evaluation",
            "GET /evaluation/tasks": "Get task evaluations",
            "DELETE /evaluation": "Clear evaluation data",
            "GET /status/{session_id}": "Get session result",
            "POST /agents/register": "Register a custom agent",
            "GET /agents": "List registered agents",
            "GET /health": "Health check",
        },
    }


@app.post("/run", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest) -> dict[str, Any]:
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:12]}"
    result = await app.state.orchestrator.execute(req.goal, session_id=session_id)
    app.state.sessions[session_id] = result
    return ExecuteResponse(session_id=session_id, result=result)


@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str) -> dict[str, Any]:
    result = app.state.sessions.get(session_id)
    if result is None:
        result_data = await app.state.orchestrator.memory.load(f"{session_id}:result")
        if result_data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return StatusResponse(session_id=session_id, result=result_data, complete=True)

    plan_data = await app.state.orchestrator.memory.load(f"{session_id}:plan")
    return StatusResponse(
        session_id=session_id,
        result=result,
        complete=True,
        plan=plan_data,
    )


@app.post("/agents/register")
async def register_agent(req: AgentRegisterRequest) -> dict[str, str]:
    from ..core.agent import BaseAgent

    class DynamicAgent(BaseAgent):
        async def run(self, task, state=None):
            llm = create_llm(LLMConfig())
            content, _ = await llm.generate(
                system_prompt=req.system_prompt or f"You are {req.name}. {req.description}",
                messages=[{"role": "user", "content": task.description}],
            )
            return content

    agent = DynamicAgent(name=req.name, description=req.description)
    app.state.orchestrator.register_agent(agent)
    return {"status": "registered", "name": req.name}


@app.get("/agents")
async def list_agents() -> list[dict[str, Any]]:
    from ..core.agent import BaseAgent
    return [
        a.to_registration()
        for a in app.state.orchestrator._agents.values()
    ]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/dag")
async def get_dag(req: ExecuteRequest) -> dict[str, Any]:
    """Generate and return DAG for a goal without executing."""
    dag = app.state.orchestrator.get_dag(req.goal)
    return dag


@app.get("/dag/{session_id}")
async def get_session_dag(session_id: str) -> dict[str, Any]:
    """Get the DAG for an existing session."""
    plan_data = await app.state.orchestrator.memory.load(f"{session_id}:plan")
    if plan_data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    from ..core.task import Plan
    plan = Plan.model_validate(plan_data)
    return plan.to_dag()


@app.get("/events")
async def get_events(
    event_type: str | None = None,
    session_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get event history."""
    from ..events.types import EventType
    
    event_type_enum = None
    if event_type:
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
    
    return app.state.orchestrator.event_bus.get_history(
        event_type=event_type_enum,
        session_id=session_id,
        limit=limit,
    )


@app.get("/events/types")
async def get_event_types() -> list[str]:
    """Get list of available event types."""
    from ..events.types import EventType
    return [e.value for e in EventType]


@app.delete("/events")
async def clear_events() -> dict[str, str]:
    """Clear event history."""
    app.state.orchestrator.event_bus.clear_history()
    return {"status": "cleared"}


@app.get("/marketplace/agents")
async def list_marketplace_agents() -> list[dict[str, Any]]:
    """List all agents in marketplace."""
    return [r.to_dict() for r in app.state.orchestrator.marketplace.list_all()]


@app.get("/marketplace/agents/{name}")
async def get_marketplace_agent(name: str) -> dict[str, Any]:
    """Get agent details from marketplace."""
    agent = app.state.orchestrator.marketplace.get(name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {name} not found")
    return agent.to_dict()


@app.get("/marketplace/capabilities")
async def get_capabilities() -> list[str]:
    """Get available capabilities."""
    from ..marketplace.types import Capability
    return [c.value for c in Capability]


@app.get("/marketplace/capabilities/{capability}/agents")
async def get_agents_by_capability(capability: str) -> list[dict[str, Any]]:
    """Get agents with specific capability."""
    from ..marketplace.types import Capability
    try:
        cap = Capability(capability)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid capability: {capability}")
    
    agents = app.state.orchestrator.marketplace.find_by_capability(cap)
    return [a.to_dict() for a in agents]


@app.get("/marketplace/stats")
async def get_marketplace_stats() -> dict[str, Any]:
    """Get marketplace statistics."""
    return {
        "total_agents": len(app.state.orchestrator.marketplace.list_all()),
        "capabilities": app.state.orchestrator.marketplace.get_capability_stats(),
    }


@app.get("/cost/summary")
async def get_cost_summary() -> dict[str, Any]:
    """Get cost summary for all sessions and agents."""
    return app.state.orchestrator.cost_tracker.get_summary()


@app.get("/cost/session/{session_id}")
async def get_session_cost(session_id: str) -> dict[str, Any]:
    """Get cost for a specific session."""
    cost = app.state.orchestrator.cost_tracker.get_session_cost(session_id)
    records = app.state.orchestrator.cost_tracker.get_records(session_id=session_id)
    return {
        "session_id": session_id,
        "total_cost": cost,
        "records": records,
    }


@app.get("/cost/agent/{agent_name}")
async def get_agent_cost(agent_name: str) -> dict[str, Any]:
    """Get cost for a specific agent."""
    cost = app.state.orchestrator.cost_tracker.get_agent_cost(agent_name)
    records = app.state.orchestrator.cost_tracker.get_records(agent_name=agent_name)
    return {
        "agent_name": agent_name,
        "total_cost": cost,
        "records": records,
    }


@app.get("/cost/models")
async def get_model_costs() -> list[dict[str, Any]]:
    """Get costs grouped by model."""
    summary = app.state.orchestrator.cost_tracker.get_summary()
    return [
        {"model": model, "cost": cost}
        for model, cost in summary.get("model_costs", {}).items()
    ]


@app.get("/cost/pricing")
async def get_pricing() -> dict[str, Any]:
    """Get available pricing models."""
    from ..cost import MODEL_PRICING
    return MODEL_PRICING


@app.get("/evaluation/summary")
async def get_evaluation_summary() -> dict[str, Any]:
    """Get evaluation summary."""
    return app.state.orchestrator.evaluator.get_summary()


@app.get("/evaluation/agents")
async def get_agent_evaluations() -> dict[str, Any]:
    """Get evaluations for all agents."""
    return app.state.orchestrator.evaluator.get_agent_metrics()


@app.get("/evaluation/agents/{agent_name}")
async def get_agent_evaluation(agent_name: str) -> dict[str, Any]:
    """Get evaluation for a specific agent."""
    metrics = app.state.orchestrator.evaluator.get_agent_metrics(agent_name)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No evaluation found for agent: {agent_name}")
    return metrics


@app.get("/evaluation/leaderboard")
async def get_leaderboard(limit: int = 10) -> list[dict[str, Any]]:
    """Get agent leaderboard by success rate."""
    return app.state.orchestrator.evaluator.get_leaderboard(limit=limit)


@app.get("/evaluation/sessions/{session_id}")
async def get_session_evaluation(session_id: str) -> dict[str, Any]:
    """Get evaluation for a specific session."""
    evaluation = app.state.orchestrator.evaluator.get_session_evaluation(session_id)
    if evaluation is None:
        raise HTTPException(status_code=404, detail=f"No evaluation found for session: {session_id}")
    return evaluation


@app.get("/evaluation/tasks")
async def get_task_evaluations(
    session_id: str | None = None,
    agent_name: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get task evaluations."""
    return app.state.orchestrator.evaluator.get_task_evaluations(
        session_id=session_id,
        agent_name=agent_name,
        limit=limit,
    )


@app.delete("/evaluation")
async def clear_evaluation() -> dict[str, str]:
    """Clear evaluation data."""
    app.state.orchestrator.evaluator.clear()
    return {"status": "cleared"}


@app.delete("/cost")
async def clear_costs() -> dict[str, str]:
    """Clear cost tracking data."""
    app.state.orchestrator.cost_tracker.clear()
    return {"status": "cleared"}
