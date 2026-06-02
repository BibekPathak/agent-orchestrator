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
