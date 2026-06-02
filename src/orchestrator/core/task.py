from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    id: str
    description: str
    agent: str | None = None
    deps: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    goal: str
    tasks: list[Task]

    def dependency_graph(self) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        for t in self.tasks:
            graph[t.id] = t.deps
        return graph

    def topological_waves(self) -> list[list[Task]]:
        remaining = {t.id: t for t in self.tasks}
        completed: set[str] = set()
        waves: list[list[Task]] = []

        while remaining:
            wave = [
                t
                for t in remaining.values()
                if all(d in completed and remaining.get(d) is None or d in completed for d in t.deps)
            ]
            wave = [t for t in wave if all(d in completed for d in t.deps)]
            if not wave:
                remaining_ids = list(remaining.keys())
                wave = [remaining[remaining_ids[0]]]
            waves.append(wave)
            for t in wave:
                completed.add(t.id)
                del remaining[t.id]

        return waves
