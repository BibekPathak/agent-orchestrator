from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    provider: str = "huggingface"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.3
    max_tokens: int = 4096
    api_key: str | None = None


class LLM(ABC):

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        ...
