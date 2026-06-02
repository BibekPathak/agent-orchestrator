from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from .base import LLM, LLMConfig


class OpenAILLM(LLM):
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(provider="openai")
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if messages:
            msgs.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": msgs,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        return content, tool_calls
