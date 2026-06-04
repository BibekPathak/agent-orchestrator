from __future__ import annotations

import json
from typing import Any

from ..core.agent import BaseAgent
from ..core.state import ExecutionState
from ..llm.base import LLM
from ..tools import (
    DatabaseTool,
    DelegateTaskTool,
    EditFileTool,
    GitHubTool,
    ListDirectoryTool,
    PythonExecutionTool,
    ReadFileTool,
    WriteFileTool,
)


class CodingAgent(BaseAgent):
    def __init__(self, llm: LLM, name: str = "coding", description: str = "Writes, runs, and debugs code using Python execution and file operation tools. Can also interact with GitHub, databases, and delegate tasks to other agents.") -> None:
        super().__init__(
            name=name,
            description=description,
            tools=[
                PythonExecutionTool(),
                ReadFileTool(),
                WriteFileTool(),
                EditFileTool(),
                ListDirectoryTool(),
                GitHubTool(),
                DatabaseTool(),
                DelegateTaskTool(),
            ],
        )
        self._llm = llm

    async def run(self, task: Any, state: ExecutionState | None = None) -> str:
        """Execute a coding task using LLM-driven tool calling."""
        query = getattr(task, 'description', str(task))
        tools_schema = [tool.to_llm_tool() for tool in self.tools]

        system_prompt = """You are a coding agent. You have access to the following tools:
- python_execute: Run Python code and get output
- read_file: Read the contents of a file
- write_file: Write content to a file
- edit_file: Edit a file by replacing specific text
- list_directory: List contents of a directory
- github: Interact with GitHub repositories (list repos, read/write files, create issues/PRs, search code)
- database: Execute SQL queries against a local SQLite database
- delegate_task: Delegate a sub-task to another agent (research, finance, critic, reviewer, etc.)

Given a coding task, decide which tools to use and in what order to accomplish the goal.
You may use multiple tools sequentially. After using tools, synthesize the results into a final answer."""

        content, tool_calls = await self._llm.generate(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": query}],
            tools=tools_schema,
            tool_choice="auto",
        )

        if tool_calls:
            results = []
            for tool_call in tool_calls:
                if tool_call["type"] != "function":
                    continue
                fn_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                tool = next((t for t in self.tools if t.name == fn_name), None)
                if not tool:
                    results.append(f"Tool not found: {fn_name}")
                    continue
                try:
                    result = await tool.execute(**args)
                    results.append(f"Tool {fn_name} returned:\n{result}")
                except Exception as e:
                    results.append(f"Tool {fn_name} failed: {e}")

            context = f"Task: {query}\n\nTool results:\n" + "\n\n".join(results)
            synthesis_prompt = """You are a coding agent. Based on the tool results above, provide a complete answer.
            Include the code you wrote, the output from execution, and any analysis of the results."""
            final_content, _ = await self._llm.generate(
                system_prompt=synthesis_prompt,
                messages=[{"role": "user", "content": context}],
            )
            return final_content
        else:
            return content
