from __future__ import annotations

from typing import Any

from ..core.agent import BaseAgent
from ..core.state import ExecutionState
from ..llm.base import LLM
from ..tools import WebSearchTool


class ResearchAgent(BaseAgent):
    def __init__(self, llm: LLM, name: str = "research", description: str = "Performs web research using search tools") -> None:
        super().__init__(
            name=name,
            description=description,
            tools=[WebSearchTool()]
        )
        self._llm = llm

    async def run(self, task: Any, state: ExecutionState | None = None) -> str:
        """Research a topic using web search and LLM synthesis with tool calling."""
        # Extract the query from task description or use the task description directly
        query = getattr(task, 'description', str(task))
        
        # Build tools schema from self.tools
        tools_schema = [tool.to_llm_tool() for tool in self.tools]
        
        # First LLM call: decide whether to use a tool and which one
        system_prompt = """You are a research agent. You have access to a web search tool. 
        Given a query, decide if you need to search the web. If yes, return a tool call to web_search with the query.
        If you can answer without searching, return a text response."""
        content, tool_calls = await self._llm.generate(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": query}],
            tools=tools_schema,
            tool_choice="auto",  # Let the model decide
        )
        
        # If the LLM decided to use a tool
        if tool_calls:
            # We expect only one tool call for simplicity, but we can loop
            for tool_call in tool_calls:
                if tool_call["type"] == "function" and tool_call["function"]["name"] == "web_search":
                    # Parse the arguments
                    import json
                    args = json.loads(tool_call["function"]["arguments"])
                    search_query = args.get("query", query)
                    # Execute the tool
                    search_tool = next((t for t in self.tools if t.name == "web_search"), None)
                    if search_tool:
                        search_results = await search_tool.execute(query=search_query)
                    else:
                        search_results = [{"title": "Error", "snippet": "Web search tool not found"}]
                    # Now, call the LLM again to generate a final answer based on the search results
                    context = f"Research query: {query}\n\nSearch results:\n"
                    for i, result in enumerate(search_results, 1):
                        context += f"{i}. {result.get('title', 'No title')}: {result.get('snippet', 'No snippet')}\n"
                        if result.get('url'):
                            context += f"   URL: {result['url']}\n"
                    # We can use the same system prompt or a new one for synthesis
                    synthesis_prompt = """You are a research agent. You have performed a web search and obtained the following results.
                    Synthesize them into a coherent answer to the original query. Cite your sources."""
                    final_content, _ = await self._llm.generate(
                        system_prompt=synthesis_prompt,
                        messages=[{"role": "user", "content": context}],
                    )
                    return final_content
            # If we didn't find a web_search tool, fall back to the original content
            return content
        else:
            # The LLM returned a text response without using a tool
            return content