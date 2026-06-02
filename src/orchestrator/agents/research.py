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
        """Research a topic using web search and LLM synthesis."""
        # Extract the query from task description or use the task description directly
        query = getattr(task, 'description', str(task))
        
        # Use the web search tool to get information
        search_tool = self.tools[0] if self.tools else None
        search_results = []
        
        if search_tool:
            try:
                search_results = await search_tool.execute(query=query, num_results=5)
            except Exception as e:
                search_results = [{"title": "Search Error", "snippet": f"Failed to search: {str(e)}"}]
        
        # Prepare context for the LLM
        context = f"Research query: {query}\n\n"
        if search_results:
            context += "Search results:\n"
            for i, result in enumerate(search_results, 1):
                context += f"{i}. {result.get('title', 'No title')}: {result.get('snippet', 'No snippet')}\n"
                if result.get('url'):
                    context += f"   URL: {result['url']}\n"
                context += "\n"
        else:
            context += "No search results available.\n"
        
        # Generate a comprehensive answer using the LLM
        system_prompt = """You are a research agent. Your job is to provide accurate, well-researched information based on search results and your knowledge. 
        When given search results, synthesize them into a coherent answer. Always cite your sources when possible.
        If search results are limited or unavailable, use your knowledge but indicate when information might be incomplete."""
        
        content, _ = await self._llm.generate(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": context}],
        )
        
        return content