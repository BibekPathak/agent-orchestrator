from __future__ import annotations

from typing import Any

from ..core.agent import BaseAgent
from ..core.state import ExecutionState
from ..llm.base import LLM
from ..tools import WebSearchTool


class FinanceAgent(BaseAgent):
    def __init__(self, llm: LLM, name: str = "finance", description: str = "Analyzes financial data and stock information") -> None:
        super().__init__(
            name=name,
            description=description,
            tools=[WebSearchTool()]
        )
        self._llm = llm

    async def run(self, task: Any, state: ExecutionState | None = None) -> str:
        """Analyze financial information using web search and LLM."""
        # Extract the query from task description
        query = getattr(task, 'description', str(task))
        
        # Enhance the query for financial context
        financial_query = f"financial stock data {query} latest price metrics"
        
        # Use the web search tool to get financial information
        search_tool = self.tools[0] if self.tools else None
        search_results = []
        
        if search_tool:
            try:
                search_results = await search_tool.execute(query=financial_query, num_results=5)
            except Exception as e:
                search_results = [{"title": "Search Error", "snippet": f"Failed to search financial data: {str(e)}"}]
        
        # Prepare context for the LLM
        context = f"Financial analysis request: {query}\n\n"
        if search_results:
            context += "Financial data from search:\n"
            for i, result in enumerate(search_results, 1):
                context += f"{i}. {result.get('title', 'No title')}: {result.get('snippet', 'No snippet')}\n"
                if result.get('url'):
                    context += f"   URL: {result['url']}\n"
                context += "\n"
        else:
            context += "No financial search results available.\n"
        
        # Generate a financial analysis using the LLM
        system_prompt = """You are a finance agent specializing in stock market analysis and financial data interpretation. 
        When given financial information or search results, provide clear analysis including:
        - Key financial metrics and ratios
        - Trends and patterns
        - Risk factors
        - Investment considerations
        Always base your analysis on the provided data and indicate when information is limited or speculative."""
        
        content, _ = await self._llm.generate(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": context}],
        )
        
        return content