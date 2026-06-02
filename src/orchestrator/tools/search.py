from __future__ import annotations

from typing import Any

from ..core.tool import Tool


class WebSearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="web_search",
            description="Search the web for current information. Use this for research, news, facts.",
        )

    async def execute(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        import httpx
        url = "https://duckduckgo-api.example.com/search"
        params = {"q": query, "max_results": num_results}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                return data.get("results", [])
        except Exception:
            return [{"title": "Search unavailable", "snippet": "Web search tool could not fetch results."}]
