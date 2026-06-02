from __future__ import annotations

import json
from typing import Any

import httpx

from ..core.tool import Tool


class WebSearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="web_search",
            description="Search the web for current information. Use this for research, news, facts.",
        )

    async def execute(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Search the web using DuckDuckGo instant answer API."""
        try:
            # Use DuckDuckGo's instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            
            results = []
            
            # Add abstract if available
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "Abstract"),
                    "snippet": data["Abstract"],
                    "url": data.get("AbstractURL", "")
                })
            
            # Add related topics
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                    results.append({
                        "title": topic.get("Text", "").split(" - ")[0][:100],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", "")
                    })
                    if len(results) >= num_results:
                        break
            
            # If we don't have enough results, add some from the definition
            if len(results) < num_results and data.get("Definition"):
                results.append({
                    "title": data.get("DefinitionSource", "Definition"),
                    "snippet": data["Definition"],
                    "url": data.get("DefinitionURL", "")
                })
            
            # If still no results, return a message
            if not results:
                results.append({
                    "title": "No results found",
                    "snippet": f"No search results found for query: {query}",
                    "url": ""
                })
            
            return results[:num_results]
            
        except Exception as e:
            # Fallback to a basic response if API fails
            return [{
                "title": "Search temporarily unavailable",
                "snippet": f"Unable to perform web search at this time: {str(e)}",
                "url": ""
            }]