from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.config import Settings

from ..core.memory import Memory


class ChromaMemory(Memory):
    def __init__(self, persist_dir: str | None = None) -> None:
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR") or "./chroma_db"
        self._client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        ))
        self._collection = self._client.get_or_create_collection(
            name="agent_memory",
            metadata={"hnsw:space": "cosine"},
        )

    async def save(self, key: str, value: Any) -> None:
        import json
        text = json.dumps(value) if not isinstance(value, str) else value
        self._collection.add(
            ids=[key],
            documents=[text],
            metadatas=[{"key": key}],
        )

    async def load(self, key: str) -> Any | None:
        result = self._collection.get(ids=[key])
        if result and result["documents"]:
            return result["documents"][0]
        return None

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        result = self._collection.query(
            query_texts=[query],
            n_results=limit,
        )
        items = []
        if result["documents"]:
            for i, doc in enumerate(result["documents"][0]):
                items.append({
                    "key": result["ids"][0][i],
                    "value": doc,
                    "distance": result["distances"][0][i] if result.get("distances") else None,
                })
        return items

    async def delete(self, key: str) -> None:
        self._collection.delete(ids=[key])

    async def clear(self) -> None:
        self._client.delete_collection(name="agent_memory")
        self._collection = self._client.get_or_create_collection(name="agent_memory")
