import pytest
from src.orchestrator.agents.planner import _extract_json


class TestExtractJson:
    def test_extract_array(self) -> None:
        text = "Here is the plan:\n[{\"id\": \"t1\", \"description\": \"test\"}]\nDone."
        result = _extract_json(text)
        assert result == "[{\"id\": \"t1\", \"description\": \"test\"}]"

    def test_extract_object(self) -> None:
        text = "Result: {\"tasks\": [{\"id\": \"t1\"}]}"
        result = _extract_json(text)
        assert result == "{\"tasks\": [{\"id\": \"t1\"}]}"

    def test_no_json_found(self) -> None:
        text = "No JSON here"
        result = _extract_json(text)
        assert result == text
