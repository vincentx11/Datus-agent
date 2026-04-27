"""
Test cases for datus/tools/func_tool/base.py
Focuses on trans_to_function_tool parameter filtering for LLM-hallucinated arguments.
"""

import json

import pytest

from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, trans_to_function_tool


class TestTransToFunctionTool:
    """Tests for trans_to_function_tool and its parameter filtering logic."""

    def _make_tool_from_method(self, method):
        """Helper to create a FunctionTool from a bound method."""
        return trans_to_function_tool(method)

    @pytest.mark.asyncio
    async def test_filters_unexpected_parameters(self):
        """LLM-hallucinated parameters should be filtered out silently."""

        class FakeTool:
            def search_table(self, query_text: str, top_n: int = 5) -> FuncToolResult:
                return FuncToolResult(result={"query_text": query_text, "top_n": top_n})

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.search_table)

        # Simulate LLM sending an extra 'database_type' parameter
        args = json.dumps({"query_text": "test query", "database_type": "sqlite"})
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 1
        assert result["result"]["query_text"] == "test query"
        assert result["result"]["top_n"] == 5

    @pytest.mark.asyncio
    async def test_valid_parameters_pass_through(self):
        """All valid parameters should be passed through correctly."""

        class FakeTool:
            def search_table(self, query_text: str, top_n: int = 5) -> FuncToolResult:
                return FuncToolResult(result={"query_text": query_text, "top_n": top_n})

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.search_table)

        args = json.dumps({"query_text": "hello", "top_n": 10})
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 1
        assert result["result"]["query_text"] == "hello"
        assert result["result"]["top_n"] == 10

    @pytest.mark.asyncio
    async def test_empty_args(self):
        """Empty arguments should work without errors."""

        class FakeTool:
            def no_args_method(self) -> FuncToolResult:
                return FuncToolResult(result="ok")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.no_args_method)

        result = await tool.on_invoke_tool(None, "")
        assert result["success"] == 1
        assert result["result"] == "ok"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self):
        """Invalid JSON should return an error result."""

        class FakeTool:
            def some_method(self, x: str) -> FuncToolResult:
                return FuncToolResult(result=x)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.some_method)

        result = await tool.on_invoke_tool(None, "not-valid-json{")
        assert result["success"] == 0
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_multiple_extra_params_all_filtered(self):
        """Multiple hallucinated parameters should all be filtered out."""

        class FakeTool:
            def simple(self, name: str) -> FuncToolResult:
                return FuncToolResult(result=name)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.simple)

        args = json.dumps({"name": "test", "fake1": 1, "fake2": "x", "fake3": True})
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 1
        assert result["result"] == "test"


class TestFuncToolListResult:
    """Tests for the canonical list-shaped envelope."""

    def test_defaults_empty_items_and_none_pagination(self):
        env = FuncToolListResult()
        assert env.items == []
        assert env.total is None
        assert env.has_more is None
        assert env.extra is None

    def test_serialization_round_trips_through_funcresult(self):
        env = FuncToolListResult(
            items=[{"id": "1", "name": "foo"}, {"id": "2", "name": "bar"}],
            total=137,
            has_more=True,
            extra={"next_offset": 20},
        )
        outer = FuncToolResult(result=env.model_dump())
        dumped = outer.model_dump(mode="json")

        assert dumped["success"] == 1
        assert dumped["error"] is None
        assert dumped["result"] == {
            "items": [{"id": "1", "name": "foo"}, {"id": "2", "name": "bar"}],
            "total": 137,
            "has_more": True,
            "extra": {"next_offset": 20},
        }

    def test_items_stay_a_list_when_none_passed(self):
        # Pydantic rejects items=None (default_factory returns []), so the
        # "always a list" invariant is enforced at construction time.
        with pytest.raises(ValueError):
            FuncToolListResult(items=None)

    def test_extra_accepts_arbitrary_tool_metadata(self):
        env = FuncToolListResult(
            items=[{"k": "v"}],
            extra={"next_offset": 5, "cursor": "abc", "filters_applied": ["x"]},
        )
        assert env.extra["cursor"] == "abc"
        assert env.extra["filters_applied"] == ["x"]

    def test_empty_items_is_empty_list_not_missing(self):
        env = FuncToolListResult(items=[], total=0, has_more=False)
        dumped = env.model_dump()
        assert dumped["items"] == []
        assert dumped["total"] == 0
        assert dumped["has_more"] is False

    def test_required_excludes_params_with_defaults(self):
        """Parameters with default values should not be in the 'required' list."""

        class FakeTool:
            def method(
                self,
                req: str,
                opt: str | None = None,
                with_default: int = 100,
                opt_str: str | None = None,
            ) -> FuncToolResult:
                return FuncToolResult(result=req)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.method)

        assert tool.params_json_schema.get("required") == ["req"]

    def test_required_all_params_required(self):
        """All params without defaults should be in 'required'."""

        class FakeTool:
            def method(self, a: str, b: int, c: float) -> FuncToolResult:
                return FuncToolResult(result=a)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.method)

        assert set(tool.params_json_schema.get("required", [])) == {"a", "b", "c"}

    def test_required_no_params(self):
        """Methods with no parameters should have empty 'required' list."""

        class FakeTool:
            def method(self) -> FuncToolResult:
                return FuncToolResult(result="ok")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.method)

        assert tool.params_json_schema.get("required", []) == []

    def test_required_all_optional(self):
        """All optional params should result in empty 'required' list."""

        class FakeTool:
            def method(self, a: str | None = None, b: int = 0) -> FuncToolResult:
                return FuncToolResult(result="ok")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.method)

        assert tool.params_json_schema.get("required", []) == []

    def test_required_self_not_in_list(self):
        """The 'self' parameter should never appear in 'required' or 'properties'."""

        class FakeTool:
            def method(self, name: str, opt: str | None = None) -> FuncToolResult:
                return FuncToolResult(result=name)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.method)

        assert "self" not in tool.params_json_schema.get("properties", {})
        assert "self" not in tool.params_json_schema.get("required", [])
        assert tool.params_json_schema.get("required") == ["name"]

    def test_optional_type_simplifies_anyof_to_plain_type(self):
        """Optional[str] should produce {\"type\": \"string\"} not anyOf[str, null]."""

        class FakeTool:
            def method(
                self,
                opt_str: str | None = None,
                opt_int: int | None = None,
                opt_list: list[str] | None = None,
            ) -> FuncToolResult:
                return FuncToolResult()

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.method)

        props = tool.params_json_schema["properties"]
        # All three should have a plain "type" key, not "anyOf"
        assert "anyOf" not in props["opt_str"]
        assert props["opt_str"]["type"] == "string"
        assert "anyOf" not in props["opt_int"]
        assert props["opt_int"]["type"] == "integer"
        assert "anyOf" not in props["opt_list"]
        assert props["opt_list"]["type"] == "array"

    def test_required_param_keeps_anyof_untouched(self):
        """Required parameters should not have their schema modified."""

        class FakeTool:
            def method(self, req_opt: str | None) -> FuncToolResult:
                return FuncToolResult()

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.method)

        assert tool.params_json_schema.get("required") == ["req_opt"]
        # Required Optional params keep the anyOf since they are in required
        assert "anyOf" in tool.params_json_schema["properties"]["req_opt"]
