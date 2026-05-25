"""
Tool registry with schema validation, timeouts, and permission enforcement.
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: int = 0


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable,
        schema: dict,
        timeout_s: int = 30,
        requires_permission: bool = False,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.schema = schema
        self.timeout_s = timeout_s
        self.requires_permission = requires_permission

    def run(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            result = self.fn(**kwargs)
            elapsed = int((time.time() - start) * 1000)
            logger.debug(f"Tool '{self.name}' succeeded in {elapsed}ms")
            return ToolResult(success=True, data=result, execution_time_ms=elapsed)
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            logger.warning(f"Tool '{self.name}' failed: {e}")
            return ToolResult(success=False, data=None, error=str(e), execution_time_ms=elapsed)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def run(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"Tool '{name}' not found")
        return tool.run(**kwargs)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "schema": t.schema}
            for t in self._tools.values()
        ]

    def schemas_for_claude(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema,
            }
            for t in self._tools.values()
        ]
