"""
Base agent with shared LLM call logic, budget tracking, and tool use.
"""
import logging
import os
from typing import Any, Optional

import anthropic

from core.budget_manager import BudgetManager
from core.model_router import ModelRouter, ModelConfig
from core.tool_system import ToolRegistry
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)


class BaseAgent:
    name: str = "base"
    task_type: str = "simple_qa"

    def __init__(
        self,
        budget: BudgetManager,
        router: ModelRouter,
        short_mem: ShortTermMemory,
        long_mem: LongTermMemory,
        tools: ToolRegistry,
        settings: dict,
    ):
        self.budget = budget
        self.router = router
        self.short_mem = short_mem
        self.long_mem = long_mem
        self.tools = tools
        self.settings = settings
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def _call_llm(
        self,
        messages: list[dict],
        system: str = "",
        task_type: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        tools: list[dict] | None = None,
    ) -> tuple[str, int, int]:
        task = task_type or self.task_type
        model_cfg = self.router.select(task)

        can, reason = self.budget.can_spend(model_cfg.estimate_cost(2000, max_tokens))
        if not can:
            logger.warning(f"[{self.name}] Budget check failed: {reason} — using cheap model")
            from core.model_router import MODELS
            model_cfg = MODELS["cheap"]

        kwargs: dict[str, Any] = {
            "model": model_cfg.id,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = model_cfg.estimate_cost(tokens_in, tokens_out)
        self.budget.record_spend(cost, f"{self.name}:{task}", model_cfg.id, tokens_in, tokens_out)

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_parts), tokens_in, tokens_out

    def _build_context(self, extra: str = "") -> str:
        rules = self.long_mem.failure_prevention_rules()
        rules_text = "\n".join(f"- {r}" for r in rules[:5]) if rules else "None"
        budget_status = self.budget.status_report()
        return (
            f"Budget status: {budget_status['spending_level']} | "
            f"Monthly spent: {budget_status['monthly_spent_eur']:.3f}€ / {budget_status['monthly_limit_eur']}€\n"
            f"Prevention rules from past failures:\n{rules_text}\n"
            + (f"\n{extra}" if extra else "")
        )
