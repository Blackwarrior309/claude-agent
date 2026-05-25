"""
Base agent with transparent local-first LLM routing.

Call order: Ollama (free) → Claude Haiku → Sonnet → Opus
Falls back to next tier automatically on failure.
"""
import logging
import os
from typing import Any, Optional

import anthropic

from core.budget_manager import BudgetManager
from core.model_router import ModelRouter, ModelConfig, MODELS
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
        local_router=None,      # core.local_model.LocalModelRouter | None
    ):
        self.budget = budget
        self.router = router
        self.short_mem = short_mem
        self.long_mem = long_mem
        self.tools = tools
        self.settings = settings
        self.local = local_router  # None when Ollama not configured
        self._claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self._local_call_count = 0

    # ── LLM dispatch ──────────────────────────────────────────────────────────

    def _call_llm(
        self,
        messages: list[dict],
        system: str = "",
        task_type: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        tools: list[dict] | None = None,
    ) -> tuple[str, int, int]:
        """
        Route to local model or Claude depending on task, budget, and availability.
        Returns (text, tokens_in, tokens_out) — local calls report 0 tokens (no cost).
        """
        task = task_type or self.task_type
        model_cfg = self.router.select(task)

        # ── Local path ──────────────────────────────────────────────────────
        if model_cfg.is_local and self.local is not None:
            result = self.local.call(
                task_type=task,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if result is not None:
                self._local_call_count += 1
                logger.debug(f"[{self.name}] Local model handled '{task}' (0 cost)")
                return result, 0, 0
            # Local failed → fall through to Haiku
            logger.debug(f"[{self.name}] Local unavailable for '{task}', falling back to Haiku")
            model_cfg = MODELS["cheap"]

        # ── Claude API path ──────────────────────────────────────────────────
        estimated = model_cfg.estimate_cost(2000, max_tokens)
        can, reason = self.budget.can_spend(estimated)
        if not can:
            logger.warning(f"[{self.name}] Budget: {reason} — downgrading to Haiku")
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

        response = self._claude.messages.create(**kwargs)

        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = model_cfg.estimate_cost(tokens_in, tokens_out)
        self.budget.record_spend(cost, f"{self.name}:{task}", model_cfg.id, tokens_in, tokens_out)

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_parts), tokens_in, tokens_out

    # ── Context helpers ───────────────────────────────────────────────────────

    def _build_context(self, extra: str = "") -> str:
        rules = self.long_mem.failure_prevention_rules()
        rules_text = "\n".join(f"- {r}" for r in rules[:5]) if rules else "None"
        budget_status = self.budget.status_report()
        local_status = ""
        if self.local:
            local_status = f" | Local calls saved: {self._local_call_count}"
        return (
            f"Budget: {budget_status['spending_level']} | "
            f"Spent: {budget_status['monthly_spent_eur']:.3f}€/{budget_status['monthly_limit_eur']}€"
            f"{local_status}\n"
            f"Prevention rules:\n{rules_text}\n"
            + (f"\n{extra}" if extra else "")
        )

    def local_savings(self) -> dict:
        return self.router.local_savings_report(self._local_call_count)
