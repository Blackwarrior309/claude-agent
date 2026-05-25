"""
Routes tasks to the appropriate model based on complexity and budget state.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    id: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    use_for: list[str]

    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in / 1000 * self.cost_per_1k_input) + (tokens_out / 1000 * self.cost_per_1k_output)


MODELS = {
    "premium": ModelConfig(
        id="claude-opus-4-7",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        use_for=["complex_reasoning", "planning", "coding", "strategic_decisions", "difficult_problems"],
    ),
    "standard": ModelConfig(
        id="claude-sonnet-4-6",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        use_for=["analysis", "writing", "moderate_complexity", "market_research"],
    ),
    "cheap": ModelConfig(
        id="claude-haiku-4-5-20251001",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        use_for=["classification", "parsing", "summaries", "simple_answers", "routing", "filtering"],
    ),
}

TASK_MODEL_MAP = {
    "planning": "premium",
    "strategy": "premium",
    "coding": "premium",
    "architecture": "premium",
    "complex_analysis": "premium",
    "market_research": "standard",
    "writing": "standard",
    "content": "standard",
    "analysis": "standard",
    "outreach": "standard",
    "classification": "cheap",
    "summarization": "cheap",
    "parsing": "cheap",
    "routing": "cheap",
    "simple_qa": "cheap",
    "filtering": "cheap",
}


class ModelRouter:
    def __init__(self, spending_level_fn):
        self._spending_level = spending_level_fn

    def select(self, task_type: str, force_cheap: bool = False) -> ModelConfig:
        level = self._spending_level()

        if level == "HARD_STOP" or force_cheap:
            return MODELS["cheap"]

        if level == "SOFT_WARNING":
            # downgrade one tier
            tier = TASK_MODEL_MAP.get(task_type, "cheap")
            if tier == "premium":
                return MODELS["standard"]
            return MODELS["cheap"]

        tier = TASK_MODEL_MAP.get(task_type, "cheap")
        return MODELS[tier]

    def estimate_cost(self, task_type: str, tokens_in: int, tokens_out: int) -> float:
        model = self.select(task_type)
        return model.estimate_cost(tokens_in, tokens_out)
