"""
Routes tasks to the cheapest viable model.

Priority (lowest cost first):
  local  → Ollama (phi3.5:mini, llama3.2:3b) — 0 € — classification, parsing, summaries
  cheap  → Claude Haiku                        — ~0.001 €/call — routing, simple QA
  standard → Claude Sonnet                     — ~0.01 €/call  — analysis, writing
  premium  → Claude Opus                       — ~0.05 €/call  — planning, coding, strategy

4-GB-VRAM Hinweis: 7B/8B Modelle passen nicht — local deckt nur nano/small Tasks.
Haiku übernimmt alle anderen günstigen Tasks.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    id: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    use_for: list[str]
    is_local: bool = False

    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in / 1000 * self.cost_per_1k_input) + (tokens_out / 1000 * self.cost_per_1k_output)


MODELS = {
    "local": ModelConfig(
        id="local",  # resolved to actual Ollama model at call time
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        use_for=["classification", "filtering", "routing", "parsing", "simple_qa",
                 "summarization", "extraction", "embeddings"],
        is_local=True,
    ),
    "cheap": ModelConfig(
        id="claude-haiku-4-5-20251001",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        use_for=["classification", "parsing", "summaries", "simple_answers", "routing", "filtering"],
    ),
    "standard": ModelConfig(
        id="claude-sonnet-4-6",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        use_for=["analysis", "writing", "moderate_complexity", "market_research"],
    ),
    "premium": ModelConfig(
        id="claude-opus-4-7",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        use_for=["complex_reasoning", "planning", "coding", "strategic_decisions", "difficult_problems"],
    ),
}

# Tasks that can be handled locally (4 GB GPU — only nano/small models)
LOCAL_CAPABLE_TASKS = {
    "classification", "filtering", "routing", "parsing",
    "simple_qa", "summarization", "extraction",
}

TASK_MODEL_MAP = {
    # Local-first (0 cost if Ollama running)
    "classification": "local",
    "filtering":      "local",
    "routing":        "local",
    "parsing":        "local",
    "simple_qa":      "local",
    "summarization":  "local",
    "extraction":     "local",
    # API-only (local models too weak on 4 GB)
    "analysis":       "standard",
    "writing":        "standard",
    "content":        "standard",
    "market_research":"standard",
    "outreach":       "standard",
    "complex_analysis":"premium",
    "planning":       "premium",
    "strategy":       "premium",
    "coding":         "premium",
    "architecture":   "premium",
}


class ModelRouter:
    def __init__(self, spending_level_fn, local_available_fn=None):
        self._spending_level = spending_level_fn
        # Callable that returns True if Ollama is running
        self._local_available = local_available_fn or (lambda: False)

    def select(self, task_type: str, force_cheap: bool = False) -> ModelConfig:
        level = self._spending_level()

        # Hard stop: local only, then cheapest API fallback
        if level == "HARD_STOP" or force_cheap:
            if task_type in LOCAL_CAPABLE_TASKS and self._local_available():
                return MODELS["local"]
            return MODELS["cheap"]

        tier = TASK_MODEL_MAP.get(task_type, "local")

        # Try local first for local-tier tasks
        if tier == "local":
            if self._local_available():
                return MODELS["local"]
            return MODELS["cheap"]  # Haiku fallback when Ollama down

        # Soft warning: downgrade one tier
        if level == "SOFT_WARNING":
            if tier == "premium":
                return MODELS["standard"]
            if tier == "standard":
                # Try local if capable, else cheap
                if task_type in LOCAL_CAPABLE_TASKS and self._local_available():
                    return MODELS["local"]
                return MODELS["cheap"]

        return MODELS[tier]

    def estimate_cost(self, task_type: str, tokens_in: int, tokens_out: int) -> float:
        model = self.select(task_type)
        return model.estimate_cost(tokens_in, tokens_out)

    def local_savings_report(self, local_call_count: int, avg_tokens: int = 500) -> dict:
        """Estimate how much was saved by using local models."""
        haiku_cost = MODELS["cheap"].estimate_cost(avg_tokens, avg_tokens)
        saved = haiku_cost * local_call_count
        return {
            "local_calls": local_call_count,
            "estimated_saved_eur": round(saved, 4),
            "equivalent_haiku_cost_eur": round(haiku_cost, 5),
        }
