"""
Budget management with hard/soft spending limits and ROI tracking.
"""
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpendingRecord:
    timestamp: str
    amount_eur: float
    task: str
    model: str
    tokens_in: int
    tokens_out: int
    category: str


class BudgetManager:
    def __init__(self, settings: dict):
        self.total_budget = settings["total_budget_eur"]
        self.monthly_limit = settings["monthly_limit_eur"]
        self.min_reserve = settings["min_reserve_eur"]
        self.soft_warning = settings["soft_warning_eur"]
        self.hard_stop = settings["hard_stop_eur"]

        self.spend_file = Path("data/memory/spending.json")
        self.spend_file.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[SpendingRecord] = self._load()

    def _load(self) -> list[SpendingRecord]:
        if self.spend_file.exists():
            data = json.loads(self.spend_file.read_text())
            return [SpendingRecord(**r) for r in data]
        return []

    def _save(self):
        self.spend_file.write_text(json.dumps([asdict(r) for r in self._records], indent=2))

    @property
    def current_month(self) -> str:
        return date.today().strftime("%Y-%m")

    def monthly_spend(self) -> float:
        month = self.current_month
        return sum(r.amount_eur for r in self._records if r.timestamp.startswith(month))

    def total_spend(self) -> float:
        return sum(r.amount_eur for r in self._records)

    def remaining_budget(self) -> float:
        return self.total_budget - self.total_spend()

    def remaining_monthly(self) -> float:
        return self.monthly_limit - self.monthly_spend()

    def spending_level(self) -> str:
        monthly = self.monthly_spend()
        if monthly >= self.hard_stop:
            return "HARD_STOP"
        if monthly >= self.soft_warning:
            return "SOFT_WARNING"
        return "NORMAL"

    def can_spend(self, amount: float) -> tuple[bool, str]:
        level = self.spending_level()
        if level == "HARD_STOP":
            return False, "Hard stop reached — monthly limit exceeded"
        if self.remaining_budget() - amount < self.min_reserve:
            return False, f"Would breach minimum reserve of {self.min_reserve}€"
        if self.remaining_monthly() < amount:
            return False, f"Would exceed monthly limit ({self.monthly_limit}€)"
        return True, "OK"

    def record_spend(
        self,
        amount_eur: float,
        task: str,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        category: str = "api",
    ):
        record = SpendingRecord(
            timestamp=datetime.utcnow().isoformat(),
            amount_eur=amount_eur,
            task=task,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            category=category,
        )
        self._records.append(record)
        self._save()
        logger.info(f"Spent {amount_eur:.4f}€ on '{task}' ({model}) | Monthly total: {self.monthly_spend():.3f}€")

    def status_report(self) -> dict:
        return {
            "total_budget_eur": self.total_budget,
            "total_spent_eur": round(self.total_spend(), 4),
            "remaining_budget_eur": round(self.remaining_budget(), 4),
            "monthly_limit_eur": self.monthly_limit,
            "monthly_spent_eur": round(self.monthly_spend(), 4),
            "monthly_remaining_eur": round(self.remaining_monthly(), 4),
            "spending_level": self.spending_level(),
            "min_reserve_eur": self.min_reserve,
        }

    def evaluate_roi(self, cost: float, expected_revenue: float) -> dict:
        if cost <= 0:
            return {"roi": float("inf"), "viable": True, "ratio": float("inf")}
        ratio = expected_revenue / cost
        return {
            "cost_eur": cost,
            "expected_revenue_eur": expected_revenue,
            "roi": round((expected_revenue - cost) / cost * 100, 1),
            "ratio": round(ratio, 2),
            "viable": ratio > 1.0,
        }
