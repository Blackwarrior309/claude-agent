"""
Revenue tracking and projection utilities.
"""
from datetime import datetime


def project_revenue(
    current_monthly_eur: float,
    growth_rate: float = 0.1,
    months: int = 3,
) -> list[dict]:
    projections = []
    revenue = current_monthly_eur
    for m in range(1, months + 1):
        revenue *= 1 + growth_rate
        projections.append({
            "month": m,
            "projected_revenue_eur": round(revenue, 2),
            "breaks_even": revenue >= 20.0,
        })
    return projections


def calculate_break_even(monthly_costs: float, price_per_sale: float) -> int:
    if price_per_sale <= 0:
        return -1
    return -(-int(monthly_costs) // int(price_per_sale))  # ceiling division


def revenue_summary(long_mem) -> dict:
    monthly = long_mem.monthly_revenue()
    total = long_mem.total_revenue()
    target = 20.0
    return {
        "monthly_eur": round(monthly, 2),
        "total_eur": round(total, 2),
        "target_monthly_eur": target,
        "completion_pct": round(monthly / target * 100, 1) if target else 0,
        "gap_eur": round(max(0, target - monthly), 2),
        "month": datetime.utcnow().strftime("%Y-%m"),
    }
