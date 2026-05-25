"""
Analyst Agent: KPI monitoring, cost/revenue analysis, market opportunity identification.
"""
import json
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist der Analyst Agent eines autonomen Mikro-Business-Systems.

Deine Aufgaben:
- Kosten und Einnahmen analysieren
- KPIs überwachen
- Marktchancen identifizieren
- Risiken bewerten
- Handlungsempfehlungen geben

Fokus auf: Zahlen, Fakten, klare Empfehlungen.
Antworte strukturiert und präzise.
"""


class AnalystAgent(BaseAgent):
    name = "analyst"
    task_type = "analysis"

    def analyze_performance(self) -> dict:
        """Full system performance analysis."""
        budget_status = self.budget.status_report()
        db_stats = self.long_mem.stats()
        monthly_revenue = self.long_mem.monthly_revenue()
        strategies = self.long_mem.get_profitable_strategies()

        performance = {
            "budget": budget_status,
            "revenue": {
                "monthly_eur": monthly_revenue,
                "total_eur": db_stats["total_revenue_eur"],
                "target_eur": 20.0,
                "gap_eur": max(0, 20.0 - monthly_revenue),
            },
            "profitability": {
                "monthly_profit_eur": monthly_revenue - budget_status["monthly_spent_eur"],
                "break_even": monthly_revenue >= budget_status["monthly_spent_eur"],
            },
            "strategies": {
                "active_count": db_stats["active_strategies"],
                "top_strategies": strategies[:3],
            },
        }

        messages = [{
            "role": "user",
            "content": (
                f"Analysiere die aktuelle Business-Performance:\n{json.dumps(performance, indent=2)}\n\n"
                "Gib eine klare Bewertung:\n"
                "1. Was läuft gut?\n"
                "2. Was muss verbessert werden?\n"
                "3. Konkrete Handlungsempfehlung (1 Aktion)\n"
                "4. Risiko-Level: LOW/MEDIUM/HIGH\n"
            ),
        }]

        analysis, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, task_type="analysis", max_tokens=800)

        result = {**performance, "ai_analysis": analysis}
        self.long_mem.store("analysis", "latest_performance", result, importance=0.7)
        return result

    def identify_opportunities(self, market_context: str = "") -> list[dict]:
        """Identify low-cost, high-ROI business opportunities."""
        prevention_rules = self.long_mem.failure_prevention_rules()
        past_failures = self.long_mem.get_recent_failures(5)

        messages = [{
            "role": "user",
            "content": (
                f"Identifiziere 3-5 konkrete Geschäftsmöglichkeiten für ein Mikro-Business.\n\n"
                f"Budget: max 5€ pro Experiment\n"
                f"Ziel: 20€/Monat Einnahmen\n"
                f"Zielgruppen: kleine Unternehmen, Vereine, Tierpflege, Hobby-Betriebe\n"
                f"Modelle: Templates, Automationen, kleine Tools, PDF/OCR-Services\n\n"
                + (f"Markt-Kontext:\n{market_context}\n\n" if market_context else "")
                + f"Gescheiterte Ansätze (vermeiden):\n{json.dumps(past_failures, indent=2)}\n\n"
                f"Prevention Rules:\n" + "\n".join(f"- {r}" for r in prevention_rules[:5]) + "\n\n"
                "Antworte als JSON-Array:\n"
                '[{"name": "...", "description": "...", "target_market": "...", '
                '"implementation_cost_eur": 0.0, "expected_monthly_revenue_eur": 0.0, '
                '"time_to_first_sale_days": 0, "confidence": "high/medium/low"}]'
            ),
        }]

        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, max_tokens=1500, temperature=0.4)

        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            opportunities = json.loads(raw[start:end])
        except Exception:
            logger.warning("Analyst: JSON parse failed for opportunities")
            opportunities = []

        self.long_mem.store("analysis", "opportunities", opportunities, importance=0.8)
        return opportunities

    def score_opportunity(self, opportunity: dict) -> dict:
        """Score and rank a single opportunity."""
        cost = opportunity.get("implementation_cost_eur", 1.0)
        revenue = opportunity.get("expected_monthly_revenue_eur", 0.0)
        roi = self.budget.evaluate_roi(cost, revenue)
        time_days = opportunity.get("time_to_first_sale_days", 30)

        score = (roi["ratio"] * 10) - (time_days / 30)
        return {**opportunity, **roi, "score": round(score, 2)}

    def market_research(self, niche: str) -> dict:
        """Lightweight market research for a niche."""
        messages = [{
            "role": "user",
            "content": (
                f"Führe eine kurze Marktanalyse für diese Nische durch: {niche}\n\n"
                "Analysiere:\n"
                "- Größe der Zielgruppe in Deutschland\n"
                "- Häufigste Probleme/Pain Points\n"
                "- Zahlungsbereitschaft (geschätzt)\n"
                "- Beste Kanäle zur Kundengewinnung\n"
                "- Top 3 Produktideen für diese Nische\n\n"
                "Antworte präzise und praxisnah. Kein Bullshit-Bingo."
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, max_tokens=800, temperature=0.3)
        result = {"niche": niche, "research": raw}
        self.long_mem.store("market_research", niche, result, importance=0.7)
        return result
