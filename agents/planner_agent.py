"""
Planner Agent: goal decomposition, prioritization, ROI evaluation, strategy decisions.
"""
import json
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist der Planner Agent eines autonomen Mikro-Business-Systems.

Deine Aufgaben:
- Ziele in konkrete Tasks aufteilen
- Prioritäten nach ROI bewerten
- Strategien auswählen und abbrechen
- Ressourcen effizient einsetzen

Kontext-Ziel: Mindestens 20€ monatlicher Gewinn bei max. 20€/Monat Betriebskosten.

Zielgruppen: kleine Unternehmen, Vereine, Tierpflege, Hobby-Betriebe, kleine Creator.
Geschäftsmodelle: Automatisierungen, digitale Templates, OCR/PDF-Tools, einfache Bots.

Antworte immer als strukturiertes JSON:
{
  "analysis": "kurze Situationsanalyse",
  "priority_tasks": [
    {"task": "...", "expected_revenue_eur": 0.0, "cost_eur": 0.0, "roi_ratio": 0.0, "urgency": "high/medium/low"}
  ],
  "abort_strategies": ["Strategien die gestoppt werden sollen"],
  "reasoning": "Begründung"
}
"""


class PlannerAgent(BaseAgent):
    name = "planner"
    task_type = "planning"

    def plan(self, current_state: dict) -> dict:
        """Generate a prioritized action plan based on current state."""
        context = self._build_context()

        strategies = self.long_mem.get_profitable_strategies()
        strat_text = json.dumps(strategies[:5], indent=2) if strategies else "Keine bisherigen Strategien"

        revenue_stats = self.long_mem.stats()

        messages = [{
            "role": "user",
            "content": (
                f"Aktuelle Situation:\n{json.dumps(current_state, indent=2)}\n\n"
                f"Budget-Kontext:\n{context}\n\n"
                f"Profitable Strategien bisher:\n{strat_text}\n\n"
                f"Business Stats:\n{json.dumps(revenue_stats, indent=2)}\n\n"
                "Erstelle einen priorisierten Aktionsplan für den nächsten Zyklus. "
                "Fokus auf schnelle, kostengünstige Experimente mit positivem ROI. "
                "Antworte als JSON."
            ),
        }]

        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, max_tokens=2000, temperature=0.3)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            plan = json.loads(raw[start:end])
        except Exception:
            logger.warning("Planner: JSON parse failed, using raw output")
            plan = {"analysis": raw, "priority_tasks": [], "abort_strategies": [], "reasoning": ""}

        self.short_mem.set("latest_plan", plan, tags=["plan"])
        self.long_mem.store("planning", "latest_plan", plan, importance=0.8)
        return plan

    def evaluate_idea(self, idea: str, cost_eur: float, expected_revenue_eur: float) -> dict:
        """Quick ROI evaluation of a business idea."""
        roi = self.budget.evaluate_roi(cost_eur, expected_revenue_eur)
        messages = [{
            "role": "user",
            "content": (
                f"Bewerte diese Geschäftsidee:\n{idea}\n\n"
                f"Kosten: {cost_eur}€\nErwartete Einnahmen: {expected_revenue_eur}€\n"
                f"ROI-Berechnung: {json.dumps(roi)}\n\n"
                "Antworte kurz: PROCEED / MODIFY / ABORT und warum. "
                "Beachte: Ziel ist nachhaltige Profitabilität, nicht maximales Wachstum."
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, task_type="routing", max_tokens=300)
        return {"idea": idea, "roi_analysis": roi, "recommendation": raw.strip()}
