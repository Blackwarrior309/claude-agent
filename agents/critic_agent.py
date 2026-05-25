"""
Critic Agent: error detection, quality control, security checks, strategy validation.
"""
import json
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist der Critic Agent eines autonomen Mikro-Business-Systems.

Deine Aufgaben:
- Fehler und Schwachstellen erkennen
- Qualität sicherstellen
- Sicherheit prüfen
- Strategien kritisch bewerten
- Verbesserungen vorschlagen

Sei ehrlich, direkt und konkret. Keine falschen Komplimente.
Bewerte immer aus der Sicht: "Was kann schiefgehen?"

Verbotene Aktionen des Systems:
- Illegale Aktivitäten
- Spamkampagnen
- Riskante Finanzspekulationen
- Manipulation von Menschen
- Missbrauch sensibler Daten
- Unbegrenzte Ausgaben
"""


class CriticAgent(BaseAgent):
    name = "critic"
    task_type = "analysis"

    def validate_plan(self, plan: dict) -> dict:
        """Validate a plan for risks, ethics, and feasibility."""
        messages = [{
            "role": "user",
            "content": (
                f"Prüfe diesen Plan kritisch:\n{json.dumps(plan, indent=2)}\n\n"
                "Bewerte:\n"
                "1. Ethische Risiken (0-10, 0=kein Risiko)\n"
                "2. Technische Machbarkeit (0-10, 10=sehr machbar)\n"
                "3. Budget-Risiko (0-10, 0=kein Risiko)\n"
                "4. Verstöße gegen verbotene Aktionen? JA/NEIN\n"
                "5. Top 3 Schwachstellen\n"
                "6. Empfehlung: APPROVE / MODIFY / REJECT\n\n"
                "Antworte als JSON:\n"
                '{"ethics_risk": 0, "feasibility": 0, "budget_risk": 0, '
                '"violates_rules": false, "weaknesses": [], "recommendation": "APPROVE", "notes": ""}'
            ),
        }]

        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, task_type="analysis", max_tokens=600)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            verdict = json.loads(raw[start:end])
        except Exception:
            verdict = {"recommendation": "MODIFY", "notes": raw, "ethics_risk": 5}

        if verdict.get("violates_rules") or verdict.get("ethics_risk", 0) > 7:
            logger.warning(f"Critic: REJECTED plan — ethics_risk={verdict.get('ethics_risk')}")
            self.long_mem.record_failure(
                "plan_rejected",
                f"High ethics risk: {verdict.get('notes', '')}",
                cost_eur=0,
                prevention_rule="Reject plans with ethics_risk > 7",
            )

        return verdict

    def review_code(self, code: str, context: str = "") -> dict:
        """Review generated code for bugs and security issues."""
        messages = [{
            "role": "user",
            "content": (
                f"Code Review:\n```\n{code[:3000]}\n```\n\n"
                + (f"Kontext: {context}\n\n" if context else "")
                + "Prüfe auf:\n"
                "- Sicherheitslücken (injection, etc.)\n"
                "- Offensichtliche Bugs\n"
                "- Fehlende Fehlerbehandlung\n"
                "- Ineffizienzen\n\n"
                "Antworte: PASS oder FAIL mit konkreten Issues."
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, task_type="classification", max_tokens=400)
        passed = "PASS" in raw.upper() and "FAIL" not in raw.upper()
        return {"passed": passed, "review": raw}

    def assess_strategy(self, strategy_name: str, metrics: dict) -> str:
        """Decide whether to continue, modify, or abort a strategy."""
        messages = [{
            "role": "user",
            "content": (
                f"Bewerte diese Strategie:\nName: {strategy_name}\n"
                f"Metriken: {json.dumps(metrics, indent=2)}\n\n"
                "Entscheide: CONTINUE / MODIFY / ABORT\n"
                "Begründung in 2 Sätzen."
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, task_type="classification", max_tokens=200)
        return raw.strip()
