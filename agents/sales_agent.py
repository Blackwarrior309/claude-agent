"""
Sales Agent: outreach strategies, lead generation, offer optimization, conversion.
"""
import json
import logging
from pathlib import Path
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist der Sales Agent eines autonomen Mikro-Business-Systems.

Deine Aufgaben:
- Verkaufsstrategien entwickeln
- Outreach-Templates erstellen
- Angebote optimieren
- Zielgruppen analysieren
- Conversion-Texte schreiben

Prinzipien:
- Kein Spam, kein Hard-Selling
- Echten Mehrwert kommunizieren
- Kleine Zielgruppen, hohe Relevanz
- Ehrliche, transparente Kommunikation
- Günstige/kostenlose Kanäle bevorzugen

Zielgruppen: kleine Unternehmen, Vereine, Tierpflege, Hobby-Betriebe.
"""


class SalesAgent(BaseAgent):
    name = "sales"
    task_type = "writing"

    def create_outreach_template(self, product: str, target_market: str, channel: str = "email") -> str:
        """Create an outreach message template."""
        messages = [{
            "role": "user",
            "content": (
                f"Erstelle eine {channel}-Vorlage für dieses Produkt:\n"
                f"Produkt: {product}\n"
                f"Zielgruppe: {target_market}\n\n"
                "Anforderungen:\n"
                "- Persönlich und authentisch, kein Spam-Gefühl\n"
                "- Klar kommunizieren welches Problem gelöst wird\n"
                "- Call-to-Action ohne Druck\n"
                "- Max. 150 Wörter\n"
                "- Auf Deutsch\n"
                "- Platzhalter für Personalisierung: [NAME], [FIRMA]\n\n"
                "Schreibe NUR den Text, keine Erklärungen."
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, max_tokens=400, temperature=0.5)

        # Save template
        leads_dir = Path("data/leads")
        leads_dir.mkdir(parents=True, exist_ok=True)
        fname = f"outreach_{product[:30]}_{target_market[:20]}".lower().replace(" ", "_")
        output_file = leads_dir / f"{fname}.txt"
        output_file.write_text(f"Channel: {channel}\nProduct: {product}\nTarget: {target_market}\n\n---\n{raw}")

        self.long_mem.store("outreach", fname, {"product": product, "target": target_market, "template": raw}, importance=0.6)
        return raw

    def generate_offer(self, product_name: str, value_prop: str, price_eur: float) -> dict:
        """Generate a compelling sales offer."""
        messages = [{
            "role": "user",
            "content": (
                f"Erstelle ein überzeugendes Angebot:\n"
                f"Produkt: {product_name}\n"
                f"Nutzen: {value_prop}\n"
                f"Preis: {price_eur}€\n\n"
                "Liefere als JSON:\n"
                '{"headline": "...", "subheadline": "...", "benefits": ["..."], '
                '"price_justification": "...", "cta": "...", "urgency": "..."}'
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, max_tokens=600, temperature=0.5)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            offer = json.loads(raw[start:end])
        except Exception:
            offer = {"headline": product_name, "raw": raw}

        return offer

    def suggest_pricing(self, product: str, target_market: str, cost_eur: float) -> dict:
        """Suggest optimal pricing strategy."""
        messages = [{
            "role": "user",
            "content": (
                f"Empfehle eine Preisstrategie:\n"
                f"Produkt: {product}\n"
                f"Zielmarkt: {target_market}\n"
                f"Eigene Kosten: {cost_eur}€\n\n"
                "Berücksichtige: kleine Zielgruppe, niedriges Budget, erste Verkäufe wichtig.\n"
                "Antworte als JSON:\n"
                '{"recommended_price_eur": 0.0, "price_range": {"min": 0.0, "max": 0.0}, '
                '"strategy": "...", "reasoning": "..."}'
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, task_type="analysis", max_tokens=400)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            pricing = json.loads(raw[start:end])
        except Exception:
            pricing = {"recommended_price_eur": cost_eur * 3, "raw": raw}

        return pricing

    def identify_free_channels(self, niche: str) -> list[str]:
        """Identify free marketing channels for a niche."""
        messages = [{
            "role": "user",
            "content": (
                f"Welche kostenlosen Kanäle eignen sich für diese Nische: {niche}\n\n"
                "Ziel: Erste Kunden gewinnen ohne Budget.\n"
                "Antworte als JSON-Liste von Strings (max 8 Kanäle):\n"
                '["Kanal 1: Beschreibung", "Kanal 2: Beschreibung"]'
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, task_type="classification", max_tokens=400)

        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            channels = json.loads(raw[start:end])
        except Exception:
            channels = [raw]

        return channels
