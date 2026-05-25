"""
Builder Agent: code generation, automation scripts, APIs, MVP creation.
"""
import json
import logging
import os
from pathlib import Path
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist der Builder Agent eines autonomen Mikro-Business-Systems.

Deine Aufgaben:
- Funktionierenden Python/Shell-Code schreiben
- Einfache APIs und Automationen bauen
- MVPs für digitale Produkte erstellen
- Templates und Tools generieren

Prinzipien:
- Minimaler, wartbarer Code
- Keine unnötigen Dependencies
- Produktionsreifer, sofort nutzbarer Output
- Fokus auf Geschwindigkeit über Perfektion

Antworte mit:
1. Kurzem Plan was gebaut wird
2. Vollständigem Code-Block (```python oder ```bash)
3. Anleitung zur Nutzung (1-3 Sätze)
"""


class BuilderAgent(BaseAgent):
    name = "builder"
    task_type = "coding"

    def build_product(self, spec: str, product_name: str) -> dict:
        """Generate code/content for a digital product."""
        context = self._build_context()

        # Check for similar past solutions
        past = self.long_mem.retrieve("products", product_name)

        messages = [{
            "role": "user",
            "content": (
                f"Erstelle dieses Produkt:\n{spec}\n\n"
                f"Produktname: {product_name}\n"
                f"Kontext: {context}\n\n"
                + (f"Ähnliche frühere Lösung:\n{json.dumps(past)}\n\n" if past else "")
                + "Schreibe vollständigen, lauffähigen Code. Kein Placeholder, kein TODO."
            ),
        }]

        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, max_tokens=4000, temperature=0.2)

        # Save to products directory
        product_dir = Path("data/products")
        product_dir.mkdir(parents=True, exist_ok=True)
        output_file = product_dir / f"{product_name.lower().replace(' ', '_')}.md"
        output_file.write_text(f"# {product_name}\n\n{raw}")

        result = {
            "product_name": product_name,
            "spec": spec,
            "output_file": str(output_file),
            "code": raw,
        }

        self.long_mem.store("products", product_name, {"spec": spec, "output_file": str(output_file)}, importance=0.7)
        self.long_mem.upsert_strategy(product_name, spec, status="built")
        return result

    def create_template(self, template_type: str, niche: str) -> str:
        """Create a sellable digital template."""
        messages = [{
            "role": "user",
            "content": (
                f"Erstelle ein professionelles, verkäufliches Template.\n"
                f"Template-Typ: {template_type}\n"
                f"Zielgruppe/Nische: {niche}\n\n"
                "Anforderungen:\n"
                "- Sofort nutzbar, professionell\n"
                "- Klare Struktur mit Platzhaltern\n"
                "- Deutsch, für kleine Unternehmen/Vereine geeignet\n"
                "- Praxisnah und wertvoll\n\n"
                "Liefere das vollständige Template."
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, task_type="writing", max_tokens=3000, temperature=0.4)

        # Save template
        templates_dir = Path("data/products/templates")
        templates_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{template_type}_{niche}".lower().replace(" ", "_")[:50]
        output_file = templates_dir / f"{fname}.md"
        output_file.write_text(f"# Template: {template_type} — {niche}\n\n{raw}")

        self.long_mem.store("templates", fname, {"type": template_type, "niche": niche, "file": str(output_file)}, importance=0.6)
        return str(output_file)

    def automate_task(self, task_description: str) -> str:
        """Generate an automation script for a described task."""
        messages = [{
            "role": "user",
            "content": (
                f"Schreibe ein Python-Automatisierungsskript für folgende Aufgabe:\n{task_description}\n\n"
                "Das Skript muss:\n"
                "- Sofort lauffähig sein\n"
                "- Fehlerbehandlung haben\n"
                "- Kurze Dokumentation haben\n"
                "- Keine unnötigen externen Libraries nutzen\n"
            ),
        }]
        raw, _, _ = self._call_llm(messages, system=SYSTEM_PROMPT, max_tokens=3000, temperature=0.2)

        scripts_dir = Path("data/products/scripts")
        scripts_dir.mkdir(parents=True, exist_ok=True)
        fname = task_description[:40].lower().replace(" ", "_").replace("/", "_")
        output_file = scripts_dir / f"{fname}.py"
        # extract code block if present
        code = raw
        if "```python" in raw:
            start = raw.find("```python") + 9
            end = raw.find("```", start)
            code = raw[start:end].strip()
        output_file.write_text(code)
        return str(output_file)
