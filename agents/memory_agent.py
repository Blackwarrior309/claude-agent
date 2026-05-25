"""
Memory Agent: knowledge storage, retrieval, compression, and learning management.
"""
import json
import logging
from agents.base_agent import BaseAgent
from memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


class MemoryAgent(BaseAgent):
    name = "memory"
    task_type = "summarization"

    def __init__(self, *args, semantic_mem: SemanticMemory, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic = semantic_mem

    def store_learning(self, event_type: str, content: str, metadata: dict | None = None):
        """Store a learning in semantic memory for future retrieval."""
        doc_id = f"{event_type}_{hash(content) % 10**8}"
        self.semantic.add(doc_id, content, metadata or {"type": event_type})
        logger.debug(f"Memory: stored learning [{event_type}]")

    def recall_similar(self, query: str, n: int = 5) -> list[dict]:
        """Retrieve semantically similar past experiences."""
        return self.semantic.search(query, n=n)

    def compress_context(self, messages: list[dict], target_tokens: int = 2000) -> str:
        """Summarize a long conversation/context into a compact form."""
        full_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if len(full_text) < target_tokens * 4:
            return full_text

        summary_messages = [{
            "role": "user",
            "content": (
                f"Komprimiere diese Konversation in max. {target_tokens // 4} Wörter.\n"
                "Behalte: wichtige Entscheidungen, Erkenntnisse, offene Punkte.\n"
                "Lass weg: Wiederholungen, Details, Smalltalk.\n\n"
                f"Text:\n{full_text[:8000]}"
            ),
        }]
        compressed, _, _ = self._call_llm(summary_messages, max_tokens=target_tokens, temperature=0.1)
        return compressed

    def extract_and_store_insights(self, cycle_result: dict) -> list[str]:
        """Extract key insights from a cycle result and store them."""
        messages = [{
            "role": "user",
            "content": (
                f"Extrahiere die wichtigsten Erkenntnisse aus diesem Zyklus-Ergebnis:\n"
                f"{json.dumps(cycle_result, indent=2)[:3000]}\n\n"
                "Fokus auf:\n"
                "- Was hat funktioniert? (Speichern als Erfolg)\n"
                "- Was hat nicht funktioniert? (Speichern als Fehler)\n"
                "- Welche Regel kann man daraus ableiten?\n\n"
                "Antworte als JSON:\n"
                '{"successes": ["..."], "failures": ["..."], "rules": ["..."]}'
            ),
        }]
        raw, _, _ = self._call_llm(messages, max_tokens=500, temperature=0.1)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            insights = json.loads(raw[start:end])
        except Exception:
            return []

        # Store in semantic memory
        for s in insights.get("successes", []):
            self.semantic.add(f"success_{hash(s)}", s, {"type": "success"})
        for f in insights.get("failures", []):
            self.semantic.add(f"failure_{hash(f)}", f, {"type": "failure"})
            self.long_mem.record_failure("cycle_failure", f, prevention_rule=f"Learned: {f}")

        rules = insights.get("rules", [])
        if rules:
            self.long_mem.store("learned_rules", "latest", rules, importance=0.9)

        return rules

    def get_relevant_context(self, query: str) -> str:
        """Build a relevant context string from memory for a given query."""
        similar = self.recall_similar(query, n=3)
        rules = self.long_mem.failure_prevention_rules()[:3]
        strategies = self.long_mem.get_profitable_strategies(min_roi=0)[:2]

        parts = []
        if similar:
            parts.append("Ähnliche Erfahrungen:\n" + "\n".join(f"- {r['text']}" for r in similar))
        if rules:
            parts.append("Regeln:\n" + "\n".join(f"- {r}" for r in rules))
        if strategies:
            parts.append("Erfolgreiche Strategien:\n" + "\n".join(f"- {s['name']}: ROI {s['roi']}%" for s in strategies))

        return "\n\n".join(parts) if parts else "Noch kein relevantes Kontextwissen vorhanden."
