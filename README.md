# Autonomous MicroBusiness Agent

An autonomous multi-agent AI system that builds and operates a sustainable digital micro-business.

## Goal

Generate ≥ 20 €/month profit within 3 months, staying within a 20 €/month operating cost limit and a 60 € total budget.

---

## Quickstart — Laptop mit GPU

### Schritt 1: Repository klonen & Dependencies installieren

```bash
git clone <repo-url>
cd claude-agent
pip install -r requirements.txt
```

### Schritt 2: Ollama + lokale Modelle einrichten (einmalig)

```bash
bash setup_ollama.sh
```

Das Skript installiert Ollama und lädt automatisch die richtigen Modelle für **4 GB VRAM**:
- `phi3.5:mini` (~2.2 GB) — Klassifikation, Routing, Parsing → **kostenlos**
- `llama3.2:3b` (~2.0 GB) — Zusammenfassungen, einfache Antworten → **kostenlos**
- `nomic-embed-text` (~0.3 GB) — Embeddings → **kostenlos**

### Schritt 3: API-Key setzen

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Schritt 4: Starten

```bash
# Status prüfen (kein API-Call)
python main.py --status

# Ollama / GPU Status
python main.py --local-status

# Ersten autonomen Zyklus starten
python main.py --cycles 1

# Marktchancen analysieren
python main.py --opportunities

# Dauerbetrieb (alle 1h ein Zyklus)
python main.py --continuous
```

---

## Model-Routing Strategie

```
Task eingehend
      │
      ▼
Ist Ollama aktiv + Modell vorhanden?
      │
   JA │ → lokales Modell (0 €)    ← phi3.5:mini, llama3.2:3b
      │
   NEIN
      │
      ▼
Budget SOFT_WARNING (>15 €/Monat)?
   JA → eine Stufe runter

Task-Typ → Modell:
  classification / parsing / routing → phi3.5:mini (lokal) → Haiku Fallback
  summarization / extraction         → llama3.2:3b (lokal) → Haiku Fallback
  analysis / writing / outreach      → Claude Sonnet
  planning / coding / strategy       → Claude Opus
```

### Warum keine 7B Modelle?

4 GB VRAM reichen nicht für llama3.1:8b (~4.7 GB) oder mistral:7b (~4.1 GB).  
Diese Tasks fallen auf Claude Haiku zurück (~0.001 €/Aufruf — sehr günstig).

### Lokale Token-Ersparnis

Typischer Zyklus ohne lokale Modelle: ~50 Haiku-Calls ≈ **0.06–0.10 €**  
Mit lokalen Modellen: ~30 lokale + 20 Haiku ≈ **0.02–0.04 €** → ~60% günstiger

---

## Architektur

```
main.py
├── core/
│   ├── autonomous_loop.py   # observe → plan → act → evaluate → learn
│   ├── budget_manager.py    # hard/soft limits, ROI evaluation
│   ├── model_router.py      # local → cheap → standard → premium
│   ├── local_model.py       # Ollama client, 4-GB-VRAM optimiert
│   └── tool_system.py       # tool registry
├── agents/
│   ├── base_agent.py        # transparent local-first routing
│   ├── planner_agent.py     # goal decomposition, prioritization
│   ├── builder_agent.py     # code, templates, MVPs
│   ├── analyst_agent.py     # KPIs, opportunities, market research
│   ├── critic_agent.py      # ethics/risk validation
│   ├── sales_agent.py       # outreach, offers, pricing
│   └── memory_agent.py      # compression, insight extraction
├── memory/
│   ├── short_term.py        # in-process LRU cache
│   ├── long_term.py         # SQLite: strategies, failures, revenue
│   └── semantic_memory.py   # ChromaDB + nomic-embed-text (lokal)
├── business/
│   ├── market_analyzer.py   # pre-validated niches + scoring
│   └── revenue_tracker.py   # projections, break-even
├── tools/
│   ├── web_search.py        # DuckDuckGo + page fetcher
│   └── file_tools.py        # safe file R/W + Python sandbox
└── config/settings.yaml     # alle Einstellungen inkl. Ollama-Config
```

## Budget Controls

| Schwelle | Aktion |
|----------|--------|
| 15 €/Monat | Soft warning — Modelle degradieren, mehr lokale Nutzung |
| 19 €/Monat | Hard stop — nur lokale Modelle + essenzielle Tasks |

## Fokus-Märkte

- Kleine lokale Unternehmen (Kleinunternehmer)
- Vereine & Clubs
- Tierpflege & Tierhaltung
- Hobby-Betriebe
- Kleine Creator
- Excel/PDF-Prozess-Automatisierung
