# Autonomous MicroBusiness Agent

An autonomous multi-agent AI system that builds and operates a sustainable digital micro-business.

## Goal

Generate ≥ 20 €/month profit within 3 months, staying within a 20 €/month operating cost limit and a 60 € total budget.

## Architecture

```
main.py
├── core/
│   ├── autonomous_loop.py   # observe → plan → act → evaluate → learn
│   ├── budget_manager.py    # hard/soft limits, ROI evaluation
│   ├── model_router.py      # routes tasks to cheapest viable model
│   └── tool_system.py       # tool registry with schema + error handling
├── agents/
│   ├── planner_agent.py     # goal decomposition, prioritization
│   ├── builder_agent.py     # code, templates, MVPs
│   ├── analyst_agent.py     # KPIs, opportunities, market research
│   ├── critic_agent.py      # ethics/risk validation, code review
│   ├── sales_agent.py       # outreach, offers, pricing
│   └── memory_agent.py      # compression, insight extraction
├── memory/
│   ├── short_term.py        # in-process LRU cache
│   ├── long_term.py         # SQLite: strategies, failures, revenue
│   └── semantic_memory.py   # ChromaDB vector search (keyword fallback)
├── business/
│   ├── market_analyzer.py   # pre-validated niches + scoring
│   └── revenue_tracker.py   # projections, break-even
└── tools/
    ├── web_search.py        # DuckDuckGo + page fetcher
    └── file_tools.py        # safe file R/W + Python sandbox
```

## Budget Controls

| Threshold | Action |
|-----------|--------|
| 15 €/month | Soft warning — downgrade models, reduce background tasks |
| 19 €/month | Hard stop — cheap models only, essential tasks only |

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python main.py --status          # Show system status (no API calls)
python main.py --opportunities   # Find market opportunities
python main.py --cycles 1        # Run one autonomous cycle
python main.py --continuous      # Run indefinitely (1h intervals)
python main.py --build "..."     # Build a specific product
```

## Model Strategy

| Model | Use For |
|-------|---------|
| claude-opus-4-7 | Planning, coding, strategic decisions |
| claude-sonnet-4-6 | Analysis, writing, market research |
| claude-haiku-4-5 | Classification, parsing, routing |

## Focus Markets

- Small local businesses (Kleinunternehmen)
- Clubs & associations (Vereine)
- Pet care & animal husbandry (Tierpflege)
- Hobby businesses
- Small creators
- Excel/PDF process automation

## Priority Business Models

1. Digital templates (Vorlagen)
2. Simple automations
3. Micro-SaaS tools
4. OCR/PDF processing
5. Content automation
6. Data preparation scripts
