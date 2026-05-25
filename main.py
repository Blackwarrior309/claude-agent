#!/usr/bin/env python3
"""
Autonomous MicroBusiness Agent — Entry Point

Usage:
  python main.py                        # Run single autonomous cycle
  python main.py --cycles 5             # Run N cycles (1h apart)
  python main.py --continuous           # Run indefinitely
  python main.py --status               # Show system status
  python main.py --opportunities        # Show market opportunities
  python main.py --build <product>      # Build a specific product
  python main.py --local-status         # Show Ollama/local model status
  python main.py --pull-models          # Download recommended Ollama models
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from core.budget_manager import BudgetManager
from core.model_router import ModelRouter
from core.tool_system import ToolRegistry, Tool
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.semantic_memory import SemanticMemory
from agents.planner_agent import PlannerAgent
from agents.builder_agent import BuilderAgent
from agents.analyst_agent import AnalystAgent
from agents.critic_agent import CriticAgent
from agents.sales_agent import SalesAgent
from agents.memory_agent import MemoryAgent
from core.autonomous_loop import AutonomousLoop
from business.market_analyzer import get_prioritized_niches
from business.revenue_tracker import revenue_summary

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/agent.log", mode="a"),
    ],
)
logger = logging.getLogger("main")


def load_settings() -> dict:
    config_path = Path("config/settings.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_local_router(settings: dict):
    """Initialize Ollama local model router if enabled and available."""
    local_cfg = settings.get("local_models", {})
    if not local_cfg.get("enabled", True):
        return None, lambda: False

    from core.local_model import OllamaClient, LocalModelRouter
    client = OllamaClient(base_url=local_cfg.get("ollama_url", "http://localhost:11434"))
    local_router = LocalModelRouter(client)

    if client.is_running():
        pulled = client.list_models()
        console.print(f"[green]Ollama running[/green] — {len(pulled)} model(s) loaded")
    else:
        console.print("[yellow]Ollama not running[/yellow] — local models disabled, using Claude API only")

    return local_router, client.is_running


def build_system(settings: dict):
    """Initialize all components and return the autonomous loop."""
    # Local model router (Ollama, optional)
    local_router, local_available_fn = build_local_router(settings)

    # Core infrastructure
    budget = BudgetManager(settings["budget"])
    router = ModelRouter(budget.spending_level, local_available_fn)
    short_mem = ShortTermMemory(capacity=settings["memory"]["short_term_capacity"])
    long_mem = LongTermMemory(settings["memory"]["long_term_db"])
    semantic_mem = SemanticMemory(settings["memory"]["embeddings_dir"])
    tools = ToolRegistry()

    # Register tools
    from tools.web_search import get_tool_definitions as web_tools
    from tools.file_tools import get_tool_definitions as file_tools

    for td in web_tools() + file_tools():
        tools.register(Tool(
            name=td["name"],
            description=td["description"],
            fn=td["fn"],
            schema=td["schema"],
        ))

    # Build agents (all receive local_router for free local inference)
    agent_kwargs = dict(
        budget=budget, router=router, short_mem=short_mem,
        long_mem=long_mem, tools=tools, settings=settings,
        local_router=local_router,
    )

    planner = PlannerAgent(**agent_kwargs)
    builder = BuilderAgent(**agent_kwargs)
    analyst = AnalystAgent(**agent_kwargs)
    critic = CriticAgent(**agent_kwargs)
    sales = SalesAgent(**agent_kwargs)
    memory_agent = MemoryAgent(**agent_kwargs, semantic_mem=semantic_mem)

    loop = AutonomousLoop(
        budget_manager=budget,
        planner=planner,
        builder=builder,
        analyst=analyst,
        critic=critic,
        sales=sales,
        memory_agent=memory_agent,
        long_mem=long_mem,
        short_mem=short_mem,
        settings=settings,
    )

    return loop, budget, long_mem, analyst, local_router


def print_status(budget: BudgetManager, long_mem: LongTermMemory):
    status = budget.status_report()
    rev = revenue_summary(long_mem)
    db_stats = long_mem.stats()

    table = Table(title="[bold cyan]System Status[/bold cyan]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Spending Level", f"[{'red' if status['spending_level'] != 'NORMAL' else 'green'}]{status['spending_level']}[/]")
    table.add_row("Monthly Spent", f"{status['monthly_spent_eur']:.3f}€ / {status['monthly_limit_eur']}€")
    table.add_row("Monthly Remaining", f"{status['monthly_remaining_eur']:.2f}€")
    table.add_row("Total Budget Left", f"{status['remaining_budget_eur']:.2f}€")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Monthly Revenue", f"{rev['monthly_eur']:.2f}€ / {rev['target_monthly_eur']}€ ({rev['completion_pct']}%)")
    table.add_row("Revenue Gap", f"{rev['gap_eur']:.2f}€")
    table.add_row("Total Revenue", f"{rev['total_eur']:.2f}€")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Active Strategies", str(db_stats["active_strategies"]))
    table.add_row("Recorded Failures", str(db_stats["recorded_failures"]))
    table.add_row("Reflection Cycles", str(db_stats["reflection_cycles"]))

    console.print(table)


def print_opportunities(analyst):
    console.print("\n[bold]Identifying market opportunities…[/bold]")
    opps = analyst.identify_opportunities()

    niches = get_prioritized_niches()
    console.print(Panel(
        "\n".join(f"[cyan]{n['name']}[/cyan] (score: {n.get('score', '?')}, ~{n['avg_price_eur']}€/sale)"
                  for n in niches[:5]),
        title="Pre-Validated Niches",
    ))

    if opps:
        table = Table(title="AI-Generated Opportunities")
        table.add_column("Name", style="bold")
        table.add_column("Market")
        table.add_column("Cost €")
        table.add_column("Revenue €/mo")
        table.add_column("Days to Sale")
        table.add_column("Confidence")

        for o in opps[:5]:
            table.add_row(
                o.get("name", ""),
                o.get("target_market", ""),
                str(o.get("implementation_cost_eur", "")),
                str(o.get("expected_monthly_revenue_eur", "")),
                str(o.get("time_to_first_sale_days", "")),
                o.get("confidence", ""),
            )
        console.print(table)


def print_local_status(local_router):
    if local_router is None:
        console.print("[yellow]Local models: disabled in config[/yellow]")
        return
    status = local_router.status()
    table = Table(title="[bold cyan]Local Model Status (Ollama)[/bold cyan]")
    table.add_column("Item", style="cyan")
    table.add_column("Value")

    table.add_row("Ollama running", "[green]YES[/green]" if status["ollama_running"] else "[red]NO[/red]")
    gpu = status.get("gpu", {})
    if gpu.get("gpu") and gpu["gpu"] != "unknown":
        vram_free = gpu.get("vram_free_mb", 0)
        vram_total = gpu.get("vram_total_mb", 0)
        table.add_row("GPU", gpu["gpu"])
        table.add_row("VRAM free", f"{vram_free} MB / {vram_total} MB")
    pulled = status.get("pulled_models", [])
    table.add_row("Pulled models", ", ".join(pulled) if pulled else "[dim]none[/dim]")
    table.add_row("Recommended", ", ".join(status.get("recommended_pulls", [])))
    console.print(table)

    if not pulled:
        console.print("\n[bold]To download the recommended models:[/bold]")
        console.print("  python main.py --pull-models")
        console.print("\nOr manually:")
        for m in status.get("recommended_pulls", []):
            console.print(f"  ollama pull {m}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous MicroBusiness Agent")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles to run")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    parser.add_argument("--opportunities", action="store_true", help="Show market opportunities")
    parser.add_argument("--build", type=str, help="Build a specific product (provide description)")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between cycles")
    parser.add_argument("--local-status", action="store_true", help="Show Ollama/local model status")
    parser.add_argument("--pull-models", action="store_true", help="Pull recommended Ollama models")
    parser.add_argument("--demo", action="store_true", help="Demo mode without API calls")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not args.demo and not args.status and not getattr(args, "local_status", False):
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set. Use --demo for demo mode.")
        sys.exit(1)

    settings = load_settings()
    Path("logs").mkdir(exist_ok=True)

    console.print(Panel(
        "[bold cyan]Autonomous MicroBusiness Agent v1.1[/bold cyan]\n"
        "Goal: 20€/month profit | Budget: 60€ total | Max 20€/month ops\n"
        "Local models: Ollama (phi3.5:mini, llama3.2:3b) for 0-cost tasks",
        expand=False,
    ))

    loop, budget, long_mem, analyst, local_router = build_system(settings)

    if args.status:
        print_status(budget, long_mem)
        print_local_status(local_router)
        return

    if getattr(args, "local_status", False):
        print_local_status(local_router)
        return

    if getattr(args, "pull_models", False):
        if local_router is None:
            console.print("[red]Local models disabled in config[/red]")
            return
        from core.local_model import LOCAL_MODELS
        for tier, cfg in LOCAL_MODELS.items():
            console.print(f"Pulling [cyan]{cfg.name}[/cyan] (~{cfg.vram_gb} GB VRAM)…")
            ok = local_router.client.pull_model(cfg.name)
            console.print("[green]OK[/green]" if ok else "[red]FAILED[/red]")
        return

    if args.opportunities:
        print_status(budget, long_mem)
        if not args.demo:
            print_opportunities(analyst)
        return

    if args.build:
        if args.demo:
            console.print(f"[Demo] Would build: {args.build}")
            return
        console.print(f"Building: [cyan]{args.build}[/cyan]")
        result = loop.builder.build_product(args.build, args.build[:30])
        console.print(f"[green]Built:[/green] {result['output_file']}")
        return

    # Run cycles
    if args.continuous:
        console.print(f"[bold]Running continuously (interval: {args.interval}s)…[/bold]")
        loop.run_continuous(interval_s=args.interval)
    else:
        cycles = args.cycles
        console.print(f"[bold]Running {cycles} cycle(s)…[/bold]")
        for i in range(cycles):
            if cycles > 1:
                console.print(f"\n[dim]--- Cycle {i+1}/{cycles} ---[/dim]")
            result = loop.run_cycle()
            eval_data = result["evaluation"]
            console.print(
                f"[green]Cycle done[/green] | "
                f"Revenue: {eval_data['monthly_revenue_eur']:.2f}€ | "
                f"Profit: {eval_data['monthly_profit_eur']:.2f}€ | "
                f"Tasks: {eval_data['tasks_run']}"
            )

    print_status(budget, long_mem)


if __name__ == "__main__":
    main()
