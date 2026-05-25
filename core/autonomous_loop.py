"""
Autonomous loop: observe → plan → act → evaluate → store_memory → optimize.
Implements cycle management, budget guards, and loop-prevention.
"""
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AutonomousLoop:
    def __init__(
        self,
        budget_manager,
        planner,
        builder,
        analyst,
        critic,
        sales,
        memory_agent,
        long_mem,
        short_mem,
        settings: dict,
    ):
        self.budget = budget_manager
        self.planner = planner
        self.builder = builder
        self.analyst = analyst
        self.critic = critic
        self.sales = sales
        self.memory = memory_agent
        self.long_mem = long_mem
        self.short_mem = short_mem
        self.settings = settings

        self.cycle_count = 0
        self.max_consecutive_failures = 3
        self._consecutive_failures = 0
        self._running = False

        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    # ── Phases ───────────────────────────────────────────────────────────────

    def observe(self) -> dict:
        """Collect current system state."""
        budget_status = self.budget.status_report()
        monthly_revenue = self.long_mem.monthly_revenue()
        db_stats = self.long_mem.stats()
        recent_failures = self.long_mem.get_recent_failures(3)
        recent_memory = [e.key for e in self.short_mem.recent(5)]

        state = {
            "cycle": self.cycle_count,
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_status,
            "revenue": {
                "monthly_eur": monthly_revenue,
                "total_eur": db_stats["total_revenue_eur"],
                "target_monthly_eur": 20.0,
                "gap_eur": max(0, 20.0 - monthly_revenue),
            },
            "system": db_stats,
            "recent_failures": recent_failures,
            "active_context": recent_memory,
        }
        self.short_mem.set("current_state", state, tags=["state", "observe"])
        return state

    def plan(self, state: dict) -> Optional[dict]:
        """Generate action plan from planner, validate with critic."""
        if state["budget"]["spending_level"] == "HARD_STOP":
            logger.warning("Loop: HARD STOP — skipping plan, budget exhausted")
            return None

        plan = self.planner.plan(state)
        verdict = self.critic.validate_plan(plan)

        if verdict.get("violates_rules") or verdict.get("ethics_risk", 0) > 7:
            logger.warning(f"Loop: plan rejected by critic: {verdict.get('notes')}")
            self.long_mem.record_failure("rejected_plan", verdict.get("notes", ""), prevention_rule="Critic rejected plan")
            return None

        if verdict.get("recommendation") == "REJECT":
            logger.warning("Loop: plan hard-rejected")
            return None

        plan["critic_verdict"] = verdict
        self.short_mem.set("current_plan", plan, tags=["plan"])
        return plan

    def act(self, plan: dict, state: dict) -> list[dict]:
        """Execute top-priority tasks from plan."""
        results = []
        tasks = plan.get("priority_tasks", [])[:2]  # max 2 tasks per cycle

        for task_spec in tasks:
            task_name = task_spec.get("task", "")
            urgency = task_spec.get("urgency", "medium")
            expected_revenue = task_spec.get("expected_revenue_eur", 0.0)
            cost = task_spec.get("cost_eur", 0.5)

            roi = self.budget.evaluate_roi(cost, expected_revenue)
            if not roi["viable"] and urgency != "high":
                logger.info(f"Loop: skipping low-ROI task: {task_name}")
                continue

            can, reason = self.budget.can_spend(cost)
            if not can:
                logger.warning(f"Loop: cannot spend for task '{task_name}': {reason}")
                break

            result = self._execute_task(task_spec, state)
            results.append(result)

            if not result.get("success"):
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.max_consecutive_failures:
                    logger.error("Loop: too many consecutive failures, pausing")
                    break
            else:
                self._consecutive_failures = 0

        return results

    def _execute_task(self, task_spec: dict, state: dict) -> dict:
        """Route task to appropriate agent."""
        task_name = task_spec.get("task", "")
        task_lower = task_name.lower()

        try:
            if any(k in task_lower for k in ["markt", "analyse", "opportunit", "research"]):
                niche = task_spec.get("niche", task_name)
                data = self.analyst.identify_opportunities(niche)
                return {"task": task_name, "agent": "analyst", "success": True, "data": data}

            elif any(k in task_lower for k in ["template", "vorlage", "build", "erstell", "tool", "skript"]):
                product_name = task_spec.get("product_name", task_name)
                spec = task_spec.get("spec", task_name)
                data = self.builder.build_product(spec, product_name)
                return {"task": task_name, "agent": "builder", "success": True, "data": {"file": data["output_file"]}}

            elif any(k in task_lower for k in ["outreach", "sales", "verkauf", "kunden"]):
                product = task_spec.get("product", task_name)
                market = task_spec.get("target_market", "kleine Unternehmen")
                template = self.sales.create_outreach_template(product, market)
                return {"task": task_name, "agent": "sales", "success": True, "data": {"template_preview": template[:200]}}

            elif any(k in task_lower for k in ["preis", "angebot", "pricing"]):
                product = task_spec.get("product", task_name)
                market = task_spec.get("target_market", "")
                pricing = self.sales.suggest_pricing(product, market, task_spec.get("cost_eur", 1.0))
                return {"task": task_name, "agent": "sales", "success": True, "data": pricing}

            else:
                # Default: research / planning task
                opportunities = self.analyst.identify_opportunities(task_name)
                return {"task": task_name, "agent": "analyst", "success": True, "data": opportunities}

        except Exception as e:
            logger.error(f"Task execution failed: {task_name} — {e}")
            self.long_mem.record_failure("task_execution", str(e), cause=task_name)
            return {"task": task_name, "success": False, "error": str(e)}

    def evaluate(self, state: dict, plan: dict, results: list[dict]) -> dict:
        """Evaluate cycle results and compute metrics."""
        successes = sum(1 for r in results if r.get("success"))
        failures = len(results) - successes
        budget_after = self.budget.status_report()
        monthly_revenue = self.long_mem.monthly_revenue()

        evaluation = {
            "cycle": self.cycle_count,
            "tasks_run": len(results),
            "successes": successes,
            "failures": failures,
            "monthly_revenue_eur": monthly_revenue,
            "monthly_cost_eur": budget_after["monthly_spent_eur"],
            "monthly_profit_eur": monthly_revenue - budget_after["monthly_spent_eur"],
            "profitable": monthly_revenue > budget_after["monthly_spent_eur"],
            "target_gap_eur": max(0, 20.0 - monthly_revenue),
        }
        self.short_mem.set("last_evaluation", evaluation, tags=["evaluation"])
        return evaluation

    def store_memory(self, cycle_id: str, state: dict, plan: dict | None, results: list[dict], evaluation: dict):
        """Persist cycle learnings."""
        rules = self.memory.extract_and_store_insights({
            "state": state,
            "plan": plan,
            "results": results,
            "evaluation": evaluation,
        })

        self.long_mem.log_reflection(
            cycle_id=cycle_id,
            goal="20€/month profit",
            what_worked=str([r["task"] for r in results if r.get("success")]),
            what_failed=str([r["task"] for r in results if not r.get("success")]),
            why="See results",
            roi=evaluation.get("monthly_profit_eur", 0),
            token_cost_eur=evaluation.get("monthly_cost_eur", 0),
            reuse_strategy=evaluation.get("profitable", False),
        )

        cycle_log = {
            "cycle_id": cycle_id,
            "timestamp": datetime.utcnow().isoformat(),
            "evaluation": evaluation,
            "learned_rules": rules,
        }
        log_file = self.log_dir / f"cycle_{cycle_id}.json"
        log_file.write_text(json.dumps(cycle_log, indent=2))

    def optimize(self, evaluation: dict):
        """Prune failing strategies, reinforce profitable ones."""
        strategies = self.long_mem.get_profitable_strategies()
        for s in strategies:
            if s["roi"] < -50 and s["attempts"] > 3:
                self.long_mem.upsert_strategy(s["name"], status="aborted")
                logger.info(f"Loop: aborted unprofitable strategy: {s['name']}")

        # Update profitable ones
        if evaluation.get("profitable"):
            self.long_mem.store("optimization", "last_profitable_cycle", evaluation, importance=0.9)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run_cycle(self) -> dict:
        """Execute one full autonomous cycle."""
        cycle_id = str(uuid.uuid4())[:8]
        self.cycle_count += 1
        logger.info(f"=== Cycle {self.cycle_count} [{cycle_id}] START ===")

        state = self.observe()
        plan = self.plan(state)

        if plan is None:
            logger.warning("No valid plan — skipping act phase")
            results = []
        else:
            results = self.act(plan, state)

        evaluation = self.evaluate(state, plan or {}, results)
        self.store_memory(cycle_id, state, plan, results, evaluation)
        self.optimize(evaluation)

        logger.info(
            f"=== Cycle {self.cycle_count} [{cycle_id}] END | "
            f"Revenue: {evaluation['monthly_revenue_eur']:.2f}€ | "
            f"Profit: {evaluation['monthly_profit_eur']:.2f}€ ==="
        )
        return {"cycle_id": cycle_id, "evaluation": evaluation, "results": results}

    def run_continuous(self, max_cycles: int = 0, interval_s: int = 3600):
        """Run the autonomous loop continuously."""
        self._running = True
        cycle_num = 0

        while self._running:
            if max_cycles and cycle_num >= max_cycles:
                logger.info("Max cycles reached, stopping")
                break

            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Cycle crashed: {e}")
                self.long_mem.record_failure("cycle_crash", str(e), prevention_rule="Add error handling")
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.max_consecutive_failures:
                    logger.error("Too many crashes, stopping loop")
                    break

            cycle_num += 1
            if self._running and (max_cycles == 0 or cycle_num < max_cycles):
                logger.info(f"Sleeping {interval_s}s until next cycle…")
                time.sleep(interval_s)

        self._running = False

    def stop(self):
        self._running = False
