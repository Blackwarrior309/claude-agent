"""
Persistent SQLite-based long-term memory for strategies, errors, and business data.
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LongTermMemory:
    def __init__(self, db_path: str = "data/memory/longterm.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    importance REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_cat_key
                    ON knowledge(category, key);
                CREATE INDEX IF NOT EXISTS idx_knowledge_category
                    ON knowledge(category);

                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    cost_eur REAL DEFAULT 0.0,
                    impact TEXT,
                    solution TEXT,
                    prevention_rule TEXT,
                    occurred_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    status TEXT DEFAULT 'active',
                    revenue_eur REAL DEFAULT 0.0,
                    cost_eur REAL DEFAULT 0.0,
                    roi REAL DEFAULT 0.0,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS revenue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount_eur REAL NOT NULL,
                    source TEXT NOT NULL,
                    product TEXT,
                    customer TEXT,
                    notes TEXT,
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reflection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT,
                    goal TEXT,
                    what_worked TEXT,
                    what_failed TEXT,
                    why TEXT,
                    roi REAL,
                    token_cost_eur REAL,
                    reuse_strategy INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
            """)

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    # --- Knowledge ---

    def store(self, category: str, key: str, value: Any, tags: list[str] | None = None, importance: float = 0.5):
        serialized = json.dumps(value) if not isinstance(value, str) else value
        now = self._now()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO knowledge (category, key, value, tags, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(category, key) DO UPDATE SET
                    value = excluded.value,
                    tags = excluded.tags,
                    importance = excluded.importance,
                    updated_at = excluded.updated_at
            """, (category, key, serialized, json.dumps(tags or []), importance, now, now))

    def retrieve(self, category: str, key: str) -> Optional[Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM knowledge WHERE category=? AND key=?", (category, key)
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE knowledge SET access_count = access_count + 1 WHERE category=? AND key=?",
                (category, key),
            )
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def search(self, category: str, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value, tags, importance, updated_at FROM knowledge WHERE category=? ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        results = []
        for r in rows:
            try:
                val = json.loads(r["value"])
            except json.JSONDecodeError:
                val = r["value"]
            results.append({"key": r["key"], "value": val, "tags": json.loads(r["tags"]), "importance": r["importance"]})
        return results

    def search_by_tag(self, tag: str, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT category, key, value, tags FROM knowledge WHERE tags LIKE ? LIMIT ?",
                (f'%"{tag}"%', limit),
            ).fetchall()
        return [{"category": r["category"], "key": r["key"], "value": r["value"]} for r in rows]

    # --- Failures ---

    def record_failure(self, error_type: str, cause: str, cost_eur: float = 0.0,
                       impact: str = "", solution: str = "", prevention_rule: str = ""):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO failures (error_type, cause, cost_eur, impact, solution, prevention_rule, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (error_type, cause, cost_eur, impact, solution, prevention_rule, self._now()))
        logger.warning(f"Failure recorded: [{error_type}] {cause}")

    def get_recent_failures(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM failures ORDER BY occurred_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def failure_prevention_rules(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT prevention_rule FROM failures WHERE prevention_rule != '' ORDER BY occurred_at DESC"
            ).fetchall()
        return [r["prevention_rule"] for r in rows]

    # --- Strategies ---

    def upsert_strategy(self, name: str, description: str = "", status: str = "active"):
        now = self._now()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO strategies (name, description, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = COALESCE(excluded.description, description),
                    status = excluded.status,
                    updated_at = excluded.updated_at
            """, (name, description, status, now, now))

    def update_strategy_metrics(self, name: str, revenue: float = 0.0, cost: float = 0.0, success: bool = True):
        with self._connect() as conn:
            conn.execute("""
                UPDATE strategies SET
                    revenue_eur = revenue_eur + ?,
                    cost_eur = cost_eur + ?,
                    attempts = attempts + 1,
                    successes = successes + ?,
                    roi = CASE WHEN cost_eur + ? > 0
                               THEN (revenue_eur + ? - cost_eur - ?) / (cost_eur + ?) * 100
                               ELSE 0 END,
                    updated_at = ?
                WHERE name = ?
            """, (revenue, cost, 1 if success else 0, cost, revenue, cost, cost, self._now(), name))

    def get_profitable_strategies(self, min_roi: float = 0.0) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM strategies WHERE roi >= ? AND status = 'active' ORDER BY roi DESC",
                (min_roi,),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Revenue ---

    def record_revenue(self, amount_eur: float, source: str, product: str = "", customer: str = "", notes: str = ""):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO revenue (amount_eur, source, product, customer, notes, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (amount_eur, source, product, customer, notes, self._now()))
        logger.info(f"Revenue recorded: {amount_eur}€ from {source}")

    def total_revenue(self) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(SUM(amount_eur), 0) as total FROM revenue").fetchone()
        return row["total"]

    def monthly_revenue(self, month: str | None = None) -> float:
        month = month or datetime.utcnow().strftime("%Y-%m")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount_eur), 0) as total FROM revenue WHERE recorded_at LIKE ?",
                (f"{month}%",),
            ).fetchone()
        return row["total"]

    # --- Reflection ---

    def log_reflection(self, cycle_id: str, goal: str, what_worked: str, what_failed: str,
                       why: str, roi: float, token_cost_eur: float, reuse_strategy: bool):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO reflection_log
                    (cycle_id, goal, what_worked, what_failed, why, roi, token_cost_eur, reuse_strategy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cycle_id, goal, what_worked, what_failed, why, roi, token_cost_eur, 1 if reuse_strategy else 0, self._now()))

    def stats(self) -> dict:
        with self._connect() as conn:
            revenue = conn.execute("SELECT COALESCE(SUM(amount_eur), 0) as t FROM revenue").fetchone()["t"]
            strategies = conn.execute("SELECT COUNT(*) as c FROM strategies WHERE status='active'").fetchone()["c"]
            failures = conn.execute("SELECT COUNT(*) as c FROM failures").fetchone()["c"]
            reflections = conn.execute("SELECT COUNT(*) as c FROM reflection_log").fetchone()["c"]
        return {
            "total_revenue_eur": revenue,
            "active_strategies": strategies,
            "recorded_failures": failures,
            "reflection_cycles": reflections,
        }
