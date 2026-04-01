from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parents[3] / "data" / "experiments.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER,
            config_json TEXT,
            expectancy REAL,
            win_rate REAL,
            profit_factor REAL,
            total_trades INTEGER,
            improved INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn


def log_experiment(
    iteration: int,
    config_dict: dict[str, Any],
    expectancy: float,
    win_rate: float,
    profit_factor: float,
    total_trades: int,
    improved: bool,
) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO experiments
           (iteration, config_json, expectancy, win_rate, profit_factor, total_trades, improved, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            iteration,
            json.dumps(config_dict),
            expectancy,
            win_rate,
            profit_factor,
            total_trades,
            int(improved),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_all_experiments() -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, iteration, config_json, expectancy, win_rate, profit_factor, total_trades, improved, timestamp FROM experiments ORDER BY id"
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "iteration": r[1],
            "config": json.loads(r[2]) if r[2] else {},
            "expectancy": r[3],
            "win_rate": r[4],
            "profit_factor": r[5],
            "total_trades": r[6],
            "improved": bool(r[7]),
            "timestamp": r[8],
        }
        for r in rows
    ]


def get_best_experiment() -> dict[str, Any] | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, iteration, config_json, expectancy, win_rate, profit_factor, total_trades, improved, timestamp FROM experiments ORDER BY expectancy DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "iteration": row[1],
        "config": json.loads(row[2]) if row[2] else {},
        "expectancy": row[3],
        "win_rate": row[4],
        "profit_factor": row[5],
        "total_trades": row[6],
        "improved": bool(row[7]),
        "timestamp": row[8],
    }
