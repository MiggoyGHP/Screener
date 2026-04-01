from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parents[3] / "data" / "labels.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            pattern_name TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            label INTEGER NOT NULL,
            features_json TEXT,
            chart_path TEXT,
            score REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_label(
    ticker: str,
    pattern_name: str,
    scan_date: str,
    label: int,
    features: dict[str, float] | None = None,
    chart_path: str | None = None,
    score: float = 0.0,
) -> int:
    """Save a label (1=good, 0=bad). Returns the label ID."""
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO labels (ticker, pattern_name, scan_date, label, features_json, chart_path, score, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker,
            pattern_name,
            scan_date,
            label,
            json.dumps(features) if features else None,
            chart_path,
            score,
            datetime.now().isoformat(),
        ),
    )
    label_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return label_id


def get_all_labels() -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, ticker, pattern_name, scan_date, label, features_json, chart_path, score, created_at FROM labels ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        features = json.loads(row[5]) if row[5] else {}
        result.append({
            "id": row[0],
            "ticker": row[1],
            "pattern_name": row[2],
            "scan_date": row[3],
            "label": row[4],
            "features": features,
            "chart_path": row[6],
            "score": row[7],
            "created_at": row[8],
        })
    return result


def get_label_counts() -> dict[str, int]:
    conn = _get_conn()
    good = conn.execute("SELECT COUNT(*) FROM labels WHERE label = 1").fetchone()[0]
    bad = conn.execute("SELECT COUNT(*) FROM labels WHERE label = 0").fetchone()[0]
    conn.close()
    return {"good": good, "bad": bad, "total": good + bad}


def flip_label(label_id: int) -> None:
    conn = _get_conn()
    conn.execute("UPDATE labels SET label = CASE WHEN label = 1 THEN 0 ELSE 1 END WHERE id = ?", (label_id,))
    conn.commit()
    conn.close()


def delete_label(label_id: int) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM labels WHERE id = ?", (label_id,))
    conn.commit()
    conn.close()
