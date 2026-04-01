from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DB_PATH = Path(__file__).resolve().parents[3] / "data" / "cache.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache_meta (
            ticker TEXT PRIMARY KEY,
            last_fetched TEXT,
            first_date TEXT,
            last_date TEXT
        )
    """)
    conn.commit()
    return conn


def is_cache_fresh(ticker: str, max_age_hours: int = 16) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT last_fetched FROM cache_meta WHERE ticker = ?", (ticker,)
    ).fetchone()
    conn.close()
    if row is None:
        return False
    last = datetime.fromisoformat(row[0])
    return (datetime.now() - last) < timedelta(hours=max_age_hours)


def load_from_cache(ticker: str) -> pd.DataFrame | None:
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM price_data WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
    )
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def save_to_cache(ticker: str, df: pd.DataFrame) -> None:
    conn = _get_conn()
    # Clear existing data for this ticker
    conn.execute("DELETE FROM price_data WHERE ticker = ?", (ticker,))
    records = []
    for date, row in df.iterrows():
        records.append((
            ticker,
            str(date.date()) if hasattr(date, "date") else str(date),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"]),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO price_data (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
        records,
    )
    dates = df.index
    conn.execute(
        "INSERT OR REPLACE INTO cache_meta (ticker, last_fetched, first_date, last_date) VALUES (?, ?, ?, ?)",
        (ticker, datetime.now().isoformat(), str(dates[0].date()), str(dates[-1].date())),
    )
    conn.commit()
    conn.close()


def get_cached_or_fetch(ticker: str, fetch_fn, period: str = "2y") -> pd.DataFrame | None:
    """Return cached data if fresh, otherwise fetch and cache."""
    if is_cache_fresh(ticker):
        cached = load_from_cache(ticker)
        if cached is not None and not cached.empty:
            return cached
    df = fetch_fn(ticker, period=period)
    if df is not None and not df.empty:
        save_to_cache(ticker, df)
        return df
    return None
