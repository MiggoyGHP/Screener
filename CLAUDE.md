# Stock Pattern Screener

## Overview

A Python-based stock screening tool for swing/position traders. Detects **VCPs** (Minervini/Qullamaggi), **Reversals** (Stage 1→2), **Resets** (pullback-to-MA), and **Coils** (Darvas Boxes) across the US equity market. Includes a Relative Strength line (vs SPY) as both a filter and chart overlay.

## Tech Stack

- **Python 3.11+** with `pandas`, `numpy`, `scipy`
- **yfinance** for market data (free, no API key)
- **mplfinance** + `matplotlib` for candlestick charts
- **Streamlit** for the interactive dashboard
- **Pydantic** + YAML for typed configuration
- **SQLite** for local data caching

## Project Structure

```
src/screener/           # Core library
  config.py             # Pydantic config models (loads config/default_params.yaml)
  data/                 # Data fetching (yfinance), SQLite cache, universe management
  indicators/           # Moving averages, ATR, volume, relative strength
  patterns/             # Pattern detectors: VCP, Reversal, Reset, Coil + Trend Template
  scoring/              # Composite scoring & ranking
  pipeline/             # Main screening orchestrator (run_screen, scan_single)
  visualization/        # mplfinance chart generation with overlays
dashboard/              # Streamlit app
  app.py                # Main entry point
  pages/                # Screener Results, Pattern Explorer, Parameter Tuner
config/
  default_params.yaml   # All tunable detection parameters
```

## Running

```bash
# Install
pip install -e .

# Launch dashboard
cd dashboard && streamlit run app.py
```

## Key Commands

- `run_screen(tickers, config)` — Full pipeline: fetch data → compute indicators → detect patterns → score & rank
- `scan_single(ticker)` — Scan one ticker for all patterns
- All parameters tunable via `config/default_params.yaml` or the Parameter Tuner page

## Architecture Notes

- **Trend Template** (Minervini's 8 criteria) is a hard prerequisite for VCP only; other patterns use lighter uptrend checks
- **RS Ranking** is computed universe-wide (percentile rank); for single-ticker scans it defaults to 50th percentile
- Pattern detectors follow `PatternDetector` ABC in `patterns/base.py` — each returns a `PatternResult` with score, pivot, metadata, and chart annotations
- The `zigzag()` function in `base.py` is shared by VCP and Reversal for swing detection
- Data is cached in `data/cache.db` (SQLite) — auto-refreshes after 16 hours
- Universe filters: market cap > $500M, price > $15, 10-day avg volume > 1M

## Style Guidelines

- No pandas-ta dependency — all indicators computed manually with pandas/numpy
- Pattern parameters come from config dicts (`self.params`), not hardcoded
- Charts use `mplfinance` with the Yahoo style; RS line rendered in panel 2
- Dashboard pages import from `src/screener/` via sys.path manipulation
