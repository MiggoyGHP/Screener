from __future__ import annotations

import copy
import random
from datetime import date
from typing import Any

import yaml

from screener.autoresearch.experiment_log import log_experiment, get_best_experiment
from screener.backtesting.rolling import rolling_backtest
from screener.config import ScreenerConfig, load_config


# Parameters that can be mutated and their valid ranges
MUTABLE_PARAMS = {
    "patterns.vcp.lookback_days": (60, 300, "int"),
    "patterns.vcp.min_contractions": (1, 5, "int"),
    "patterns.vcp.contraction_decay": (0.3, 0.9, "float"),
    "patterns.vcp.min_base_depth": (5.0, 25.0, "float"),
    "patterns.vcp.max_base_depth": (25.0, 70.0, "float"),
    "patterns.vcp.pivot_proximity": (0.01, 0.08, "float"),
    "patterns.reversal.min_downtrend_days": (30, 120, "int"),
    "patterns.reversal.decline_threshold": (-50.0, -10.0, "float"),
    "patterns.reversal.volume_expansion": (1.0, 3.0, "float"),
    "patterns.reversal.min_score": (20.0, 60.0, "float"),
    "patterns.reset.touch_tolerance": (0.005, 0.04, "float"),
    "patterns.reset.pullback_vol_ratio": (0.4, 1.0, "float"),
    "patterns.reset.bounce_vol_ratio": (0.8, 2.0, "float"),
    "patterns.reset.max_pullback_days": (10, 40, "int"),
    "patterns.coil.min_box_days": (5, 20, "int"),
    "patterns.coil.max_box_days": (20, 60, "int"),
    "patterns.coil.atr_contraction": (0.3, 0.9, "float"),
    "patterns.coil.box_range_max": (8.0, 25.0, "float"),
    "patterns.coil.volume_dry_pct": (0.4, 1.0, "float"),
    "trend_template.pct_above_52w_low": (10.0, 40.0, "float"),
    "trend_template.pct_within_52w_high": (10.0, 40.0, "float"),
    "trend_template.rs_rank_threshold": (40.0, 90.0, "float"),
}


def _get_param(config: ScreenerConfig, path: str) -> Any:
    """Get a nested config value by dot-separated path."""
    obj = config
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_param(config: ScreenerConfig, path: str, value: Any) -> None:
    """Set a nested config value by dot-separated path."""
    parts = path.split(".")
    obj = config
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def mutate_config(
    config: ScreenerConfig,
    n_mutations: int = 2,
    perturbation: float = 0.2,
) -> ScreenerConfig:
    """Create a mutated copy of the config by perturbing random parameters."""
    new_config = config.model_copy(deep=True)

    params_to_mutate = random.sample(
        list(MUTABLE_PARAMS.keys()),
        min(n_mutations, len(MUTABLE_PARAMS)),
    )

    for param_path in params_to_mutate:
        lo, hi, dtype = MUTABLE_PARAMS[param_path]
        current = _get_param(new_config, param_path)

        # Perturb by ±perturbation fraction
        delta = current * perturbation * random.choice([-1, 1]) * random.random()
        new_val = current + delta

        # Clamp to valid range
        new_val = max(lo, min(hi, new_val))

        if dtype == "int":
            new_val = int(round(new_val))

        _set_param(new_config, param_path, new_val)

    return new_config


def config_to_dict(config: ScreenerConfig) -> dict[str, Any]:
    """Extract only the mutable parameters as a flat dict for logging."""
    result = {}
    for path in MUTABLE_PARAMS:
        result[path] = _get_param(config, path)
    return result


def run_autoresearch(
    tickers: list[str],
    eval_start: date,
    eval_end: date,
    max_iterations: int = 50,
    scan_interval_days: int = 10,
    hold_days: int = 20,
    stop_loss_pct: float = 7.0,
    n_mutations: int = 2,
    perturbation: float = 0.2,
    progress_callback=None,
) -> dict[str, Any]:
    """Run the autoresearch optimization loop.

    Mutates parameters, backtests, keeps improvements.

    Returns dict with best_config, best_expectancy, iterations_run, improvements.
    """
    best_config = load_config()

    # Baseline
    baseline = rolling_backtest(
        tickers, eval_start, eval_end, best_config,
        scan_interval_days=scan_interval_days,
        hold_days=hold_days, stop_loss_pct=stop_loss_pct,
    )
    best_expectancy = baseline.expectancy

    log_experiment(
        iteration=0,
        config_dict=config_to_dict(best_config),
        expectancy=baseline.expectancy,
        win_rate=baseline.win_rate,
        profit_factor=baseline.profit_factor,
        total_trades=baseline.total_setups,
        improved=True,
    )

    improvements = 0

    for i in range(1, max_iterations + 1):
        candidate = mutate_config(best_config, n_mutations, perturbation)

        try:
            result = rolling_backtest(
                tickers, eval_start, eval_end, candidate,
                scan_interval_days=scan_interval_days,
                hold_days=hold_days, stop_loss_pct=stop_loss_pct,
            )
        except Exception:
            continue

        improved = result.expectancy > best_expectancy and result.total_setups >= 5

        log_experiment(
            iteration=i,
            config_dict=config_to_dict(candidate),
            expectancy=result.expectancy,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            total_trades=result.total_setups,
            improved=improved,
        )

        if improved:
            best_config = candidate
            best_expectancy = result.expectancy
            improvements += 1

            # Save the best config
            from pathlib import Path
            config_path = Path(__file__).resolve().parents[3] / "config" / "optimized_params.yaml"
            with open(config_path, "w") as f:
                yaml.dump(best_config.model_dump(), f, default_flow_style=False)

        if progress_callback:
            progress_callback(i, max_iterations, best_expectancy, improved)

    return {
        "best_config": config_to_dict(best_config),
        "best_expectancy": best_expectancy,
        "iterations_run": max_iterations,
        "improvements": improvements,
        "baseline_expectancy": baseline.expectancy,
    }
