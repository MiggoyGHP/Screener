from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class TradeResult:
    ticker: str
    pattern: str
    scan_date: date
    entry_price: float
    exit_price: float
    return_pct: float
    hit_stop: bool
    hold_days: int
    score: float


@dataclass
class BacktestResult:
    total_setups: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    trades: list[TradeResult] = field(default_factory=list)

    @classmethod
    def from_trades(cls, trades: list[TradeResult]) -> BacktestResult:
        if not trades:
            return cls()

        returns = [t.return_pct for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        win_rate = len(wins) / len(returns) if returns else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        loss_rate = 1 - win_rate

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        gross_wins = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        return cls(
            total_setups=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=round(win_rate * 100, 1),
            avg_return_pct=round(sum(returns) / len(returns), 2),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            expectancy=round(expectancy, 2),
            profit_factor=round(profit_factor, 2),
            best_trade=round(max(returns), 2),
            worst_trade=round(min(returns), 2),
            trades=trades,
        )
