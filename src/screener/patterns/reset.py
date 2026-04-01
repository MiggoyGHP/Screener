from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from screener.patterns.base import PatternDetector, PatternResult


class ResetDetector(PatternDetector):
    name = "Reset"

    def detect(
        self, ticker: str, df: pd.DataFrame, indicators: dict[str, Any]
    ) -> PatternResult | None:
        p = self.params

        min_uptrend = p.get("min_uptrend_days", 50)
        if len(df) < min_uptrend + 30:
            return None

        sma_50 = indicators["sma_50"]
        sma_200 = indicators["sma_200"]

        # Step 1: Confirm established uptrend
        if pd.isna(sma_50.iloc[-1]) or pd.isna(sma_200.iloc[-1]):
            return None
        if pd.isna(sma_50.iloc[-min_uptrend]) or pd.isna(sma_200.iloc[-min_uptrend]):
            return None

        sma_50_rising = float(sma_50.iloc[-1]) > float(sma_50.iloc[-min_uptrend])
        sma_200_rising = float(sma_200.iloc[-1]) > float(sma_200.iloc[-min_uptrend])
        current_close = float(df["Close"].iloc[-1])
        price_above = (
            current_close > float(sma_50.iloc[-1]) > float(sma_200.iloc[-1])
        )

        if not (sma_50_rising and sma_200_rising and price_above):
            return None

        # Step 2: Find pullback to MA
        ma_map = {
            "ema_10": indicators.get("ema_10"),
            "ema_21": indicators.get("ema_21"),
            "sma_50": indicators.get("sma_50"),
        }
        ma_targets = p.get("ma_targets", ["ema_10", "ema_21", "sma_50"])
        max_pb_days = p.get("max_pullback_days", 20)
        tolerance = p.get("touch_tolerance", 0.015)

        best_ma = None
        best_touch_day = None
        best_touch_low = None

        for ma_name in ma_targets:
            ma_series = ma_map.get(ma_name)
            if ma_series is None:
                continue

            for i in range(-max_pb_days, 0):
                if abs(i) >= len(df):
                    continue
                low = float(df["Low"].iloc[i])
                ma_val = float(ma_series.iloc[i])
                if pd.isna(ma_val) or ma_val <= 0:
                    continue
                # Check if low touched or penetrated the MA
                if low <= ma_val * (1 + tolerance) and low >= ma_val * (1 - 0.03):
                    if best_ma is None:
                        best_ma = ma_name
                        best_touch_day = i
                        best_touch_low = low
                    break

        if best_ma is None:
            return None

        # Step 3: Volume contraction during pullback
        vol_50_avg = indicators["volume_sma_50"]
        if pd.isna(vol_50_avg.iloc[-1]):
            return None
        vol_50 = float(vol_50_avg.iloc[-1])

        pullback_start = best_touch_day - 5
        pullback_end = best_touch_day + 1
        if abs(pullback_start) >= len(df):
            pullback_start = -len(df) + 1
        pullback_vol = df["Volume"].iloc[pullback_start:pullback_end].mean()
        vol_ratio = p.get("pullback_vol_ratio", 0.7)
        volume_contracted = float(pullback_vol) < vol_50 * vol_ratio

        # Step 4: Bounce confirmation
        post_touch = df.iloc[best_touch_day:]
        bouncing = float(post_touch["Close"].iloc[-1]) > float(post_touch["Close"].iloc[0])
        bounce_vol_ratio = p.get("bounce_vol_ratio", 1.2)
        bounce_volume = float(df["Volume"].iloc[-1]) > vol_50 * bounce_vol_ratio

        if not bouncing:
            return None

        # Step 5: Pullback depth
        # Find recent swing high before pullback
        pre_pullback = df.iloc[max(-60, pullback_start - 10) : pullback_start]
        if len(pre_pullback) == 0:
            return None
        swing_high = float(pre_pullback["High"].max())
        pullback_depth = (swing_high - best_touch_low) / swing_high * 100

        # Step 6: Score
        score = 0.0
        ma_scores = {"ema_10": 30, "ema_21": 25, "sma_50": 15}
        score += ma_scores.get(best_ma, 10)

        if volume_contracted:
            score += 20
        if bounce_volume:
            score += 20

        # RS trending up during pullback
        rs_line = indicators.get("rs_line")
        if rs_line is not None and len(rs_line) > 10:
            if float(rs_line.iloc[-1]) > float(rs_line.iloc[-10]):
                score += 15

        # Pullback shallowness
        if pullback_depth < 5:
            score += 15
        elif pullback_depth < 10:
            score += 10
        elif pullback_depth < 15:
            score += 5

        return PatternResult(
            pattern_name="Reset",
            ticker=ticker,
            score=round(score, 1),
            detected_date=df.index[-1].date(),
            pivot_price=round(swing_high, 2),
            metadata={
                "ma_touched": best_ma,
                "pullback_depth_pct": round(pullback_depth, 1),
                "volume_contracted": volume_contracted,
                "bounce_volume": bounce_volume,
                "bouncing": bouncing,
            },
            annotations=[
                {"type": "hline", "price": swing_high, "label": "Swing High"},
            ],
        )
