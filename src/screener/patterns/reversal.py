from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from screener.patterns.base import PatternDetector, PatternResult, zigzag


class ReversalDetector(PatternDetector):
    name = "Reversal"

    def detect(
        self, ticker: str, df: pd.DataFrame, indicators: dict[str, Any]
    ) -> PatternResult | None:
        p = self.params

        lookback = min(250, len(df))
        if lookback < 100:
            return None

        window = df.iloc[-lookback:]

        # Step 1: Identify prior downtrend
        peak_pos = int(window["High"].values.argmax())
        post_peak = window.iloc[peak_pos:]
        if len(post_peak) < p.get("min_downtrend_days", 60):
            return None

        trough_pos = int(post_peak["Low"].values.argmin())
        peak_price = float(window["High"].iloc[peak_pos])
        trough_price = float(post_peak["Low"].iloc[trough_pos])

        if peak_price <= 0:
            return None

        decline_pct = (trough_price - peak_price) / peak_price * 100
        if decline_pct > p.get("decline_threshold", -20):
            return None  # Not enough decline

        downtrend_days = trough_pos
        if downtrend_days < p.get("min_downtrend_days", 60):
            return None

        # Trough should not be the most recent bar (need some recovery)
        abs_trough_pos = peak_pos + trough_pos
        bars_since_trough = len(window) - 1 - abs_trough_pos
        if bars_since_trough < 5:
            return None

        # Step 2: Trendline break
        pivots = zigzag(
            post_peak["High"].iloc[:trough_pos].reset_index(drop=True),
            post_peak["Low"].iloc[:trough_pos].reset_index(drop=True),
            pct_threshold=3.0,
        )
        swing_highs = [(idx, price) for typ, idx, price in pivots if typ == "high"]

        trendline_broken = False
        if len(swing_highs) >= 2:
            x = np.array([sh[0] for sh in swing_highs])
            y = np.array([sh[1] for sh in swing_highs])
            slope, intercept, _, _, _ = stats.linregress(x, y)
            current_x = len(post_peak) - 1
            projected = slope * current_x + intercept
            trendline_broken = float(df["Close"].iloc[-1]) > projected

        # Step 3: MA reclaim
        ema_21 = indicators["ema_21"]
        sma_50 = indicators["sma_50"]
        reclaim_window = p.get("ma_reclaim_window", 10)

        reclaimed_21ema = False
        reclaimed_50sma = False

        for i in range(-min(reclaim_window, len(df) - 1), 0):
            if float(df["Close"].iloc[i]) > float(ema_21.iloc[i]):
                if float(df["Close"].iloc[i - 1]) <= float(ema_21.iloc[i - 1]):
                    reclaimed_21ema = True
                    break

        current_close = float(df["Close"].iloc[-1])
        if not pd.isna(sma_50.iloc[-1]):
            reclaimed_50sma = current_close > float(sma_50.iloc[-1])

        # Step 4: Volume confirmation
        vol_expansion = p.get("volume_expansion", 1.5)
        vol_50_avg = float(indicators["volume_sma_50"].iloc[-1]) if not pd.isna(indicators["volume_sma_50"].iloc[-1]) else 0
        high_vol_days = 0
        if vol_50_avg > 0:
            for i in range(-5, 0):
                if float(df["Volume"].iloc[i]) > vol_50_avg * vol_expansion:
                    high_vol_days += 1
        volume_confirmed = high_vol_days >= 1

        # Step 5: Weinstein stage transition
        sma_150 = indicators["sma_150"]
        stage_transition = False
        if len(sma_150.dropna()) >= 60:
            slope_recent = float(sma_150.iloc[-1]) - float(sma_150.iloc[-20])
            slope_prior = float(sma_150.iloc[-40]) - float(sma_150.iloc[-60])
            stage_transition = (
                slope_prior <= 0
                and slope_recent > 0
                and current_close > float(sma_150.iloc[-1])
            )

        # Step 6: Score
        score = 0.0
        if trendline_broken:
            score += 25
        if reclaimed_21ema:
            score += 15
        if reclaimed_50sma:
            score += 15
        if volume_confirmed:
            score += 20
        if stage_transition:
            score += 25

        min_score = p.get("min_score", 40)
        if score < min_score:
            return None

        return PatternResult(
            pattern_name="Reversal",
            ticker=ticker,
            score=round(score, 1),
            detected_date=df.index[-1].date(),
            pivot_price=round(float(df["High"].iloc[-5:].max()), 2),
            metadata={
                "decline_pct": round(decline_pct, 1),
                "downtrend_days": downtrend_days,
                "trendline_broken": trendline_broken,
                "reclaimed_21ema": reclaimed_21ema,
                "reclaimed_50sma": reclaimed_50sma,
                "volume_confirmed": volume_confirmed,
                "stage_transition": stage_transition,
                "high_vol_days": high_vol_days,
            },
            annotations=[],
        )
