from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from screener.patterns.base import PatternDetector, PatternResult


class CoilDetector(PatternDetector):
    name = "Coil"

    def detect(
        self, ticker: str, df: pd.DataFrame, indicators: dict[str, Any]
    ) -> PatternResult | None:
        p = self.params

        min_box = p.get("min_box_days", 10)
        max_box = p.get("max_box_days", 45)
        range_min = p.get("box_range_min", 2)
        range_max = p.get("box_range_max", 15)

        if len(df) < max_box + 60:
            return None

        # Step 1: Find a Darvas-style box
        box_found = False
        box_high = 0.0
        box_low = 0.0
        box_range_pct = 0.0
        box_days = 0

        for window_size in range(min_box, max_box + 1):
            window = df.iloc[-window_size:]
            bh = float(window["High"].max())
            bl = float(window["Low"].min())
            if bl <= 0:
                continue
            rng = (bh - bl) / bl * 100

            if rng < range_min or rng > range_max:
                continue

            # Check containment: all bars stay within the box (small tolerance)
            tol = 0.005
            contained = (
                (window["High"] <= bh * (1 + tol)).all()
                and (window["Low"] >= bl * (1 - tol)).all()
            )

            if contained:
                box_found = True
                box_high = bh
                box_low = bl
                box_range_pct = rng
                box_days = window_size
                # Keep searching for a larger box (longer consolidation is stronger)
                continue

        if not box_found:
            return None

        # Step 2: ATR contraction
        atr_14 = indicators.get("atr_14")
        if atr_14 is None or len(atr_14.dropna()) < 60:
            return None

        atr_current = float(atr_14.iloc[-5:].mean())
        atr_prior = float(atr_14.iloc[-60:-20].mean())
        if atr_prior <= 0:
            return None

        atr_ratio = atr_current / atr_prior
        coiling = atr_ratio <= p.get("atr_contraction", 0.6)

        # Step 3: Volume dry-up in box
        vol_in_box = float(df["Volume"].iloc[-box_days:].mean())
        pre_box_start = -box_days * 3
        pre_box_end = -box_days
        if abs(pre_box_start) > len(df):
            pre_box_start = -len(df) + 1
        vol_prior = float(df["Volume"].iloc[pre_box_start:pre_box_end].mean())
        volume_dry = False
        if vol_prior > 0:
            volume_dry = vol_in_box < vol_prior * p.get("volume_dry_pct", 0.7)

        # Step 4: Uptrend context
        sma_200 = indicators.get("sma_200")
        in_uptrend = False
        if sma_200 is not None and not pd.isna(sma_200.iloc[-1]):
            in_uptrend = float(df["Close"].iloc[-1]) > float(sma_200.iloc[-1])

        # Advance into box
        price_before = float(df["Close"].iloc[-box_days - 20]) if len(df) > box_days + 20 else float(df["Close"].iloc[0])
        advance_pct = (box_high - price_before) / price_before * 100 if price_before > 0 else 0

        # Step 5: Score
        score = 0.0

        if coiling:
            score += 25
        if volume_dry:
            score += 20
        if in_uptrend:
            score += 15

        # Box tightness
        if box_range_pct < 5:
            score += 20
        elif box_range_pct < 8:
            score += 15
        elif box_range_pct < 12:
            score += 10

        # Duration sweet spot
        if 15 <= box_days <= 30:
            score += 10
        elif 10 <= box_days <= 40:
            score += 5

        # RS line trending up during consolidation
        rs_line = indicators.get("rs_line")
        if rs_line is not None and len(rs_line) > box_days:
            if float(rs_line.iloc[-1]) > float(rs_line.iloc[-box_days]):
                score += 10

        if score < 30:
            return None

        return PatternResult(
            pattern_name="Coil",
            ticker=ticker,
            score=round(score, 1),
            detected_date=df.index[-1].date(),
            pivot_price=round(box_high, 2),
            metadata={
                "box_high": round(box_high, 2),
                "box_low": round(box_low, 2),
                "box_range_pct": round(box_range_pct, 1),
                "box_days": box_days,
                "atr_ratio": round(atr_ratio, 2),
                "coiling": coiling,
                "volume_dry": volume_dry,
                "in_uptrend": in_uptrend,
                "advance_into_box_pct": round(advance_pct, 1),
            },
            annotations=[
                {"type": "hline", "price": box_high, "label": "Box Top"},
                {"type": "hline", "price": box_low, "label": "Box Bottom"},
                {"type": "box", "top": box_high, "bottom": box_low, "days": box_days},
            ],
        )
