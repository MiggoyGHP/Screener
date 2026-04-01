from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from screener.patterns.base import PatternDetector, PatternResult, zigzag
from screener.patterns.trend_template import check_trend_template


class VCPDetector(PatternDetector):
    name = "VCP"

    def detect(
        self, ticker: str, df: pd.DataFrame, indicators: dict[str, Any]
    ) -> PatternResult | None:
        p = self.params

        # Prerequisite: must pass Trend Template
        passes, _ = check_trend_template(df, indicators)
        if not passes:
            return None

        lookback = min(p.get("lookback_days", 180), len(df))
        if lookback < 40:
            return None

        window = df.iloc[-lookback:]
        high = window["High"]
        low = window["Low"]
        close = window["Close"]
        volume = window["Volume"]

        # Step 1: Find the left-side high and base depth
        lsh_pos = int(high.values.argmax())
        lsh_price = float(high.iloc[lsh_pos])

        if lsh_pos >= len(window) - 5:
            return None  # LSH is too recent

        post_lsh = window.iloc[lsh_pos:]
        base_low = float(post_lsh["Low"].min())
        base_depth = (lsh_price - base_low) / lsh_price * 100

        min_depth = p.get("min_base_depth", 10)
        max_depth = p.get("max_base_depth", 50)
        if base_depth < min_depth or base_depth > max_depth:
            return None

        # Step 2: Zigzag swing detection within the base
        zz_threshold = max(3.0, base_depth / 5)
        pivots = zigzag(
            post_lsh["High"].reset_index(drop=True),
            post_lsh["Low"].reset_index(drop=True),
            pct_threshold=zz_threshold,
        )

        # Extract swing highs and lows in order
        swing_highs = [(idx, price) for typ, idx, price in pivots if typ == "high"]
        swing_lows = [(idx, price) for typ, idx, price in pivots if typ == "low"]

        if len(swing_highs) < 2 or len(swing_lows) < 1:
            return None

        # Step 3: Measure contractions
        contractions = []
        n_pairs = min(len(swing_highs), len(swing_lows))
        for i in range(n_pairs):
            sh_idx, sh_price = swing_highs[i]
            sl_idx, sl_price = swing_lows[i]
            if sh_price <= 0:
                continue
            depth_pct = (sh_price - sl_price) / sh_price * 100
            if depth_pct <= 0:
                continue
            # Volume during this contraction
            start = min(sh_idx, sl_idx)
            end = max(sh_idx, sl_idx) + 1
            vol_slice = post_lsh["Volume"].iloc[start:end]
            avg_vol = float(vol_slice.mean()) if len(vol_slice) > 0 else 0

            contractions.append({
                "depth": depth_pct,
                "high": sh_price,
                "low": sl_price,
                "avg_volume": avg_vol,
                "high_idx": sh_idx,
                "low_idx": sl_idx,
            })

        if len(contractions) < p.get("min_contractions", 2):
            return None

        # Step 4: Validate contractions are tightening
        decay = p.get("contraction_decay", 0.65)
        valid_contractions = [contractions[0]]
        for i in range(1, len(contractions)):
            if contractions[i]["depth"] <= contractions[i - 1]["depth"] * (1 + decay):
                valid_contractions.append(contractions[i])
            else:
                break

        if len(valid_contractions) < p.get("min_contractions", 2):
            return None

        # Step 5: Volume decay check
        vol_decay = p.get("volume_decay", 0.85)
        volume_declining = True
        for i in range(1, len(valid_contractions)):
            if valid_contractions[i]["avg_volume"] > valid_contractions[i - 1]["avg_volume"] * (1 + vol_decay):
                volume_declining = False
                break

        # Step 6: Pivot point
        pivot_price = valid_contractions[-1]["high"]

        # Step 7: Proximity check
        current_price = float(df["Close"].iloc[-1])
        proximity = p.get("pivot_proximity", 0.03)
        distance_to_pivot = (pivot_price - current_price) / pivot_price
        near_pivot = -0.02 <= distance_to_pivot <= proximity

        # Step 8: Score
        score = 0.0
        n_cont = min(len(valid_contractions), 4)
        score += n_cont * 10  # max 40

        last_depth = valid_contractions[-1]["depth"]
        if last_depth < 5:
            score += 20
        elif last_depth < 8:
            score += 15
        elif last_depth < 12:
            score += 10
        else:
            score += 5

        if volume_declining:
            score += 15

        if near_pivot:
            score += 15

        rs_rank = indicators.get("rs_rank", 50)
        score += (rs_rank / 100) * 10

        return PatternResult(
            pattern_name="VCP",
            ticker=ticker,
            score=round(score, 1),
            detected_date=df.index[-1].date(),
            pivot_price=round(pivot_price, 2),
            metadata={
                "num_contractions": len(valid_contractions),
                "contractions": valid_contractions,
                "base_depth": round(base_depth, 1),
                "volume_declining": volume_declining,
                "near_pivot": near_pivot,
                "distance_to_pivot_pct": round(distance_to_pivot * 100, 2),
            },
            annotations=[
                {"type": "hline", "price": pivot_price, "label": "Pivot"},
            ],
        )
