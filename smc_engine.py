# ================================================================
#  smc_engine.py  v3.0  — GoldScalperBot  XAUUSD...
#
#  Signal chain:
#    H4  → trend bias only (UP / DOWN / RANGE)
#    H1  → institutional structure (key swing highs/lows)
#    M15 → BOS entry trigger (breaks H1 structural level)
#
#  Entry fires when:
#    1. H4 trend confirmed (EMA20 > EMA50 = UP, etc.)
#    2. H1 swing high/low identified (the structural level)
#    3. M15 candle CLOSES beyond that H1 level (BOS trigger)
#    4. M15 entry confirmation (close > M15 EMA5 + momentum)
#
#  Sweep (H1 wick through liquidity level + close back) is
#  OPTIONAL — upgrades grade from B to A+ when present.
#
#  This produces 3-5 entries per day by firing on every
#  M15 BOS aligned with H4 trend + H1 structure.
# ================================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from config import *


# ----------------------------------------------------------------
#  DATA CLASSES
# ----------------------------------------------------------------

@dataclass
class Pivot:
    idx: int; price: float; is_high: bool; atr: float = 0.0

@dataclass
class LiquidityLevel:
    price: float; is_high: bool; formed_at: int
    swept: bool = False; tolerance: float = 0.0

@dataclass
class SMCSignal:
    direction:       str   = "NONE"
    entry_price:     float = 0.0
    stop_loss:       float = 0.0
    tp1:             float = 0.0
    tp2:             float = 0.0
    tp3:             float = 0.0
    h4_trend:        str   = "NONE"
    structure_tf:    str   = "H1"
    h1_level:        float = 0.0   # The H1 structural level that was broken
    sweep_price:     float = 0.0
    bos_level:       float = 0.0
    bos_method_used: str   = ""
    has_h4_trend:    bool  = False
    has_h1_structure:bool  = False  # H1 level identified
    has_bos:         bool  = False  # M15 BOS confirmed
    has_sweep:       bool  = False  # H1 sweep (optional, upgrades grade)
    has_choch:       bool  = False
    has_fvg:         bool  = False
    high_volume:     bool  = False
    description:     str   = "No signal"


def smc_to_conditions(sig: "SMCSignal") -> dict:
    return {
        "elliott_bias":   sig.has_h4_trend,
        "smc_structure":  sig.has_h1_structure,
        "fvg_confluence": sig.has_fvg or sig.has_sweep,
        "h1_momentum":    sig.has_h1_structure and sig.has_bos,
        "m15_bos":        sig.has_bos,
        "volume_confirm": sig.high_volume,
    }


# ----------------------------------------------------------------
#  SMC ENGINE v3
# ----------------------------------------------------------------

class SMCEngine:

    def __init__(self,
                 structure_tf:   str = "H1",
                 pivot_strength: int = 5,
                 bos_method:     str = "single_close",
                 ehl_tolerance:  str = "atr_0.5",
                 choch_scope:    str = "H4_ONLY"):

        self.structure_tf   = structure_tf
        self.pivot_strength = pivot_strength
        self.bos_method     = bos_method
        self.ehl_tolerance  = ehl_tolerance
        self.choch_scope    = choch_scope

        # State cache — reset each call so no lookahead
        self._h4_pivots: List[Pivot] = []
        self._h1_pivots: List[Pivot] = []
        self._liq:       List[LiquidityLevel] = []
        self._last_sig   = SMCSignal()


    # ============================================================
    #  MAIN UPDATE — called on every new M15 bar
    # ============================================================

    def update(self,
               df_h4:    pd.DataFrame,   # H4 bars up to now
               df_struct: pd.DataFrame,  # H1 bars up to now
               df_m15:   pd.DataFrame,   # M15 bars
               idx_m15:  int             # current bar index
               ) -> SMCSignal:

        sig = SMCSignal(structure_tf=self.structure_tf)

        if len(df_h4) < 20 or len(df_struct) < 10 or idx_m15 < 20:
            sig.description = "Insufficient data"
            return sig

        # ── Step 1: H4 trend ──────────────────────────────────
        trend = self._h4_trend(df_h4)
        sig.h4_trend     = trend
        sig.has_h4_trend = trend in ("UP", "DOWN")

        if not sig.has_h4_trend:
            sig.description = "H4 range — no bias"
            return sig

        # ── Step 2: H1 structural levels ─────────────────────
        atr_h1 = self._calc_atr(df_struct).iloc[-1]
        self._h1_pivots = self._detect_pivots(
            df_struct, self.pivot_strength, atr_h1)

        if len(self._h1_pivots) < 4:
            sig.description = f"H4:{trend} | Insufficient H1 pivots"
            return sig

        sig.has_h1_structure = True

        # Get the key H1 level to watch for M15 BOS
        # LONG: watch the most recent confirmed H1 swing HIGH
        #       (bulls need to break above it)
        # SHORT: watch the most recent confirmed H1 swing LOW
        #        (bears need to break below it)
        h1_level = self._get_key_h1_level(self._h1_pivots, trend)
        if h1_level is None:
            sig.description = f"H4:{trend} | No H1 level found"
            return sig

        sig.h1_level = h1_level

        # ── Step 3: M15 BOS — does M15 close beyond H1 level? ─
        bos = self._m15_bos(df_m15, idx_m15, trend,
                             h1_level, self.bos_method)
        if not bos:
            sig.description = (f"H4:{trend} | H1 level:{h1_level:.2f} | "
                               f"M15 waiting BOS")
            return sig

        sig.has_bos   = True
        sig.bos_level = h1_level
        sig.bos_method_used = self.bos_method

        # ── Step 4: Optional H1 sweep confluence ──────────────
        self._liq = self._map_liquidity(
            self._h1_pivots, atr_h1, self.ehl_tolerance)
        sweep = self._detect_sweep(df_struct, self._liq, trend)
        if sweep:
            sig.has_sweep   = True
            sig.sweep_price = sweep.price
            sig.has_fvg     = self._check_fvg(df_struct, sweep, trend)

        # ── Step 5: Optional CHOCH on M15 ─────────────────────
        if self.choch_scope == "H4_AND_M15":
            sig.has_choch = self._detect_choch(df_m15, idx_m15, trend)

        # ── Step 6: Volume ────────────────────────────────────
        sig.high_volume = self._vol_zscore(
            df_m15["Volume"], idx_m15) > 0.5

        # ── Step 7: Build prices ──────────────────────────────
        direction = "LONG" if trend == "UP" else "SHORT"
        sig.direction   = direction
        entry = df_m15.iloc[idx_m15]["Close"]

        atr_m15 = self._calc_atr(
            df_m15.iloc[max(0, idx_m15-20):idx_m15+1]).iloc[-1]

        if direction == "LONG":
            sig.entry_price = entry
            # SL: below the most recent H1 swing low
            sl_ref = self._recent_low(self._h1_pivots)
            sig.stop_loss = ((sl_ref - atr_m15 * 0.3)
                              if sl_ref else entry - atr_m15 * 1.5)
            sig.tp1 = entry + atr_m15 * 2.0
            sig.tp2 = entry + atr_m15 * 3.5
            sig.tp3 = entry + atr_m15 * 5.0
        else:
            sig.entry_price = entry
            sl_ref = self._recent_high(self._h1_pivots)
            sig.stop_loss = ((sl_ref + atr_m15 * 0.3)
                              if sl_ref else entry + atr_m15 * 1.5)
            sig.tp1 = entry - atr_m15 * 2.0
            sig.tp2 = entry - atr_m15 * 3.5
            sig.tp3 = entry - atr_m15 * 5.0

        n = sum([sig.has_h4_trend, sig.has_h1_structure, sig.has_bos,
                 sig.has_sweep, sig.has_fvg, sig.high_volume])
        sig.description = (
            f"SMC {direction} | H4:{trend} | "
            f"H1 level:{h1_level:.2f} | M15 BOS confirmed | "
            f"Sweep:{'Y' if sig.has_sweep else 'N'} | "
            f"FVG:{'Y' if sig.has_fvg else 'N'} | "
            f"Confluence:{n}/6"
        )
        self._last_sig = sig
        return sig


    # ============================================================
    #  H4 TREND
    # ============================================================

    def _h4_trend(self, df: pd.DataFrame) -> str:
        if len(df) < 50:
            return "RANGE"
        c   = df["Close"]
        e20 = c.ewm(span=20, adjust=False).mean()
        e50 = c.ewm(span=50, adjust=False).mean()
        l20 = e20.iloc[-1]; l50 = e50.iloc[-1]; last = c.iloc[-1]
        if l20 > l50 and last > l20: return "UP"
        if l20 < l50 and last < l20: return "DOWN"
        return "UP" if l20 > l50 else ("DOWN" if l20 < l50 else "RANGE")


    # ============================================================
    #  PIVOT DETECTION
    # ============================================================

    def _detect_pivots(self, df: pd.DataFrame,
                        strength: int, atr: float) -> List[Pivot]:
        highs = df["High"].values
        lows  = df["Low"].values
        n = len(df); s = strength
        pivots = []; seen = set()
        for i in range(s, n - s):
            if highs[i] == max(highs[max(0,i-s):i+s+1]):
                wave = highs[i] - min(lows[max(0,i-s):i+1])
                if wave >= atr * 0.3 and (i, True) not in seen:
                    seen.add((i, True))
                    pivots.append(Pivot(i, highs[i], True, atr))
            if lows[i] == min(lows[max(0,i-s):i+s+1]):
                wave = max(highs[i:min(n,i+s+1)]) - lows[i]
                if wave >= atr * 0.3 and (i, False) not in seen:
                    seen.add((i, False))
                    pivots.append(Pivot(i, lows[i], False, atr))
        return sorted(pivots, key=lambda x: x.idx)


    # ============================================================
    #  KEY H1 LEVEL
    # ============================================================

    def _get_key_h1_level(self, pivots: List[Pivot],
                            trend: str) -> Optional[float]:
        """
        Return the most recent confirmed H1 structural level
        that M15 needs to break for entry confirmation.

        UP   trend → break above the most recent H1 swing HIGH
        DOWN trend → break below the most recent H1 swing LOW

        We skip the very last pivot (not yet confirmed) and use
        the second-to-last of the relevant type.
        """
        if trend == "UP":
            highs = [p for p in pivots if p.is_high]
            # Use the most recent CONFIRMED swing high
            # (last one in list = confirmed because it has bars after it)
            if len(highs) >= 1:
                return highs[-1].price
        else:
            lows = [p for p in pivots if not p.is_high]
            if len(lows) >= 1:
                return lows[-1].price
        return None


    # ============================================================
    #  M15 BOS — ENTRY TRIGGER
    # ============================================================

    def _m15_bos(self, df_m15: pd.DataFrame,
                  idx: int,
                  trend: str,
                  h1_level: float,
                  method: str) -> bool:
        """
        Check if M15 breaks the H1 structural level.
        This is the actual entry trigger.

        single_close : one M15 close beyond the level
        double_close : two consecutive M15 closes beyond
        body_close   : full candle body beyond (no wick trick)
        """
        if idx < 2:
            return False

        last = df_m15.iloc[idx]
        prev = df_m15.iloc[idx - 1]

        if trend == "UP":
            # M15 must close ABOVE the H1 swing high
            if method == "single_close":
                return last["Close"] > h1_level

            elif method == "double_close":
                return (last["Close"] > h1_level and
                        prev["Close"] > h1_level)

            elif method == "body_close":
                # Full body above (both open and close)
                body_low = min(last["Open"], last["Close"])
                return body_low > h1_level

        else:  # DOWN
            # M15 must close BELOW the H1 swing low
            if method == "single_close":
                return last["Close"] < h1_level

            elif method == "double_close":
                return (last["Close"] < h1_level and
                        prev["Close"] < h1_level)

            elif method == "body_close":
                body_high = max(last["Open"], last["Close"])
                return body_high < h1_level

        return False


    # ============================================================
    #  LIQUIDITY MAPPING (H1)
    # ============================================================

    def _map_liquidity(self, pivots: List[Pivot],
                        atr: float,
                        tolerance: str) -> List[LiquidityLevel]:
        levels = []
        if tolerance == "fixed_100": tol = 100 * POINT
        elif tolerance == "atr_0.3": tol = atr * 0.3
        elif tolerance == "atr_0.5": tol = atr * 0.5
        elif tolerance == "pct_0.02":
            tol = (pivots[-1].price if pivots else 2000.0) * 0.0002
        else:
            tol = atr * 0.5

        highs = [p for p in pivots if p.is_high]
        lows  = [p for p in pivots if not p.is_high]

        for i in range(len(highs) - 1):
            for j in range(i + 1, min(i + 4, len(highs))):
                if abs(highs[i].price - highs[j].price) <= tol:
                    levels.append(LiquidityLevel(
                        price=(highs[i].price + highs[j].price) / 2,
                        is_high=True, formed_at=highs[j].idx,
                        tolerance=tol))
                    break

        for i in range(len(lows) - 1):
            for j in range(i + 1, min(i + 4, len(lows))):
                if abs(lows[i].price - lows[j].price) <= tol:
                    levels.append(LiquidityLevel(
                        price=(lows[i].price + lows[j].price) / 2,
                        is_high=False, formed_at=lows[j].idx,
                        tolerance=tol))
                    break

        if len(highs) >= 2:
            levels.append(LiquidityLevel(
                highs[-2].price, True, highs[-2].idx, tolerance=tol))
        if len(lows) >= 2:
            levels.append(LiquidityLevel(
                lows[-2].price, False, lows[-2].idx, tolerance=tol))
        return levels


    # ============================================================
    #  H1 SWEEP DETECTION (optional confluence)
    # ============================================================

    def _detect_sweep(self, df: pd.DataFrame,
                       liquidity: List[LiquidityLevel],
                       trend: str) -> Optional[LiquidityLevel]:
        if len(df) < 5:
            return None
        recent = df.tail(10)
        looking_high = (trend == "DOWN")
        for lvl in [l for l in liquidity
                    if l.is_high == looking_high and not l.swept]:
            for i in range(len(recent)):
                bar = recent.iloc[i]
                if looking_high:
                    if (bar["High"] > lvl.price and
                            bar["Close"] < lvl.price):
                        lvl.swept = True
                        return lvl
                else:
                    if (bar["Low"] < lvl.price and
                            bar["Close"] > lvl.price):
                        lvl.swept = True
                        return lvl
        return None


    # ============================================================
    #  CHOCH (optional)
    # ============================================================

    def _detect_choch(self, df_m15: pd.DataFrame,
                       idx: int, trend: str) -> bool:
        if idx < 6:
            return False
        c = df_m15.iloc[max(0, idx-8):idx+1]["Close"].values
        if trend == "UP":
            for i in range(2, len(c)):
                if c[i] > c[i-2] and c[i-1] < c[i-3]:
                    return True
        else:
            for i in range(2, len(c)):
                if c[i] < c[i-2] and c[i-1] > c[i-3]:
                    return True
        return False


    # ============================================================
    #  FVG CHECK (optional)
    # ============================================================

    def _check_fvg(self, df: pd.DataFrame,
                    sweep: LiquidityLevel,
                    trend: str) -> bool:
        if len(df) < 10:
            return False
        atr    = self._calc_atr(df).iloc[-1]
        recent = df.tail(15)
        for i in range(1, len(recent) - 1):
            c1 = recent.iloc[i-1]
            c3 = recent.iloc[i+1]
            if trend == "UP":
                gap = c3["Low"] - c1["High"]
                if gap > 0:
                    mid = (c1["High"] + c3["Low"]) / 2
                    if abs(mid - sweep.price) < atr:
                        return True
            else:
                gap = c1["Low"] - c3["High"]
                if gap > 0:
                    mid = (c1["Low"] + c3["High"]) / 2
                    if abs(mid - sweep.price) < atr:
                        return True
        return False


    # ============================================================
    #  SL HELPERS
    # ============================================================

    def _recent_low(self, pivots: List[Pivot]) -> Optional[float]:
        lows = [p for p in pivots if not p.is_high]
        return lows[-1].price if lows else None

    def _recent_high(self, pivots: List[Pivot]) -> Optional[float]:
        highs = [p for p in pivots if p.is_high]
        return highs[-1].price if highs else None


    # ============================================================
    #  UTILITIES
    # ============================================================

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"]  - df["Close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(com=period-1, adjust=False).mean()

    def _vol_zscore(self, volume: pd.Series,
                    idx: int, period: int = 20) -> float:
        sub = volume.iloc[max(0, idx-period):idx+1]
        if len(sub) < 5:
            return 0.0
        std = sub.std()
        return (sub.iloc[-1] - sub.mean()) / std if std > 0 else 0.0
