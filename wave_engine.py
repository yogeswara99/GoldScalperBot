# ============================================================
#  wave_engine.py
#  Elliott Wave Detection Engine — H4 timeframe
#  Symbol: XAUUSD... (MultiBank / MEX Global)
#
#  Detects Wave 1-4 structure on H4 and generates
#  Wave 5 entry signals with Fibonacci targets
# ============================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

# ---- CONFIGURATION -----------------------------------------
from config import *
PIVOT_STRENGTH  = 5        # Bars each side to confirm a swing pivot
MIN_WAVE_PTS    = 1000     # Minimum wave size in points ($10 on Gold)
MAX_W2_RETRACE  = 0.990    # Max Wave 2 retracement of Wave 1
MIN_W2_RETRACE  = 0.236    # Min Wave 2 retracement of Wave 1
MAX_W4_RETRACE  = 0.618    # Max Wave 4 retracement of Wave 3
MIN_W4_RETRACE  = 0.236    # Min Wave 4 retracement of Wave 3
POINT           = 0.01     # XAUUSD.. point size
# ------------------------------------------------------------

# Fibonacci constants
FIB_236  = 0.236
FIB_382  = 0.382
FIB_618  = 0.618
FIB_1000 = 1.000
FIB_1618 = 1.618


# ============================================================
#  DATA CLASSES
# ============================================================

@dataclass
class Pivot:
    """A single confirmed swing high or swing low"""
    time:     datetime
    price:    float
    bar_idx:  int
    is_high:  bool


@dataclass
class WaveCount:
    """A validated Elliott Wave 1-4 structure"""
    pivots:       list         # 5 Pivot objects: [origin, W1end, W2end, W3end, W4end]
    is_bullish:   bool = False
    w1_length:    float = 0.0
    w3_length:    float = 0.0
    w2_retrace:   float = 0.0
    w4_retrace:   float = 0.0
    w3_momentum:  float = 0.0  # W3/W1 ratio — >=1.618 = extended (A+ grade)
    fib_tp1:      float = 0.0  # 61.8% projection
    fib_tp2:      float = 0.0  # 100% projection
    fib_tp3:      float = 0.0  # 161.8% projection
    proj_high:    float = 0.0  # Upper buffer of W5 projection band
    proj_low:     float = 0.0  # Lower buffer of W5 projection band
    invalid_level:float = 0.0  # Price that kills this count
    detected_at:  Optional[datetime] = None
    grade:        str = "B"    # "A+" if W3 extended, "B" otherwise


@dataclass
class WaveSignal:
    """Output signal package for the signal engine"""
    direction:    str = "NONE"   # "LONG", "SHORT", "NONE", "INVALID"
    entry_price:  float = 0.0
    stop_loss:    float = 0.0
    tp1:          float = 0.0
    tp2:          float = 0.0
    tp3:          float = 0.0
    w3_momentum:  float = 0.0
    w3_extended:  bool = False
    grade:        str = "NONE"
    description:  str = "No signal"
    wave_count:   Optional[WaveCount] = None


# ============================================================
#  WAVE ENGINE CLASS
# ============================================================

class WaveEngine:
    """
    Elliott Wave detection engine.
    Detects Wave 1-4 on H4 and signals Wave 5 entries.
    All analysis based on H4 regardless of execution timeframe.
    """

    def __init__(self,
                 symbol:         str   = SYMBOL,
                 pivot_strength: int   = PIVOT_STRENGTH,
                 min_wave_pts:   float = MIN_WAVE_PTS,
                 max_w2_retrace: float = MAX_W2_RETRACE,
                 min_w2_retrace: float = MIN_W2_RETRACE,
                 max_w4_retrace: float = MAX_W4_RETRACE,
                 min_w4_retrace: float = MIN_W4_RETRACE):

        self.symbol         = symbol
        self.pivot_strength = pivot_strength
        self.min_wave_pts   = min_wave_pts
        self.max_w2_retrace = max_w2_retrace
        self.min_w2_retrace = min_w2_retrace
        self.max_w4_retrace = max_w4_retrace
        self.min_w4_retrace = min_w4_retrace
        self.point          = POINT

        self.pivots:     list            = []
        self.best_count: Optional[WaveCount] = None

        print(f"WaveEngine initialised | Symbol: {symbol} | "
              f"HTF: H4 | PivotStrength: {pivot_strength} | "
              f"MinWave: {min_wave_pts} pts (${min_wave_pts * POINT:.0f})")


    # ----------------------------------------------------------
    #  DATA FETCHING
    # ----------------------------------------------------------

    def get_h4_data(self, bars: int = 300) -> Optional[pd.DataFrame]:
        """Fetch H4 OHLCV data from MT5"""
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, bars)

        if rates is None or len(rates) == 0:
            print(f"WaveEngine ERROR: Could not fetch H4 data — {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open":  "Open",
            "high":  "High",
            "low":   "Low",
            "close": "Close",
            "tick_volume": "Volume"
        }, inplace=True)
        return df


    # ----------------------------------------------------------
    #  PIVOT DETECTION
    # ----------------------------------------------------------

    def detect_pivots(self, df: pd.DataFrame) -> list:
        """
        Find all confirmed swing highs and lows on H4.
        A pivot high: bar[i].high is highest among (strength) bars each side.
        A pivot low:  bar[i].low  is lowest  among (strength) bars each side.
        """
        pivots = []
        highs  = df["High"].values
        lows   = df["Low"].values
        times  = df.index
        n      = len(df)
        s      = self.pivot_strength

        for i in range(s, n - s):
            left_highs  = highs[i-s : i]
            right_highs = highs[i+1 : i+s+1]
            left_lows   = lows[i-s  : i]
            right_lows  = lows[i+1  : i+s+1]

            is_ph = (highs[i] > left_highs.max()) and (highs[i] > right_highs.max())
            is_pl = (lows[i]  < left_lows.min())  and (lows[i]  < right_lows.min())

            # Cannot be both simultaneously
            if is_ph and is_pl:
                continue

            if is_ph:
                pivots.append(Pivot(
                    time    = times[i].to_pydatetime(),
                    price   = highs[i],
                    bar_idx = i,
                    is_high = True
                ))
            elif is_pl:
                pivots.append(Pivot(
                    time    = times[i].to_pydatetime(),
                    price   = lows[i],
                    bar_idx = i,
                    is_high = False
                ))

        return pivots


    # ----------------------------------------------------------
    #  WAVE RULE VALIDATION
    # ----------------------------------------------------------

    def validate_wave_rules(self, p: list) -> Optional[WaveCount]:
        """
        Validate 5 pivots against Classic Elliott Wave rules.
        p = [origin, W1end, W2end, W3end, W4end]
        Returns WaveCount if valid, None if any rule fails.
        """
        p0, p1, p2, p3, p4 = p[0].price, p[1].price, p[2].price, p[3].price, p[4].price

        # --- Direction ---
        is_bullish = (p1 > p0)

        # --- Alternation check ---
        # Bullish: low→high→low→high→low
        # Bearish: high→low→high→low→high
        if is_bullish:
            if not (p1>p0 and p2<p1 and p3>p2 and p4<p3):
                return None
        else:
            if not (p1<p0 and p2>p1 and p3<p2 and p4>p3):
                return None

        # --- Wave lengths ---
        w1_len = abs(p1 - p0)
        w2_len = abs(p2 - p1)
        w3_len = abs(p3 - p2)
        w4_len = abs(p4 - p3)

        # --- Minimum wave size filter ---
        min_move = self.min_wave_pts * self.point
        if w1_len < min_move: return None
        if w2_len < min_move: return None
        if w3_len < min_move: return None
        if w4_len < min_move: return None

        # --- Rule 1: Wave 2 cannot breach Wave 1 origin ---
        if is_bullish  and p2 <= p0: return None
        if not is_bullish and p2 >= p0: return None

        # --- Rule 2: Wave 3 never the shortest (vs Wave 1) ---
        if w3_len <= w1_len * 0.90: return None

        # --- Rule 3: Wave 4 cannot overlap Wave 1 ---
        if is_bullish  and p4 < p1: return None
        if not is_bullish and p4 > p1: return None

        # --- Wave 2 retracement check ---
        w2_retrace = w2_len / w1_len
        if w2_retrace < self.min_w2_retrace: return None
        if w2_retrace > self.max_w2_retrace: return None

        # --- Wave 4 retracement check ---
        w4_retrace = w4_len / w3_len
        if w4_retrace < self.min_w4_retrace: return None
        if w4_retrace > self.max_w4_retrace: return None

        # --- Wave 3 momentum ratio ---
        w3_momentum = w3_len / w1_len

        # --- Calculate Fibonacci targets ---
        w4_end = p[4].price
        half_min = self.min_wave_pts * self.point * 0.5

        if is_bullish:
            tp1     = w4_end + w1_len * FIB_618
            tp2     = w4_end + w1_len * FIB_1000
            tp3     = w4_end + w1_len * FIB_1618
            inv_lvl = w4_end - half_min
            band    = w1_len * FIB_382 * 0.5
        else:
            tp1     = w4_end - w1_len * FIB_618
            tp2     = w4_end - w1_len * FIB_1000
            tp3     = w4_end - w1_len * FIB_1618
            inv_lvl = w4_end + half_min
            band    = w1_len * FIB_382 * 0.5

        grade = "A+" if w3_momentum >= FIB_1618 else "B"

        return WaveCount(
            pivots       = p,
            is_bullish   = is_bullish,
            w1_length    = w1_len,
            w3_length    = w3_len,
            w2_retrace   = w2_retrace,
            w4_retrace   = w4_retrace,
            w3_momentum  = w3_momentum,
            fib_tp1      = tp1,
            fib_tp2      = tp2,
            fib_tp3      = tp3,
            proj_high    = tp2 + band,
            proj_low     = tp2 - band,
            invalid_level= inv_lvl,
            detected_at  = datetime.now(timezone.utc),
            grade        = grade
        )


    # ----------------------------------------------------------
    #  FIND BEST WAVE COUNT
    # ----------------------------------------------------------

    def find_best_count(self, pivots: list) -> Optional[WaveCount]:
        """
        Scan all sets of 5 consecutive pivots from most recent backwards.
        Return the first (most recent) valid wave count found.
        """
        n = len(pivots)
        if n < 5:
            return None

        # Scan from most recent set of 5 pivots backwards
        for i in range(n - 1, 3, -1):
            candidate = [
                pivots[i-4],
                pivots[i-3],
                pivots[i-2],
                pivots[i-1],
                pivots[i]
            ]
            result = self.validate_wave_rules(candidate)
            if result is not None:
                return result

        return None


    # ----------------------------------------------------------
    #  INVALIDATION CHECK
    # ----------------------------------------------------------

    def is_invalidated(self, count: WaveCount, current_price: float) -> bool:
        """Check if current price has breached the invalidation level"""
        if count.is_bullish and current_price < count.invalid_level:
            print(f"WaveEngine: BULL count INVALIDATED — "
                  f"price {current_price:.2f} < level {count.invalid_level:.2f}")
            return True
        if not count.is_bullish and current_price > count.invalid_level:
            print(f"WaveEngine: BEAR count INVALIDATED — "
                  f"price {current_price:.2f} > level {count.invalid_level:.2f}")
            return True
        return False


    # ----------------------------------------------------------
    #  LOGGING
    # ----------------------------------------------------------

    def log_count(self, wc: WaveCount):
        ext = " [EXTENDED W3 - A+]" if wc.w3_momentum >= FIB_1618 else ""
        print("=" * 45)
        print(f"  WaveEngine H4 — Valid Count Found{ext}")
        print("=" * 45)
        print(f"  Direction  : {'BULLISH' if wc.is_bullish else 'BEARISH'}")
        print(f"  Grade      : {wc.grade}")
        print(f"  W1 length  : {wc.w1_length / self.point:.0f} pts  (${wc.w1_length:.2f})")
        print(f"  W2 retrace : {wc.w2_retrace*100:.1f}%")
        print(f"  W3 length  : {wc.w3_length / self.point:.0f} pts  (${wc.w3_length:.2f})")
        print(f"  W3/W1 ratio: {wc.w3_momentum:.3f}x")
        print(f"  W4 retrace : {wc.w4_retrace*100:.1f}%")
        print(f"  W4 end     : {wc.pivots[4].price:.2f}")
        print(f"  Invalidate : {wc.invalid_level:.2f}")
        print(f"  TP1 (61.8%): {wc.fib_tp1:.2f}  (+${wc.fib_tp1 - wc.pivots[4].price:.2f})" if wc.is_bullish
              else f"  TP1 (61.8%): {wc.fib_tp1:.2f}  (-${wc.pivots[4].price - wc.fib_tp1:.2f})")
        print(f"  TP2 (100%) : {wc.fib_tp2:.2f}")
        print(f"  TP3 (161.8%): {wc.fib_tp3:.2f}")
        print("=" * 45)


    # ----------------------------------------------------------
    #  MAIN UPDATE — call this on every new H4 candle
    # ----------------------------------------------------------

    def update(self) -> WaveSignal:
        """
        Main method. Fetches H4 data, detects pivots,
        validates wave count, returns WaveSignal.
        Call once per new H4 candle.
        """
        sig = WaveSignal()

        # Step 1: Get live H4 data
        df = self.get_h4_data(bars=300)
        if df is None:
            sig.description = "Failed to fetch H4 data"
            return sig

        # Step 2: Detect pivots
        self.pivots = self.detect_pivots(df)
        if len(self.pivots) < 5:
            sig.description = f"Only {len(self.pivots)} pivots found — need 5+"
            return sig

        # Step 3: Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            sig.description = "Could not get live tick"
            return sig
        bid = tick.bid
        ask = tick.ask

        # Step 4: Check if existing count is invalidated
        if self.best_count is not None:
            if self.is_invalidated(self.best_count, bid):
                self.best_count = None
                sig.direction   = "INVALID"
                sig.description = "Wave count invalidated — rescanning"
                return sig

        # Step 5: Find best wave count
        count = self.find_best_count(self.pivots)
        if count is None:
            sig.description = (f"{len(self.pivots)} H4 pivots found — "
                               f"no valid W1-W4 structure")
            return sig

        self.best_count = count
        self.log_count(count)

        # Step 6: Check for Wave 5 entry
        w4_end = count.pivots[4].price
        buffer = self.min_wave_pts * self.point * 0.3

        w5_bull = count.is_bullish  and bid > w4_end + buffer
        w5_bear = not count.is_bullish and bid < w4_end - buffer

        if w5_bull or w5_bear:
            direction  = "LONG" if count.is_bullish else "SHORT"
            entry      = ask if count.is_bullish else bid

            sig.direction   = direction
            sig.entry_price = entry
            sig.stop_loss   = count.invalid_level
            sig.tp1         = count.fib_tp1
            sig.tp2         = count.fib_tp2
            sig.tp3         = count.fib_tp3
            sig.w3_momentum = count.w3_momentum
            sig.w3_extended = (count.w3_momentum >= FIB_1618)
            sig.grade       = count.grade
            sig.wave_count  = count
            sig.description = (
                f"WAVE 5 {direction} | Grade:{count.grade} | "
                f"Entry:{entry:.2f} SL:{count.invalid_level:.2f} "
                f"TP1:{count.fib_tp1:.2f} TP2:{count.fib_tp2:.2f} "
                f"TP3:{count.fib_tp3:.2f}"
            )
            print(f"WaveEngine SIGNAL: {sig.description}")

        else:
            direction = "BULL" if count.is_bullish else "BEAR"
            sig.description = (
                f"H4 Wave 1-4 VALID ({direction}) | "
                f"Grade:{count.grade} | "
                f"Waiting for W5 break of {w4_end:.2f}"
            )
            sig.wave_count = count

        return sig


    # ----------------------------------------------------------
    #  STATUS
    # ----------------------------------------------------------

    def get_status(self) -> str:
        if self.best_count is None:
            return f"No valid H4 wave count | Pivots: {len(self.pivots)}"
        direction = "BULL" if self.best_count.is_bullish else "BEAR"
        return (f"H4 Wave 1-4 VALID ({direction}) | "
                f"Grade: {self.best_count.grade} | "
                f"W4 end: {self.best_count.pivots[4].price:.2f} | "
                f"Inv: {self.best_count.invalid_level:.2f}")


# ============================================================
#  QUICK TEST — run this file directly to test the engine
# ============================================================

if __name__ == "__main__":

    print("Testing WaveEngine...")
    print()

    # Connect to MT5
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        exit(1)

    print(f"MT5 connected")

    # Create and run engine
    engine = WaveEngine(
        symbol         = SYMBOL,
        pivot_strength = PIVOT_STRENGTH,
        min_wave_pts   = MIN_WAVE_PTS
    )

    print()
    print("Running wave detection on H4 data...")
    print()

    signal = engine.update()

    print()
    print("=" * 45)
    print("  WAVE ENGINE TEST RESULT")
    print("=" * 45)
    print(f"  Signal    : {signal.direction}")
    print(f"  Grade     : {signal.grade}")
    print(f"  Description: {signal.description}")
    print(f"  Pivots detected: {len(engine.pivots)}")
    print()
    print(f"  Engine status: {engine.get_status()}")
    print("=" * 45)

    mt5.shutdown()
    print("\nDone.")
