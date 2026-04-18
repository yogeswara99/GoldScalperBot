# ================================================================
#  breakeven_analyser.py  --  GoldScalperBot  XAUUSD...
#
#  PURPOSE:
#    Systematically test all 16 breakeven trail combinations and
#    analyse market conditions (volume, volatility, RSI, EMA) at
#    entry time to find the optimal BE parameters.
#
#  BREAKEVEN GRID (4 x 4 = 16 combinations):
#    Profit trigger  : 3.5x, 4.5x, 5.5x, 6.0x spread
#    SL lock level   : entry + 1x, 2x, 3x, 4x spread
#
#  CONDITION ANALYSIS (5-bar and 10-bar lookback on M15):
#    Volume  : z-score, 5-bar slope, 5-bar ratio
#    Volatility : ATR ratio (current / 20-bar mean), 5-bar slope
#    RSI     : level, 3-bar slope, 5-bar slope
#    EMA20   : 3-bar slope, 5-bar slope
#    Price   : body ratio, close position in range, 5-bar momentum
#    Session : hour, day, London/NY
#
#  OUTCOME BUCKETS:
#    TP           -- hit take profit
#    SL_LOSS      -- hit original SL (pnl < -$1)
#    SL_BREAKEVEN -- SL was trailed to entry, then hit (pnl >= -$1)
#    MOMENTUM     -- momentum exit at profit
#
#  OUTPUTS:
#    analysis_results/be_grid_results.csv      -- 16-combo comparison
#    analysis_results/be_trade_conditions.csv  -- per-trade conditions
#    analysis_results/be_ml_features.csv       -- XGBoost-ready matrix
#    Console: full comparison tables + recommendation
#
#  USAGE: python breakeven_analyser.py
#  REQUIRES: MT5 open and connected
# ================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import csv, os, time, itertools
from config import *

# ---- IMPORT BASE ENGINE ------------------------------------
from backtest_engine import (
    DataLoader, Indicators, ScoringEngine,
    BacktestTrade, BacktestResult,
    INSAMPLE_START, INSAMPLE_END,
    SLIPPAGE_POINTS
)

OUTPUT_DIR = "analysis_results"

# ---- BREAKEVEN GRID ----------------------------------------
# (profit_trigger_mult, sl_lock_mult)  -- multiples of spread
BE_GRID = [
    (3.5, 1), (3.5, 2), (3.5, 3), (3.5, 4),
    (4.5, 1), (4.5, 2), (4.5, 3), (4.5, 4),
    (5.5, 1), (5.5, 2), (5.5, 3), (5.5, 4),
    (6.0, 1), (6.0, 2), (6.0, 3), (6.0, 4),
]

# ---- RATE-OF-CHANGE WINDOWS --------------------------------
# 5 bars  = 75  min  on M15  (1 trading hour   -- primary window)
# 10 bars = 150 min  on M15  (2.5 trading hours -- secondary window)
ROC_SHORT = 5
ROC_LONG  = 10


# ================================================================
#  EXTENDED TRADE DATACLASS
# ================================================================

@dataclass
class ExtendedTrade(BacktestTrade):
    # Breakeven tracking
    be_activated: bool = False      # True once SL was trailed to entry
    be_profit_trigger: float = 0.0  # profit (pts) that triggered the BE trail

    # Entry conditions
    vol_z:         float = 0.0   # volume z-score at entry
    vol_5bar_slope:float = 0.0   # volume linear slope over 5 bars
    vol_ratio:     float = 0.0   # volume[0] / mean(volume[-5:])

    atr_pts:       float = 0.0   # ATR in points at entry
    atr_ratio:     float = 0.0   # ATR / 20-bar mean ATR
    atr_5bar_roc:  float = 0.0   # (ATR[0] - ATR[-5]) / ATR[-5]

    rsi_level:     float = 50.0  # RSI at entry
    rsi_3bar_slope:float = 0.0   # RSI[0] - RSI[-3]
    rsi_5bar_slope:float = 0.0   # RSI[0] - RSI[-5]

    ema20_3bar_slope:float = 0.0  # EMA20[0] - EMA20[-3]
    ema20_5bar_slope:float = 0.0  # EMA20[0] - EMA20[-5]

    body_ratio:    float = 0.0   # candle body / range
    close_pos:     float = 0.0   # (close - low) / range  (0=low, 1=high)
    price_5bar_mom:float = 0.0   # (close - close[-5]) / ATR  (normalised)

    session:       str   = ""    # "London" or "NY"
    hour:          int   = 0
    day_of_week:   int   = 0     # 0=Monday ... 4=Friday

    outcome:       str   = ""    # TP / SL_LOSS / SL_BREAKEVEN / MOMENTUM


# ================================================================
#  PARAMETRIZED SIMULATOR (accepts BE params per run)
# ================================================================

class ParametrizedSimulator:
    """
    Minimal trade simulator with configurable breakeven parameters.
    profit_trigger_mult x spread = distance at which SL is trailed
    sl_lock_mult x spread        = how far above entry the SL is locked
    """
    def __init__(self, profit_trigger_mult: float, sl_lock_mult: float):
        self.pt_mult = profit_trigger_mult
        self.sl_mult = sl_lock_mult
        self.balance       = ACCOUNT_BALANCE
        self.equity        = ACCOUNT_BALANCE
        self.peak_equity   = ACCOUNT_BALANCE
        self.open_trades:   List[ExtendedTrade] = []
        self.closed_trades: List[ExtendedTrade] = []
        self.trade_counter = 0
        self.daily_loss    = 0.0
        self.weekly_loss   = 0.0
        self.session_losses= 0
        self.equity_curve  = [ACCOUNT_BALANCE]
        self.session_trade_count = 0
        self.current_session_key = None

    def reset(self):
        self.__init__(self.pt_mult, self.sl_mult)

    def calc_lot_size(self, sl_pts, pw, scorer):
        risk = self.equity * RISK_NORMAL_PCT
        base = risk / (sl_pts * CONTRACT_SIZE * POINT)
        raw  = base * scorer.lot_multiplier(pw)
        lot  = round(raw / LOT_STEP) * LOT_STEP
        return max(LOT_MIN, min(LOT_MAX, round(lot, 2)))

    def open_trade(self, direction, entry, sl, tp, lot, spread,
                   score, grade, pw, open_time, conditions_dict):
        self.trade_counter += 1
        sp = spread * POINT
        adj_e  = entry + SLIPPAGE_POINTS*POINT if direction=="LONG" \
                 else entry - SLIPPAGE_POINTS*POINT
        adj_sl = sl - sp if direction=="LONG" else sl + sp
        adj_tp = tp + sp if direction=="LONG" else tp - sp

        t = ExtendedTrade(
            trade_id=self.trade_counter, direction=direction,
            open_time=open_time, close_time=None,
            entry_price=adj_e, sl_price=adj_sl, tp_price=adj_tp,
            lot_size=lot, spread_at_entry=spread,
            signal_score=score, signal_grade=grade, prob_weight=pw,
            is_open=True)

        # Attach conditions
        for k, v in conditions_dict.items():
            if hasattr(t, k):
                setattr(t, k, v)

        self.open_trades.append(t)
        self.session_trade_count += 1
        return t

    def update_trades(self, bar, bar_time, atr_val, rsi_slope):
        to_close = []
        for t in self.open_trades:
            cp = None; reason = ""
            sp  = t.spread_at_entry * POINT
            min_profit = sp * self.pt_mult   # configurable trigger

            if t.direction == "LONG":
                cur    = bar["Close"] - t.entry_price
                sl_hit = bar["Low"]  <= t.sl_price
                tp_hit = bar["High"] >= t.tp_price
                if sl_hit and tp_hit:
                    dist_sl = abs(bar["Open"] - t.sl_price)
                    dist_tp = abs(bar["Open"] - t.tp_price)
                    cp, reason = (t.tp_price,"TP") if dist_tp<=dist_sl else (t.sl_price,"SL")
                elif sl_hit: cp, reason = t.sl_price, "SL"
                elif tp_hit: cp, reason = t.tp_price, "TP"
                elif cur > min_profit and rsi_slope < -5 and \
                        bar["BodySize"] < bar["Range"] * 0.3:
                    cp, reason = bar["Close"], "MOMENTUM"
                elif cur > min_profit and t.sl_price < t.entry_price:
                    # Trail SL to entry + sl_lock_mult * spread
                    new_sl = t.entry_price + sp * self.sl_mult
                    if new_sl > t.sl_price:
                        t.sl_price = new_sl
                        t.be_activated = True
                        t.be_profit_trigger = cur / POINT
                        t.tp_price = bar["High"] + atr_val * TP_TRAIL_ATR_MULT
            else:
                cur    = t.entry_price - bar["Close"]
                sl_hit = bar["High"] >= t.sl_price
                tp_hit = bar["Low"]  <= t.tp_price
                if sl_hit and tp_hit:
                    dist_sl = abs(bar["Open"] - t.sl_price)
                    dist_tp = abs(bar["Open"] - t.tp_price)
                    cp, reason = (t.tp_price,"TP") if dist_tp<=dist_sl else (t.sl_price,"SL")
                elif sl_hit: cp, reason = t.sl_price, "SL"
                elif tp_hit: cp, reason = t.tp_price, "TP"
                elif cur > min_profit and rsi_slope > 5 and \
                        bar["BodySize"] < bar["Range"] * 0.3:
                    cp, reason = bar["Close"], "MOMENTUM"
                elif cur > min_profit and t.sl_price > t.entry_price:
                    new_sl = t.entry_price - sp * self.sl_mult
                    if new_sl < t.sl_price:
                        t.sl_price = new_sl
                        t.be_activated = True
                        t.be_profit_trigger = cur / POINT
                        t.tp_price = bar["Low"] - atr_val * TP_TRAIL_ATR_MULT

            if cp is not None:
                to_close.append((t, cp, reason, bar_time))

        for t, cp, r, bt in to_close:
            self._close(t, cp, r, bt)

    def _close(self, t, cp, reason, ct):
        pnl_pts = (cp-t.entry_price)/POINT if t.direction=="LONG" \
                  else (t.entry_price-cp)/POINT
        pnl_usd = pnl_pts * POINT * CONTRACT_SIZE * t.lot_size
        t.exit_price=cp; t.close_time=ct
        t.pnl_usd=round(pnl_usd,2); t.pnl_points=round(pnl_pts,1)
        t.exit_reason=reason; t.is_open=False

        # Classify outcome
        if reason == "TP":
            t.outcome = "TP"
        elif reason == "MOMENTUM":
            t.outcome = "MOMENTUM"
        elif reason == "SL":
            t.outcome = "SL_BREAKEVEN" if t.be_activated else "SL_LOSS"
        else:
            t.outcome = "BACKTEST_END"

        self.balance   += pnl_usd
        self.equity     = self.balance
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(round(self.equity, 2))
        if pnl_usd < 0:
            self.daily_loss    += abs(pnl_usd)
            self.weekly_loss   += abs(pnl_usd)
            self.session_losses += 1
        else:
            self.session_losses = 0
        self.open_trades.remove(t)
        self.closed_trades.append(t)

    def is_trading_allowed(self):
        if self.daily_loss    >= MAX_DAILY_LOSS_USD:         return False
        if self.session_losses >= MAX_CONSEC_LOSSES:          return False
        if self.weekly_loss   >= WEEKLY_DD_USD:               return False
        if self.session_trade_count >= MAX_TRADES_PER_SESS:   return False
        if self.equity <= self.balance * 0.50:                return False
        return True


# ================================================================
#  CONDITION EXTRACTOR
# ================================================================

def extract_conditions(df_m15, df_h1, i, bar_time, direction):
    """
    Extract entry-time market conditions using 5-bar and 10-bar lookbacks.
    Returns a dict of condition values.
    """
    cond = {}

    # --- M15 indicators at entry ---
    sub = df_m15.iloc[max(0, i-25):i+1]
    if len(sub) < ROC_LONG + 2:
        return cond

    # Volume
    vol   = sub["Volume"].values
    vol_z = sub["VolZ"].iloc[-1]  if "VolZ"  in sub.columns else 0.0
    vol_mean5 = np.mean(vol[-ROC_SHORT:]) if len(vol) >= ROC_SHORT else vol.mean()
    vol_ratio = vol[-1] / vol_mean5 if vol_mean5 > 0 else 1.0

    # Volume 5-bar slope via linear regression
    if len(vol) >= ROC_SHORT:
        x = np.arange(ROC_SHORT)
        y = vol[-ROC_SHORT:]
        slope = np.polyfit(x, y, 1)[0]
        vol_5bar_slope = slope / (vol_mean5 + 1e-9)  # normalised
    else:
        vol_5bar_slope = 0.0

    cond["vol_z"]          = round(float(vol_z), 3)
    cond["vol_5bar_slope"] = round(float(vol_5bar_slope), 4)
    cond["vol_ratio"]      = round(float(vol_ratio), 3)

    # ATR
    atr = sub["ATR14"].values if "ATR14" in sub.columns else np.ones(len(sub)) * 10.0
    atr_now  = atr[-1]
    atr_mean20 = np.mean(atr[-20:]) if len(atr) >= 20 else atr.mean()
    atr_ratio  = atr_now / atr_mean20 if atr_mean20 > 0 else 1.0
    atr_5ago   = atr[-ROC_SHORT-1] if len(atr) > ROC_SHORT else atr[0]
    atr_5bar_roc = (atr_now - atr_5ago) / (atr_5ago + 1e-9)

    cond["atr_pts"]      = round(float(atr_now / POINT), 1)
    cond["atr_ratio"]    = round(float(atr_ratio), 3)
    cond["atr_5bar_roc"] = round(float(atr_5bar_roc), 4)

    # RSI (from M15)
    rsi = sub["RSI14"].values if "RSI14" in sub.columns else np.ones(len(sub)) * 50.0
    rsi_now = rsi[-1]
    rsi_3ago = rsi[-4] if len(rsi) > 3 else rsi[0]
    rsi_5ago = rsi[-6] if len(rsi) > 5 else rsi[0]

    cond["rsi_level"]      = round(float(rsi_now), 2)
    cond["rsi_3bar_slope"] = round(float(rsi_now - rsi_3ago), 3)
    cond["rsi_5bar_slope"] = round(float(rsi_now - rsi_5ago), 3)

    # EMA20 (from H1)
    h1_sub = df_h1[df_h1.index <= bar_time].tail(15)
    if len(h1_sub) >= 6 and "EMA20" in h1_sub.columns:
        ema = h1_sub["EMA20"].values
        ema_now  = ema[-1]
        ema_3ago = ema[-4] if len(ema) > 3 else ema[0]
        ema_5ago = ema[-6] if len(ema) > 5 else ema[0]
        cond["ema20_3bar_slope"] = round(float(ema_now - ema_3ago), 2)
        cond["ema20_5bar_slope"] = round(float(ema_now - ema_5ago), 2)
    else:
        cond["ema20_3bar_slope"] = 0.0
        cond["ema20_5bar_slope"] = 0.0

    # Price structure
    bar = sub.iloc[-1]
    rng = bar["Range"] if bar["Range"] > 0 else 1.0
    body_ratio = bar["BodySize"] / rng
    close_pos  = (bar["Close"] - bar["Low"]) / rng

    # 5-bar price momentum normalised by ATR
    close5ago = sub["Close"].iloc[-ROC_SHORT-1] if len(sub) > ROC_SHORT else sub["Close"].iloc[0]
    price_mom = (bar["Close"] - close5ago) / (atr_now + 1e-9)

    cond["body_ratio"]     = round(float(body_ratio), 3)
    cond["close_pos"]      = round(float(close_pos), 3)
    cond["price_5bar_mom"] = round(float(price_mom), 3)

    # Session
    hour = bar_time.hour
    cond["hour"]        = hour
    cond["day_of_week"] = bar_time.weekday()
    cond["session"]     = ("London" if LONDON_OPEN_H <= hour < LONDON_CLOSE_H
                           else "NY" if NY_OPEN_H <= hour < NY_CLOSE_H else "Other")

    return cond


# ================================================================
#  GRID RUNNER
# ================================================================

class GridRunner:
    def __init__(self):
        self.loader  = DataLoader(SYMBOL)
        self.scorer  = ScoringEngine()

    def run_all(self, df_h4, df_h1, df_m15) -> List[Dict]:
        """Run all 16 BE combinations and return results list."""
        results = []
        total = len(BE_GRID)
        print(f"\nRunning {total} breakeven combinations...")
        print(f"{'PT':>5} {'SL':>5} | {'Trades':>7} {'WR':>7} {'Profit':>10} "
              f"{'MaxDD':>7} {'PF':>6} | {'TP%':>6} {'SL_L%':>6} {'SL_BE%':>7} {'MOM%':>6}")
        print("-" * 85)

        for idx, (pt_mult, sl_mult) in enumerate(BE_GRID):
            sim = ParametrizedSimulator(pt_mult, sl_mult)
            trades = self._run_single(sim, df_h4, df_h1, df_m15)

            r = self._analyse(sim, trades, pt_mult, sl_mult)
            results.append(r)

            print(f"{pt_mult:>5.1f} {sl_mult:>5.0f} | "
                  f"{r['total_trades']:>7} {r['win_rate']*100:>6.1f}% "
                  f"${r['net_profit']:>9,.0f} {r['max_dd_pct']:>6.1f}% "
                  f"{r['profit_factor']:>6.2f} | "
                  f"{r['pct_tp']:>5.1f}% {r['pct_sl_loss']:>5.1f}% "
                  f"{r['pct_sl_be']:>6.1f}% {r['pct_mom']:>5.1f}%")

        return results

    def _run_single(self, sim, df_h4, df_h1, df_m15):
        sim.reset()
        daily_reset_date = None
        weekly_reset_key = None
        df_h4=df_h4.copy(); df_h1=df_h1.copy(); df_m15=df_m15.copy()

        def _norm(df):
            m = {}
            for c in df.columns:
                cl = c.lower()
                if   cl == "open":                    m[c] = "Open"
                elif cl == "high":                    m[c] = "High"
                elif cl == "low":                     m[c] = "Low"
                elif cl == "close":                   m[c] = "Close"
                elif cl in ("volume","tick_volume"):  m[c] = "Volume"
                elif cl == "spread":                  m[c] = "Spread"
            return df.rename(columns=m)

        df_h4=_norm(df_h4); df_h1=_norm(df_h1); df_m15=_norm(df_m15)

        # Pre-compute all indicators
        df_h4["ATR14"]       = Indicators.atr(df_h4["High"],df_h4["Low"],df_h4["Close"],14)
        df_h1["RSI14"]       = Indicators.rsi(df_h1["Close"],14)
        df_h1["EMA20"]       = Indicators.ema(df_h1["Close"],20)
        df_h1["EMA20slope"]  = df_h1["EMA20"].diff(3)
        df_m15["RSI14"]      = Indicators.rsi(df_m15["Close"],14)
        df_m15["RSIslope"]   = Indicators.rsi_slope(df_m15["RSI14"],3)
        df_m15["ATR14"]      = Indicators.atr(df_m15["High"],df_m15["Low"],df_m15["Close"],14)
        df_m15["VolZ"]       = Indicators.volume_zscore(df_m15["Volume"],20)
        stk, std             = Indicators.stoch_rsi(df_m15["Close"])
        df_m15["StochK"]     = stk; df_m15["StochD"] = std

        # Expose rsi_slope static method reference
        rsi_slope_fn = Indicators.rsi_slope

        for i in range(50, len(df_m15)):
            bar_time = df_m15.index[i]; bar = df_m15.iloc[i]

            bar_date = bar_time.date()
            if bar_date != daily_reset_date:
                sim.daily_loss     = 0.0
                sim.session_losses = 0
                daily_reset_date   = bar_date

            week_key = f"{bar_time.year}_{bar_time.isocalendar()[1]}"
            if week_key != weekly_reset_key:
                sim.weekly_loss  = 0.0
                weekly_reset_key = week_key

            hour = bar_time.hour
            in_london = (LONDON_OPEN_H <= hour < LONDON_CLOSE_H)
            in_ny     = (NY_OPEN_H     <= hour < NY_CLOSE_H)
            in_session= in_london or in_ny

            if in_london:   sess_key = f"london_{bar_date}"
            elif in_ny:     sess_key = f"ny_{bar_date}"
            else:           sess_key = None

            if sess_key and sess_key != sim.current_session_key:
                sim.current_session_key = sess_key
                sim.session_trade_count = 0

            atr_val = bar["ATR14"]    if not pd.isna(bar["ATR14"])    else 10.0
            rsi_val = bar["RSI14"]    if not pd.isna(bar["RSI14"])    else 50.0
            rsi_slp = bar["RSIslope"] if not pd.isna(bar["RSIslope"]) else 0.0
            vol_z   = bar["VolZ"]     if not pd.isna(bar["VolZ"])     else 0.0

            sim.update_trades(bar, bar_time, atr_val, rsi_slp)

            if not in_session: continue
            if not sim.is_trading_allowed(): continue
            if len(sim.open_trades) >= MAX_CONCURRENT_TRADES: continue

            spread_pts = bar.get("Spread", 25)
            if pd.isna(spread_pts): spread_pts = 25
            if spread_pts > MAX_SPREAD_PTS: continue

            h4_bars = df_h4[df_h4.index <= bar_time].tail(20)
            h1_bars = df_h1[df_h1.index <= bar_time].tail(20)
            if len(h4_bars) < 10 or len(h1_bars) < 5: continue

            h1_last  = h1_bars.iloc[-1]
            h4_trend_up = h4_bars["Close"].iloc[-1] > h4_bars["Close"].iloc[-5]

            h1_rsi   = h1_last.get("RSI14", 50.0)
            h1_slope = h1_last.get("EMA20slope", 0.0)
            h1_bull  = (h1_rsi > 45) or (h1_slope > 0)
            h1_bear  = (h1_rsi < 55) or (h1_slope < 0)

            stk_k  = bar.get("StochK", 50.0)
            stk_d  = bar.get("StochD", 50.0)
            prev   = df_m15.iloc[i-1]
            prev_k = prev.get("StochK", 50.0)
            prev_d = prev.get("StochD", 50.0)
            bull_cross = (stk_k > stk_d and prev_k <= prev_d and stk_k < 85)
            bear_cross = (stk_k < stk_d and prev_k >= prev_d and stk_k > 15)

            long_signal  = h4_trend_up       and h1_bull and bull_cross
            short_signal = (not h4_trend_up) and h1_bear and bear_cross
            if not (long_signal or short_signal): continue

            direction  = "LONG" if long_signal else "SHORT"
            open_dirs  = [t.direction for t in sim.open_trades]
            if open_dirs.count(direction) >= 1: continue

            if vol_z < FILTER_VOL_MIN_Z: continue

            conditions_dict = {
                "elliott_bias":  h4_trend_up if direction=="LONG" else not h4_trend_up,
                "smc_structure": h4_trend_up if direction=="LONG" else not h4_trend_up,
                "fvg_confluence":False,
                "h1_momentum":   h1_bull if direction=="LONG" else h1_bear,
                "m15_bos":       bull_cross if direction=="LONG" else bear_cross,
                "volume_confirm":vol_z > 0.5,
            }
            pw, raw_score, grade = self.scorer.calculate(conditions_dict)
            if grade == "SKIP": continue

            atr_pts = atr_val / POINT
            sl_pts  = int(atr_pts * SL_ATR_MULTIPLIER)
            sl_pts  = max(int(atr_pts*SL_ATR_MIN_MULT),
                          min(int(atr_pts*SL_ATR_MAX_MULT), sl_pts))
            sl_pts  = max(sl_pts, 100)
            tp_pts  = int(sl_pts * TP_RR_RATIO)
            entry   = bar["Close"]
            sl = entry - sl_pts*POINT if direction=="LONG" else entry + sl_pts*POINT
            tp = entry + tp_pts*POINT if direction=="LONG" else entry - tp_pts*POINT

            lot = sim.calc_lot_size(sl_pts, pw, self.scorer)
            if lot <= 0: continue

            if bar_time.weekday() == 0:
                lot = max(LOT_MIN, round(
                    round(lot*FILTER_MONDAY_LOT/LOT_STEP)*LOT_STEP, 2))

            # Extract per-trade conditions for condition analysis
            entry_cond = extract_conditions(df_m15, df_h1, i, bar_time, direction)
            entry_cond["direction"] = direction

            sim.open_trade(direction, entry, sl, tp, lot, spread_pts,
                          raw_score, grade, pw, bar_time, entry_cond)

        # Close remaining
        if sim.open_trades:
            last_bar = df_m15.iloc[-1]
            for t in list(sim.open_trades):
                sim._close(t, last_bar["Close"], "BACKTEST_END", df_m15.index[-1])

        return sim.closed_trades

    def _analyse(self, sim, trades, pt_mult, sl_mult) -> Dict:
        """Compute comprehensive metrics for one BE combination."""
        pnls = [t.pnl_usd for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        tp_trades   = [t for t in trades if t.outcome == "TP"]
        sl_loss     = [t for t in trades if t.outcome == "SL_LOSS"]
        sl_be       = [t for t in trades if t.outcome == "SL_BREAKEVEN"]
        mom_trades  = [t for t in trades if t.outcome == "MOMENTUM"]
        be_activated= [t for t in trades if t.be_activated]

        n = len(trades) or 1
        eq = sim.equity_curve; peak = eq[0]; max_dd = 0.0
        for e in eq: peak = max(peak,e); max_dd = max(max_dd, peak-e)

        # BE efficiency: of trades that activated BE, what % hit TP?
        be_n = len(be_activated)
        be_tp = len([t for t in be_activated if t.outcome=="TP"])
        be_efficiency = be_tp / be_n if be_n > 0 else 0.0

        # Avg profit at BE trigger point
        be_trigger_pts = [t.be_profit_trigger for t in be_activated if t.be_profit_trigger > 0]

        return {
            "pt_mult":       pt_mult,
            "sl_mult":       sl_mult,
            "label":         f"PT{pt_mult:.1f}_SL{sl_mult}",
            "total_trades":  len(trades),
            "win_rate":      len(wins)/n,
            "net_profit":    sum(pnls),
            "max_dd_pct":    (max_dd/ACCOUNT_BALANCE)*100,
            "profit_factor": sum(wins)/abs(sum(losses)) if losses else 999.0,
            "avg_win":       np.mean(wins) if wins else 0.0,
            "avg_loss":      np.mean(losses) if losses else 0.0,
            "sharpe":        (np.mean(pnls)/np.std(pnls)*np.sqrt(252)
                              if len(pnls)>1 and np.std(pnls)>0 else 0.0),
            "n_tp":          len(tp_trades),
            "n_sl_loss":     len(sl_loss),
            "n_sl_be":       len(sl_be),
            "n_mom":         len(mom_trades),
            "pct_tp":        len(tp_trades)/n*100,
            "pct_sl_loss":   len(sl_loss)/n*100,
            "pct_sl_be":     len(sl_be)/n*100,
            "pct_mom":       len(mom_trades)/n*100,
            "be_activated":  be_n,
            "be_efficiency": be_efficiency,
            "avg_be_trigger_pts": np.mean(be_trigger_pts) if be_trigger_pts else 0.0,
            "trades":        trades,   # full trade list for condition analysis
        }


# ================================================================
#  CONDITION ANALYSER
# ================================================================

def analyse_conditions(results: List[Dict]) -> pd.DataFrame:
    """
    Aggregate all trades across all BE combos and compute mean
    conditions by outcome type.
    """
    all_rows = []
    for r in results:
        for t in r["trades"]:
            row = {
                "pt_mult":       r["pt_mult"],
                "sl_mult":       r["sl_mult"],
                "trade_id":      t.trade_id,
                "direction":     t.direction,
                "open_time":     t.open_time,
                "pnl_usd":       t.pnl_usd,
                "outcome":       t.outcome,
                "be_activated":  t.be_activated,
                "be_trigger_pts":t.be_profit_trigger,
                "spread":        t.spread_at_entry,
                # conditions
                "vol_z":         t.vol_z,
                "vol_5bar_slope":t.vol_5bar_slope,
                "vol_ratio":     t.vol_ratio,
                "atr_pts":       t.atr_pts,
                "atr_ratio":     t.atr_ratio,
                "atr_5bar_roc":  t.atr_5bar_roc,
                "rsi_level":     t.rsi_level,
                "rsi_3bar_slope":t.rsi_3bar_slope,
                "rsi_5bar_slope":t.rsi_5bar_slope,
                "ema20_3bar_slope":t.ema20_3bar_slope,
                "ema20_5bar_slope":t.ema20_5bar_slope,
                "body_ratio":    t.body_ratio,
                "close_pos":     t.close_pos,
                "price_5bar_mom":t.price_5bar_mom,
                "session":       t.session,
                "hour":          t.hour,
                "day_of_week":   t.day_of_week,
            }
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def print_condition_comparison(df: pd.DataFrame):
    """Print mean conditions by outcome type across all combos."""
    outcomes = ["TP", "SL_LOSS", "SL_BREAKEVEN", "MOMENTUM"]
    numeric_cols = [
        "vol_z", "vol_5bar_slope", "vol_ratio",
        "atr_pts", "atr_ratio", "atr_5bar_roc",
        "rsi_level", "rsi_3bar_slope", "rsi_5bar_slope",
        "ema20_3bar_slope", "ema20_5bar_slope",
        "body_ratio", "close_pos", "price_5bar_mom",
    ]
    labels = {
        "vol_z":            "Volume Z-score",
        "vol_5bar_slope":   "Volume 5-bar slope",
        "vol_ratio":        "Volume / 5-bar mean",
        "atr_pts":          "ATR (points)",
        "atr_ratio":        "ATR / 20-bar avg",
        "atr_5bar_roc":     "ATR 5-bar ROC",
        "rsi_level":        "RSI level",
        "rsi_3bar_slope":   "RSI 3-bar slope",
        "rsi_5bar_slope":   "RSI 5-bar slope",
        "ema20_3bar_slope":  "EMA20 H1 3-bar slope",
        "ema20_5bar_slope":  "EMA20 H1 5-bar slope",
        "body_ratio":       "Candle body ratio",
        "close_pos":        "Close position (0=low,1=high)",
        "price_5bar_mom":   "Price 5-bar momentum / ATR",
    }

    print("\n" + "="*85)
    print("  MARKET CONDITIONS BY OUTCOME  (mean values at entry, all BE combos pooled)")
    print("  Lookback: 5 M15 bars = 75 min (1 trading hour)")
    print("="*85)

    # Count per outcome
    counts = {o: len(df[df["outcome"]==o]) for o in outcomes}
    header = f"  {'Condition':<32}"
    for o in outcomes:
        header += f" {o:>12}"
    print(header)
    print(f"  {'Count':<32}" + "".join(f" {counts[o]:>12}" for o in outcomes))
    print("-"*85)

    for col in numeric_cols:
        line = f"  {labels.get(col, col):<32}"
        for o in outcomes:
            subset = df[df["outcome"]==o][col].dropna()
            val = subset.mean() if len(subset) > 0 else float("nan")
            line += f" {val:>12.3f}"
        print(line)

    print("="*85)

    # Session breakdown
    print("\n  SESSION BREAKDOWN BY OUTCOME:")
    print(f"  {'Session':<15}" + "".join(f" {o:>12}" for o in outcomes))
    print("-"*45)
    for sess in ["London", "NY"]:
        line = f"  {sess:<15}"
        for o in outcomes:
            n = len(df[(df["outcome"]==o) & (df["session"]==sess)])
            line += f" {n:>12}"
        print(line)

    # Day of week breakdown
    print("\n  DAY OF WEEK BREAKDOWN (SL_LOSS count -- higher = worse day):")
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    sl_df = df[df["outcome"]=="SL_LOSS"]
    for d_idx, day in enumerate(days):
        n = len(sl_df[sl_df["day_of_week"]==d_idx])
        bar = "█" * n
        print(f"  {day:<12}: {n:>4}  {bar}")


def print_grid_summary(results: List[Dict]):
    """Print ranked summary of all 16 BE combinations."""
    print("\n" + "="*85)
    print("  BREAKEVEN GRID -- RANKED BY PROFIT FACTOR")
    print("="*85)
    ranked = sorted(results, key=lambda r: r["profit_factor"], reverse=True)
    print(f"  {'Combo':<14} {'Trades':>7} {'WR':>7} {'Profit':>10} "
          f"{'DD%':>6} {'PF':>6} {'AvgWin':>8} {'AvgLoss':>9} "
          f"{'BE%':>6} {'BEeff':>7}")
    print("-"*85)
    for r in ranked:
        print(f"  {r['label']:<14} {r['total_trades']:>7} "
              f"{r['win_rate']*100:>6.1f}% ${r['net_profit']:>9,.0f} "
              f"{r['max_dd_pct']:>5.1f}% {r['profit_factor']:>6.2f} "
              f"${r['avg_win']:>7.2f} ${r['avg_loss']:>8.2f} "
              f"{r['be_activated']/max(r['total_trades'],1)*100:>5.1f}% "
              f"{r['be_efficiency']*100:>6.1f}%")


def print_recommendation(results: List[Dict]):
    """Score each combo on a weighted basis and recommend the best."""
    print("\n" + "="*65)
    print("  RECOMMENDATION")
    print("="*65)

    scored = []
    for r in results:
        if r["total_trades"] < 100: continue
        # Weighted score: profit is king, but penalise DD and poor R:R
        rr = abs(r["avg_win"]) / abs(r["avg_loss"]) if r["avg_loss"] != 0 else 0
        score = (r["profit_factor"] * 0.35 +
                 r["win_rate"] * 3.0 +
                 rr * 0.25 -
                 r["max_dd_pct"] * 0.02 +
                 r["be_efficiency"] * 0.15)
        scored.append((score, r))

    scored.sort(reverse=True)
    if not scored:
        print("  Not enough data to recommend.")
        return

    best_score, best = scored[0]
    print(f"  Best combo   : PT{best['pt_mult']:.1f}x spread trigger -> "
          f"SL lock at entry + {best['sl_mult']}x spread")
    print(f"  Win rate     : {best['win_rate']*100:.1f}%")
    print(f"  Net profit   : ${best['net_profit']:,.2f}")
    print(f"  Max DD       : {best['max_dd_pct']:.1f}%")
    print(f"  Profit factor: {best['profit_factor']:.2f}")
    print(f"  BE efficiency: {best['be_efficiency']*100:.1f}% "
          f"(of BE-activated trades, this % hit TP)")
    print(f"\n  Logic: once {best['pt_mult']:.1f}x spread profit is reached,")
    print(f"  locking SL to entry+{best['sl_mult']}x spread gives the best")
    print(f"  balance of protecting profit while allowing price room to run.")
    print(f"\n  Runner-up    : PT{scored[1][1]['pt_mult']:.1f}x -> "
          f"SL+{scored[1][1]['sl_mult']}x  "
          f"(PF:{scored[1][1]['profit_factor']:.2f}, "
          f"DD:{scored[1][1]['max_dd_pct']:.1f}%)")
    print("="*65)


# ================================================================
#  ML FEATURE EXPORT (XGBoost-ready)
# ================================================================

def export_ml_features(df: pd.DataFrame, output_dir: str):
    """
    Export a clean feature matrix for XGBoost / ML training.
    Target: binary win (TP or MOMENTUM) vs loss (SL_LOSS).
    SL_BREAKEVEN is excluded as ambiguous.
    """
    ml_df = df[df["outcome"].isin(["TP","MOMENTUM","SL_LOSS"])].copy()
    ml_df["target"] = (ml_df["outcome"].isin(["TP","MOMENTUM"])).astype(int)

    feature_cols = [
        "vol_z","vol_5bar_slope","vol_ratio",
        "atr_pts","atr_ratio","atr_5bar_roc",
        "rsi_level","rsi_3bar_slope","rsi_5bar_slope",
        "ema20_3bar_slope","ema20_5bar_slope",
        "body_ratio","close_pos","price_5bar_mom",
        "spread","hour","day_of_week",
        "target"
    ]
    ml_df = ml_df[[c for c in feature_cols if c in ml_df.columns]]

    path = os.path.join(output_dir, "be_ml_features.csv")
    ml_df.to_csv(path, index=False)
    print(f"\n  ML feature matrix saved: {path}")
    print(f"  Rows: {len(ml_df)} | "
          f"Wins: {ml_df['target'].sum()} | "
          f"Losses: {(ml_df['target']==0).sum()}")

    # Optional: run XGBoost if installed
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score

        feat_cols = [c for c in feature_cols if c != "target"]
        X = ml_df[feat_cols].fillna(0).values
        y = ml_df["target"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:,1]
        acc   = accuracy_score(y_te, preds)
        auc   = roc_auc_score(y_te, proba)

        print(f"\n  XGBoost Results:")
        print(f"    Accuracy : {acc*100:.1f}%")
        print(f"    AUC-ROC  : {auc:.3f}")

        # Feature importance
        imp = model.feature_importances_
        fi  = sorted(zip(feat_cols, imp), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 most predictive features:")
        for name, score in fi[:10]:
            bar = "=" * int(score * 200)
            print(f"    {name:<25} {score:.4f}  {bar}")

        fi_path = os.path.join(output_dir, "be_xgb_feature_importance.csv")
        pd.DataFrame(fi, columns=["feature","importance"]).to_csv(fi_path, index=False)
        print(f"  Feature importance saved: {fi_path}")

    except ImportError:
        print("\n  XGBoost not installed. To use ML analysis:")
        print("  pip install xgboost scikit-learn")
        print("  Then re-run this script for automatic feature importance.")


# ================================================================
#  SAVE CSVs
# ================================================================

def save_csvs(results: List[Dict], df_conditions: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Grid results
    grid_path = os.path.join(output_dir, "be_grid_results.csv")
    grid_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "trades"}
        grid_rows.append(row)
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)
    print(f"  Grid results saved : {grid_path}")

    # Trade conditions
    cond_path = os.path.join(output_dir, "be_trade_conditions.csv")
    df_conditions.to_csv(cond_path, index=False)
    print(f"  Trade conditions   : {cond_path}")


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("="*65)
    print("  GoldScalperBot -- Breakeven Grid Analyser")
    print("="*65)
    print(f"  Symbol   : {SYMBOL}")
    print(f"  Period   : {INSAMPLE_START.date()} to {INSAMPLE_END.date()}")
    print(f"  BE Grid  : {len(BE_GRID)} combinations  "
          f"(4 profit triggers x 4 SL locks)")
    print(f"  ROC window: {ROC_SHORT} bars = {ROC_SHORT*15} min (M15)")
    print("="*65)

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}"); exit(1)
    a = mt5.account_info()
    if a is None:
        print("MT5 not logged in"); mt5.shutdown(); exit(1)
    print(f"MT5 connected -- {a.login}  Balance: ${a.balance:,.2f}\n")

    # Load data once
    loader = DataLoader(SYMBOL)
    df_h4  = loader.load("H4",  INSAMPLE_START, INSAMPLE_END)
    df_h1  = loader.load("H1",  INSAMPLE_START, INSAMPLE_END)
    df_m15 = loader.load("M15", INSAMPLE_START, INSAMPLE_END)
    print(f"\nH4:{len(df_h4)} | H1:{len(df_h1)} | M15:{len(df_m15)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run all 16 combinations
    t0 = time.time()
    runner  = GridRunner()
    results = runner.run_all(df_h4, df_h1, df_m15)
    elapsed = time.time() - t0
    print(f"\nGrid completed in {elapsed:.1f}s")

    # Full ranked summary
    print_grid_summary(results)

    # Condition analysis
    df_cond = analyse_conditions(results)
    print_condition_comparison(df_cond)

    # Recommendation
    print_recommendation(results)

    # Save all CSVs
    print("\n  Saving outputs...")
    save_csvs(results, df_cond, OUTPUT_DIR)
    export_ml_features(df_cond, OUTPUT_DIR)

    mt5.shutdown()
    print("\nDone. All results in analysis_results/")
