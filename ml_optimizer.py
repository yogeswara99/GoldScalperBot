#!/usr/bin/env python3
# ================================================================
#  ml_optimizer.py  --  GoldScalperBot  XAUUSD...
#
#  PHASE 1: Feature Engineering (50+ features per trade)
#  PHASE 2: XGBoost classifier + SHAP feature importance
#  PHASE 3: Optuna Bayesian optimization over full parameter space
#  PHASE 4: Write optimal config to config_v1.5.py
#
#  Features drawn from:
#    - Academic research: Jabeur et al. 2024 (gold XGBoost+SHAP)
#    - UCLA thesis: Bayesian parameter optimization +15-25% accuracy
#    - Our own condition analysis (breakeven_analyser.py results)
#
#  Parameter search space:
#    SL/TP multipliers, BE parameters, risk sizing, concurrent trades,
#    circuit breakers, RSI/EMA/volume filters, session windows,
#    StochRSI thresholds, day-of-week adjustments
#
#  REQUIRES:
#    pip install xgboost scikit-learn shap optuna
#    MT5 open and connected
#
#  USAGE: python ml_optimizer.py
#  Runtime: ~15-30 mins (200 Optuna trials)
# ================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import csv, os, time, warnings
warnings.filterwarnings("ignore")

# ---- Check dependencies before loading ----------------------
def check_deps():
    missing = []
    for pkg in ["xgboost", "sklearn", "shap", "optuna"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)
    if missing:
        print("MISSING PACKAGES. Run this first:")
        print(f"  pip install {' '.join(missing)}")
        exit(1)

check_deps()

import xgboost as xgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

from config import *
from backtest_engine import (
    DataLoader, Indicators, ScoringEngine,
    BacktestTrade, BacktestResult, ResultAnalyser,
    INSAMPLE_START, INSAMPLE_END, SLIPPAGE_POINTS
)

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
#  FEATURE ENGINEERING
# ================================================================

def compute_features(df_m15: pd.DataFrame,
                     df_h1:  pd.DataFrame,
                     df_h4:  pd.DataFrame,
                     i: int,
                     bar_time) -> Dict:
    """
    Compute 50+ features at entry bar i on M15.
    Returns dict of floats, ready for ML.
    """
    f = {}

    # Slice windows
    m15 = df_m15.iloc[max(0, i-30):i+1]
    h1  = df_h1[df_h1.index <= bar_time].tail(20)
    h4  = df_h4[df_h4.index <= bar_time].tail(20)

    if len(m15) < 15 or len(h1) < 5 or len(h4) < 5:
        return {}

    bar = m15.iloc[-1]

    # ---- 1. PRICE ACTION ----------------------------------------
    rng = bar["Range"] if bar["Range"] > 0 else 1.0
    f["body_ratio"]       = bar["BodySize"] / rng
    f["close_pos"]        = (bar["Close"] - bar["Low"]) / rng
    f["is_bullish"]       = 1.0 if bar["Close"] > bar["Open"] else 0.0
    f["gap_from_prev"]    = (bar["Open"] - m15.iloc[-2]["Close"]) / (bar["ATR14"] + 1e-9) if "ATR14" in m15.columns else 0.0

    # Consecutive same-direction candles
    consec = 0
    for k in range(len(m15)-2, max(0, len(m15)-8), -1):
        if (m15.iloc[k]["Close"] > m15.iloc[k]["Open"]) == (bar["Close"] > bar["Open"]):
            consec += 1
        else:
            break
    f["consec_candles"]   = float(consec)

    # ---- 2. ATR / VOLATILITY -----------------------------------
    atr = m15["ATR14"].values if "ATR14" in m15.columns else np.ones(len(m15)) * 10.0
    atr_now   = atr[-1]
    atr_mean20= np.mean(atr[-20:]) if len(atr) >= 20 else np.mean(atr)
    atr_3ago  = atr[-4]  if len(atr) > 3  else atr[0]
    atr_5ago  = atr[-6]  if len(atr) > 5  else atr[0]
    atr_10ago = atr[-11] if len(atr) > 10 else atr[0]

    f["atr_pts"]         = atr_now / POINT
    f["atr_ratio"]       = atr_now / (atr_mean20 + 1e-9)
    f["atr_3bar_roc"]    = (atr_now - atr_3ago)  / (atr_3ago  + 1e-9)
    f["atr_5bar_roc"]    = (atr_now - atr_5ago)  / (atr_5ago  + 1e-9)
    f["atr_10bar_roc"]   = (atr_now - atr_10ago) / (atr_10ago + 1e-9)
    f["atr_regime"]      = (1.0 if atr_now > atr_mean20 * 1.3 else
                            -1.0 if atr_now < atr_mean20 * 0.7 else 0.0)

    # ---- 3. RSI (M15) ------------------------------------------
    rsi = m15["RSI14"].values if "RSI14" in m15.columns else np.ones(len(m15)) * 50.0
    rsi_now   = rsi[-1]
    rsi_3ago  = rsi[-4]  if len(rsi) > 3  else rsi[0]
    rsi_5ago  = rsi[-6]  if len(rsi) > 5  else rsi[0]
    rsi_10ago = rsi[-11] if len(rsi) > 10 else rsi[0]

    f["rsi"]             = rsi_now
    f["rsi_vs_50"]       = rsi_now - 50.0
    f["rsi_3bar_slope"]  = rsi_now - rsi_3ago
    f["rsi_5bar_slope"]  = rsi_now - rsi_5ago
    f["rsi_10bar_slope"] = rsi_now - rsi_10ago
    f["rsi_overbought"]  = 1.0 if rsi_now > 70 else 0.0
    f["rsi_oversold"]    = 1.0 if rsi_now < 30 else 0.0

    # ---- 4. VOLUME (M15) ---------------------------------------
    vol = m15["Volume"].values
    vol_now   = vol[-1]
    vol_z     = m15["VolZ"].iloc[-1] if "VolZ" in m15.columns else 0.0
    vol_mean5 = np.mean(vol[-5:]) if len(vol) >= 5 else np.mean(vol)
    vol_mean20= np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)

    # Volume 5-bar linear slope (normalised)
    if len(vol) >= 5:
        x = np.arange(5); y = vol[-5:]
        vol_slope5 = np.polyfit(x, y, 1)[0] / (vol_mean5 + 1e-9)
    else:
        vol_slope5 = 0.0

    f["vol_z"]           = float(vol_z)
    f["vol_ratio"]       = vol_now / (vol_mean5 + 1e-9)
    f["vol_slope_5bar"]  = vol_slope5
    f["vol_vs_mean20"]   = (vol_now - vol_mean20) / (vol_mean20 + 1e-9)

    # ---- 5. STOCHRSI (M15) -------------------------------------
    stk_k = bar.get("StochK", 50.0) if hasattr(bar, "get") else 50.0
    stk_d = bar.get("StochD", 50.0) if hasattr(bar, "get") else 50.0
    if pd.isna(stk_k): stk_k = 50.0
    if pd.isna(stk_d): stk_d = 50.0

    f["stoch_k"]         = stk_k
    f["stoch_d"]         = stk_d
    f["stoch_kd_spread"] = stk_k - stk_d
    f["stoch_overbought"]= 1.0 if stk_k > 80 else 0.0
    f["stoch_oversold"]  = 1.0 if stk_k < 20 else 0.0

    # ---- 6. EMA (H1) -------------------------------------------
    if "EMA20" in h1.columns and len(h1) >= 6:
        ema = h1["EMA20"].values
        ema_now   = ema[-1]
        ema_3ago  = ema[-4] if len(ema) > 3 else ema[0]
        ema_5ago  = ema[-6] if len(ema) > 5 else ema[0]
        h1_close  = h1["Close"].iloc[-1]
        h1_rsi    = h1["RSI14"].iloc[-1] if "RSI14" in h1.columns else 50.0

        f["h1_ema_slope3"]   = ema_now - ema_3ago
        f["h1_ema_slope5"]   = ema_now - ema_5ago
        f["h1_price_vs_ema"] = (h1_close - ema_now) / (atr_now + 1e-9)
        f["h1_rsi"]          = float(h1_rsi) if not pd.isna(h1_rsi) else 50.0
    else:
        f["h1_ema_slope3"]   = 0.0
        f["h1_ema_slope5"]   = 0.0
        f["h1_price_vs_ema"] = 0.0
        f["h1_rsi"]          = 50.0

    # ---- 7. MACD (M15) -----------------------------------------
    close_m15 = m15["Close"]
    ema12 = close_m15.ewm(span=12, adjust=False).mean()
    ema26 = close_m15.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - macd_signal

    f["macd_line"]        = float(macd_line.iloc[-1]) / (atr_now + 1e-9)
    f["macd_hist"]        = float(macd_hist.iloc[-1]) / (atr_now + 1e-9)
    f["macd_hist_slope"]  = float((macd_hist.iloc[-1] - macd_hist.iloc[-2])) / (atr_now + 1e-9) if len(macd_hist) > 1 else 0.0

    # ---- 8. BOLLINGER BANDS (M15) ------------------------------
    bb_period = 20
    if len(close_m15) >= bb_period:
        bb_mid   = close_m15.rolling(bb_period).mean().iloc[-1]
        bb_std   = close_m15.rolling(bb_period).std().iloc[-1]
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)
        bb_pct_b = (bar["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    else:
        bb_width = 0.04; bb_pct_b = 0.5

    f["bb_width"]         = bb_width
    f["bb_pct_b"]         = bb_pct_b
    f["bb_squeeze"]       = 1.0 if bb_width < 0.02 else 0.0

    # ---- 9. ADX (M15) ------------------------------------------
    if len(m15) >= 14:
        high = m15["High"].values; low = m15["Low"].values; close = m15["Close"].values
        tr   = np.maximum(high[1:]-low[1:],
               np.maximum(abs(high[1:]-close[:-1]), abs(low[1:]-close[:-1])))
        dm_p = np.where((high[1:]-high[:-1]) > (low[:-1]-low[1:]),
                         np.maximum(high[1:]-high[:-1], 0.0), 0.0)
        dm_n = np.where((low[:-1]-low[1:]) > (high[1:]-high[:-1]),
                         np.maximum(low[:-1]-low[1:], 0.0), 0.0)

        # Smooth over 14 bars
        p = 14
        if len(tr) >= p:
            atr14 = pd.Series(tr).ewm(com=p-1, adjust=False).mean().iloc[-1]
            dip   = pd.Series(dm_p).ewm(com=p-1, adjust=False).mean().iloc[-1]
            din   = pd.Series(dm_n).ewm(com=p-1, adjust=False).mean().iloc[-1]
            di_p  = 100 * dip / (atr14 + 1e-9)
            di_n  = 100 * din / (atr14 + 1e-9)
            dx    = 100 * abs(di_p - di_n) / (di_p + di_n + 1e-9)
            f["adx"]          = dx
            f["di_plus"]      = di_p
            f["di_minus"]     = di_n
            f["di_spread"]    = di_p - di_n
        else:
            f["adx"] = 25.0; f["di_plus"] = 25.0
            f["di_minus"] = 25.0; f["di_spread"] = 0.0
    else:
        f["adx"] = 25.0; f["di_plus"] = 25.0
        f["di_minus"] = 25.0; f["di_spread"] = 0.0

    # ---- 10. H4 CONTEXT ----------------------------------------
    if len(h4) >= 10:
        h4_close = h4["Close"].values
        f["h4_slope5"]   = (h4_close[-1] - h4_close[-6])  / (atr_now + 1e-9) if len(h4_close) > 5  else 0.0
        f["h4_slope10"]  = (h4_close[-1] - h4_close[-11]) / (atr_now + 1e-9) if len(h4_close) > 10 else 0.0
        h4_ema20 = pd.Series(h4_close).ewm(span=20, adjust=False).mean().iloc[-1]
        f["h4_vs_ema20"] = (h4_close[-1] - h4_ema20) / (atr_now + 1e-9)
    else:
        f["h4_slope5"] = 0.0; f["h4_slope10"] = 0.0; f["h4_vs_ema20"] = 0.0

    # ---- 11. PRICE MOMENTUM (M15) ------------------------------
    close_arr = m15["Close"].values
    f["price_mom_3bar"]  = (close_arr[-1] - close_arr[-4])  / (atr_now + 1e-9) if len(close_arr) > 3  else 0.0
    f["price_mom_5bar"]  = (close_arr[-1] - close_arr[-6])  / (atr_now + 1e-9) if len(close_arr) > 5  else 0.0
    f["price_mom_10bar"] = (close_arr[-1] - close_arr[-11]) / (atr_now + 1e-9) if len(close_arr) > 10 else 0.0

    # ---- 12. SESSION / TIME ------------------------------------
    hour = bar_time.hour
    f["hour"]            = float(hour)
    f["day_of_week"]     = float(bar_time.weekday())
    f["is_london"]       = 1.0 if LONDON_OPEN_H <= hour < LONDON_CLOSE_H else 0.0
    f["is_ny"]           = 1.0 if NY_OPEN_H     <= hour < NY_CLOSE_H     else 0.0
    f["is_overlap"]      = 1.0 if LONDON_OPEN_H <= hour < LONDON_CLOSE_H and \
                                   NY_OPEN_H <= hour else 0.0
    # Session age: bars since session open
    if f["is_london"]:
        session_bars = (hour - LONDON_OPEN_H) * 4 + bar_time.minute // 15
    elif f["is_ny"]:
        session_bars = (hour - NY_OPEN_H) * 4 + bar_time.minute // 15
    else:
        session_bars = 0
    f["session_age_bars"] = float(session_bars)

    # ---- 13. SPREAD / MICROSTRUCTURE ---------------------------
    spread = bar.get("Spread", 20) if hasattr(bar, "get") else 20
    if pd.isna(spread): spread = 20
    f["spread_pts"]      = float(spread)
    f["spread_atr_ratio"]= spread / (f["atr_pts"] + 1e-9)

    return f


# ================================================================
#  TRADE GENERATOR (uses v1.3 loose params to maximise sample size)
# ================================================================

class TradeGenerator:
    """
    Runs backtest with v1.3 parameters (1,800+ trades)
    and attaches full feature vectors to each trade.
    """
    def __init__(self):
        self.scorer = ScoringEngine()

    def generate(self, df_h4, df_h1, df_m15) -> pd.DataFrame:
        print("\nGenerating trades with feature vectors...")
        rows = []

        sim_balance = ACCOUNT_BALANCE
        sim_equity  = ACCOUNT_BALANCE

        # State
        open_trades  = []
        daily_loss   = 0.0
        weekly_loss  = 0.0
        session_losses = 0
        daily_date   = None
        weekly_key   = None
        sess_key     = None
        sess_count   = 0

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

        # Pre-compute
        df_h4["ATR14"]       = Indicators.atr(df_h4["High"],df_h4["Low"],df_h4["Close"],14)
        df_h1["RSI14"]       = Indicators.rsi(df_h1["Close"],14)
        df_h1["EMA20"]       = Indicators.ema(df_h1["Close"],20)
        df_h1["EMA20slope3"] = df_h1["EMA20"].diff(3)
        df_h1["EMA20slope5"] = df_h1["EMA20"].diff(5)
        df_m15["RSI14"]      = Indicators.rsi(df_m15["Close"],14)
        df_m15["RSIslope"]   = Indicators.rsi_slope(df_m15["RSI14"],3)
        df_m15["ATR14"]      = Indicators.atr(df_m15["High"],df_m15["Low"],df_m15["Close"],14)
        df_m15["VolZ"]       = Indicators.volume_zscore(df_m15["Volume"],20)
        stk, std             = Indicators.stoch_rsi(df_m15["Close"])
        df_m15["StochK"]     = stk; df_m15["StochD"] = std

        trade_id = 0

        for i in range(50, len(df_m15)):
            bar_time = df_m15.index[i]; bar = df_m15.iloc[i]

            bdate = bar_time.date()
            if bdate != daily_date:
                daily_loss = 0.0; session_losses = 0; daily_date = bdate

            wkey = f"{bar_time.year}_{bar_time.isocalendar()[1]}"
            if wkey != weekly_key:
                weekly_loss = 0.0; weekly_key = wkey

            hour = bar_time.hour
            in_london = (LONDON_OPEN_H <= hour < LONDON_CLOSE_H)
            in_ny     = (NY_OPEN_H     <= hour < NY_CLOSE_H)
            in_sess   = in_london or in_ny

            if in_london:   sk = f"london_{bdate}"
            elif in_ny:     sk = f"ny_{bdate}"
            else:           sk = None

            if sk and sk != sess_key:
                sess_key = sk; sess_count = 0

            atr_val  = bar["ATR14"]    if not pd.isna(bar["ATR14"])    else 10.0
            rsi_slp  = bar["RSIslope"] if not pd.isna(bar["RSIslope"]) else 0.0
            vol_z    = bar["VolZ"]     if not pd.isna(bar["VolZ"])     else 0.0

            # Update open trades (simplified: mark outcome when SL/TP hit)
            closed_now = []
            for t in open_trades:
                entry = t["entry"]; sl = t["sl"]; tp = t["tp"]
                direction = t["direction"]
                if direction == "LONG":
                    if bar["Low"] <= sl and bar["High"] >= tp:
                        dist_sl = abs(bar["Open"] - sl)
                        dist_tp = abs(bar["Open"] - tp)
                        outcome = "TP" if dist_tp <= dist_sl else "SL"
                    elif bar["Low"] <= sl:  outcome = "SL"
                    elif bar["High"] >= tp: outcome = "TP"
                    else: outcome = None
                else:
                    if bar["High"] >= sl and bar["Low"] <= tp:
                        dist_sl = abs(bar["Open"] - sl)
                        dist_tp = abs(bar["Open"] - tp)
                        outcome = "TP" if dist_tp <= dist_sl else "SL"
                    elif bar["High"] >= sl: outcome = "SL"
                    elif bar["Low"] <= tp:  outcome = "TP"
                    else: outcome = None

                if outcome:
                    cp = tp if outcome == "TP" else sl
                    if direction == "LONG":
                        pnl = (cp - entry) / POINT * POINT * CONTRACT_SIZE * t["lot"]
                    else:
                        pnl = (entry - cp) / POINT * POINT * CONTRACT_SIZE * t["lot"]
                    t["pnl"] = round(pnl, 2)
                    t["outcome"] = outcome
                    t["win"] = 1 if pnl > 0 else 0
                    sim_balance += pnl; sim_equity = sim_balance
                    if pnl < 0:
                        daily_loss += abs(pnl); weekly_loss += abs(pnl)
                        session_losses += 1
                    else:
                        session_losses = 0
                    rows.append(t)
                    closed_now.append(t)

            open_trades = [t for t in open_trades if t not in closed_now]

            if not in_sess: continue
            if daily_loss >= MAX_DAILY_LOSS_USD: continue
            if session_losses >= MAX_CONSEC_LOSSES: continue
            if weekly_loss >= WEEKLY_DD_USD: continue
            if sess_count >= MAX_TRADES_PER_SESS: continue
            if len(open_trades) >= MAX_CONCURRENT_TRADES: continue

            spread_pts = bar.get("Spread", 25)
            if pd.isna(spread_pts): spread_pts = 25
            if spread_pts > MAX_SPREAD_PTS: continue

            h4_bars = df_h4[df_h4.index <= bar_time].tail(20)
            h1_bars = df_h1[df_h1.index <= bar_time].tail(20)
            if len(h4_bars) < 10 or len(h1_bars) < 5: continue

            h1_last  = h1_bars.iloc[-1]
            h4_trend_up = h4_bars["Close"].iloc[-1] > h4_bars["Close"].iloc[-5]

            h1_rsi   = h1_last.get("RSI14", 50.0) if hasattr(h1_last, "get") else 50.0
            h1_slope = h1_last.get("EMA20slope3", 0.0) if hasattr(h1_last, "get") else 0.0
            if pd.isna(h1_rsi):   h1_rsi   = 50.0
            if pd.isna(h1_slope): h1_slope = 0.0

            h1_bull  = (h1_rsi > 45) or (h1_slope > 0)
            h1_bear  = (h1_rsi < 55) or (h1_slope < 0)

            stk_k  = bar.get("StochK", 50.0); stk_d = bar.get("StochD", 50.0)
            if pd.isna(stk_k): stk_k = 50.0
            if pd.isna(stk_d): stk_d = 50.0
            prev   = df_m15.iloc[i-1]
            prev_k = prev.get("StochK", 50.0); prev_d = prev.get("StochD", 50.0)
            if pd.isna(prev_k): prev_k = 50.0
            if pd.isna(prev_d): prev_d = 50.0

            bull_cross = (stk_k > stk_d and prev_k <= prev_d and stk_k < 85)
            bear_cross = (stk_k < stk_d and prev_k >= prev_d and stk_k > 15)

            long_signal  = h4_trend_up       and h1_bull and bull_cross
            short_signal = (not h4_trend_up) and h1_bear and bear_cross
            if not (long_signal or short_signal): continue

            direction  = "LONG" if long_signal else "SHORT"
            open_dirs  = [t["direction"] for t in open_trades]
            if open_dirs.count(direction) >= 1: continue
            if vol_z < 0.0: continue

            conditions = {
                "elliott_bias":   h4_trend_up if direction=="LONG" else not h4_trend_up,
                "smc_structure":  h4_trend_up if direction=="LONG" else not h4_trend_up,
                "fvg_confluence": False,
                "h1_momentum":    h1_bull if direction=="LONG" else h1_bear,
                "m15_bos":        bull_cross if direction=="LONG" else bear_cross,
                "volume_confirm": vol_z > 0.5,
            }
            pw, _, grade = self.scorer.calculate(conditions)
            if grade == "SKIP": continue

            atr_pts = atr_val / POINT
            sl_pts  = max(int(atr_pts * SL_ATR_MULTIPLIER), 100)
            tp_pts  = int(sl_pts * TP_RR_RATIO)
            entry   = bar["Close"]
            sl = entry - sl_pts*POINT if direction=="LONG" else entry + sl_pts*POINT
            tp = entry + tp_pts*POINT if direction=="LONG" else entry - tp_pts*POINT

            risk   = sim_equity * RISK_NORMAL_PCT
            lot    = round(risk / (sl_pts * CONTRACT_SIZE * POINT) / LOT_STEP) * LOT_STEP
            lot    = max(LOT_MIN, min(LOT_MAX, round(lot, 2)))
            if lot <= 0: continue

            # Compute features
            feats = compute_features(df_m15, df_h1, df_h4, i, bar_time)
            if not feats: continue

            feats["direction"]  = direction
            feats["dir_binary"] = 1.0 if direction == "LONG" else 0.0
            feats["entry"]      = entry
            feats["sl"]         = sl
            feats["tp"]         = tp
            feats["lot"]        = lot
            feats["trade_id"]   = trade_id
            feats["open_time"]  = bar_time
            feats["sl_pts"]     = float(sl_pts)
            feats["tp_pts"]     = float(tp_pts)
            feats["pnl"]        = 0.0
            feats["win"]        = -1  # fill on close
            feats["outcome"]    = ""

            trade_id += 1
            open_trades.append(feats)
            sess_count += 1

        # Close any remaining open at end
        if open_trades and len(df_m15) > 0:
            last = df_m15.iloc[-1]
            for t in open_trades:
                entry = t["entry"]; sl = t["sl"]; tp = t["tp"]
                direction = t["direction"]
                cp = last["Close"]
                if direction == "LONG":
                    pnl = (cp - entry) / POINT * POINT * CONTRACT_SIZE * t["lot"]
                else:
                    pnl = (entry - cp) / POINT * POINT * CONTRACT_SIZE * t["lot"]
                t["pnl"] = round(pnl, 2)
                t["outcome"] = "BACKTEST_END"
                t["win"] = 1 if pnl > 0 else 0
                rows.append(t)

        df = pd.DataFrame(rows)
        # Remove non-feature columns for ML (keep them separately)
        print(f"  Generated {len(df)} trades with {len(df.columns)} columns")
        return df


# ================================================================
#  XGBOOST TRAINER + SHAP
# ================================================================

FEATURE_COLS = [
    "body_ratio","close_pos","is_bullish","gap_from_prev","consec_candles",
    "atr_pts","atr_ratio","atr_3bar_roc","atr_5bar_roc","atr_10bar_roc","atr_regime",
    "rsi","rsi_vs_50","rsi_3bar_slope","rsi_5bar_slope","rsi_10bar_slope",
    "rsi_overbought","rsi_oversold",
    "vol_z","vol_ratio","vol_slope_5bar","vol_vs_mean20",
    "stoch_k","stoch_d","stoch_kd_spread","stoch_overbought","stoch_oversold",
    "h1_ema_slope3","h1_ema_slope5","h1_price_vs_ema","h1_rsi",
    "macd_line","macd_hist","macd_hist_slope",
    "bb_width","bb_pct_b","bb_squeeze",
    "adx","di_plus","di_minus","di_spread",
    "h4_slope5","h4_slope10","h4_vs_ema20",
    "price_mom_3bar","price_mom_5bar","price_mom_10bar",
    "hour","day_of_week","is_london","is_ny","is_overlap","session_age_bars",
    "spread_pts","spread_atr_ratio","dir_binary",
]

def train_xgboost(df: pd.DataFrame) -> Tuple:
    """
    Train XGBoost classifier on win/loss.
    Returns (model, feature_importance_df, X_test, y_test, scaler).
    """
    print("\n" + "="*65)
    print("  XGBOOST TRAINING")
    print("="*65)

    # Filter to clean outcomes only (exclude BACKTEST_END)
    clean = df[df["outcome"].isin(["TP","SL"])].copy()
    clean["target"] = (clean["win"] == 1).astype(int)

    available = [c for c in FEATURE_COLS if c in clean.columns]
    print(f"  Samples: {len(clean)} | Features: {len(available)}")
    print(f"  Win rate: {clean['target'].mean()*100:.1f}%")

    X = clean[available].fillna(0).values
    y = clean["target"].values

    # Time-based split (no leakage): 80% train, 20% test
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # XGBoost with early stopping
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators       = 400,
        max_depth          = 4,
        learning_rate      = 0.03,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 5,
        gamma              = 0.1,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        scale_pos_weight   = scale_pos,
        eval_metric        = "logloss",
        verbosity          = 0,
        random_state       = 42,
    )

    model.fit(X_tr, y_tr, verbose=False)

    # Evaluate
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:,1]
    acc   = accuracy_score(y_te, preds)
    auc   = roc_auc_score(y_te, proba)

    print(f"\n  Test accuracy : {acc*100:.1f}%")
    print(f"  AUC-ROC       : {auc:.3f}")
    print(f"\n{classification_report(y_te, preds, target_names=['Loss','Win'])}")

    # Feature importance
    imp = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    available,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  TOP 20 FEATURES (XGBoost Gain Importance):")
    print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>12} {'Bar'}")
    print("  " + "-"*60)
    for idx, row in fi_df.head(20).iterrows():
        bar_len = int(row["importance"] * 400)
        bar     = "=" * bar_len
        print(f"  {idx+1:<5} {row['feature']:<30} {row['importance']:>12.4f}  {bar}")

    # SHAP values
    print("\n  Computing SHAP values...")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_te)

    # Mean absolute SHAP per feature
    shap_importance = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":          available,
        "shap_importance":  shap_importance,
        "xgb_importance":   [fi_df[fi_df["feature"]==f]["importance"].values[0]
                              if f in fi_df["feature"].values else 0 for f in available]
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    print(f"\n  TOP 20 FEATURES (SHAP Mean |Value|):")
    print(f"  {'Rank':<5} {'Feature':<30} {'SHAP':>10} {'XGB':>10}")
    print("  " + "-"*55)
    for idx, row in shap_df.head(20).iterrows():
        print(f"  {idx+1:<5} {row['feature']:<30} {row['shap_importance']:>10.4f} {row['xgb_importance']:>10.4f}")

    # Save
    path = os.path.join(OUTPUT_DIR, "ml_feature_importance.csv")
    shap_df.to_csv(path, index=False)
    print(f"\n  Feature importance saved: {path}")

    return model, shap_df, X_te, y_te, available, clean


# ================================================================
#  BACKTESTER FOR OPTUNA (fast single-pass)
# ================================================================

def run_backtest_with_params(df_h4, df_h1, df_m15, p: Dict) -> Dict:
    """
    Fast backtest using trial parameters p.
    Returns metrics dict.
    """
    balance = ACCOUNT_BALANCE; equity = ACCOUNT_BALANCE
    peak    = ACCOUNT_BALANCE; max_dd = 0.0
    open_trades = []
    closed_trades = []
    daily_loss = 0.0; weekly_loss = 0.0; session_losses = 0
    daily_date = None; weekly_key = None
    sess_key = None; sess_count = 0
    scorer = ScoringEngine()

    df_h4=df_h4.copy(); df_h1=df_h1.copy(); df_m15=df_m15.copy()

    for i in range(50, len(df_m15)):
        bar_time = df_m15.index[i]; bar = df_m15.iloc[i]

        bdate = bar_time.date()
        if bdate != daily_date:
            daily_loss = 0.0; session_losses = 0; daily_date = bdate

        wkey = f"{bar_time.year}_{bar_time.isocalendar()[1]}"
        if wkey != weekly_key:
            weekly_loss = 0.0; weekly_key = wkey

        hour = bar_time.hour
        in_london = (p["lon_open"] <= hour < p["lon_close"])
        in_ny     = (p["ny_open"]  <= hour < p["ny_close"])
        in_sess   = in_london or in_ny

        if in_london:   sk = f"london_{bdate}"
        elif in_ny:     sk = f"ny_{bdate}"
        else:           sk = None
        if sk and sk != sess_key:
            sess_key = sk; sess_count = 0

        atr_val  = bar["ATR14"]    if not pd.isna(bar["ATR14"])    else 10.0
        rsi_slp  = bar["RSIslope"] if not pd.isna(bar["RSIslope"]) else 0.0
        rsi_slp3 = bar["RSIslope3"] if not pd.isna(bar["RSIslope3"]) else 0.0
        vol_z    = bar["VolZ"]     if not pd.isna(bar["VolZ"])     else 0.0

        # Update open trades with configurable BE
        closed_now = []
        for t in open_trades:
            entry = t["entry"]; sl = t["sl"]; tp = t["tp"]
            direction = t["direction"]
            sp = t["spread"] * POINT
            be_min  = sp * p["be_trigger"]
            outcome = None

            if direction == "LONG":
                cur    = bar["Close"] - entry
                sl_hit = bar["Low"]  <= sl
                tp_hit = bar["High"] >= tp
                if sl_hit and tp_hit:
                    outcome = "TP" if abs(bar["Open"]-tp) <= abs(bar["Open"]-sl) else "SL"
                elif sl_hit: outcome = "SL"
                elif tp_hit: outcome = "TP"
                elif cur > be_min and rsi_slp < -5 and bar["BodySize"] < bar["Range"]*0.3:
                    outcome = "MOM"; t["tp"] = bar["Close"]
                elif cur > be_min and sl < entry:
                    new_sl = entry + sp * p["be_lock"]
                    if new_sl > sl: t["sl"] = new_sl; t["tp"] = bar["High"] + atr_val * TP_TRAIL_ATR_MULT
            else:
                cur    = entry - bar["Close"]
                sl_hit = bar["High"] >= sl
                tp_hit = bar["Low"]  <= tp
                if sl_hit and tp_hit:
                    outcome = "TP" if abs(bar["Open"]-tp) <= abs(bar["Open"]-sl) else "SL"
                elif sl_hit: outcome = "SL"
                elif tp_hit: outcome = "TP"
                elif cur > be_min and rsi_slp > 5 and bar["BodySize"] < bar["Range"]*0.3:
                    outcome = "MOM"; t["tp"] = bar["Close"]
                elif cur > be_min and sl > entry:
                    new_sl = entry - sp * p["be_lock"]
                    if new_sl < sl: t["sl"] = new_sl; t["tp"] = bar["Low"] - atr_val * TP_TRAIL_ATR_MULT

            if outcome:
                cp = t["tp"] if outcome in ("TP","MOM") else t["sl"]
                if direction == "LONG":
                    pnl = (cp - entry) / POINT * POINT * CONTRACT_SIZE * t["lot"]
                else:
                    pnl = (entry - cp) / POINT * POINT * CONTRACT_SIZE * t["lot"]
                balance += pnl; equity = balance
                peak = max(peak, equity); max_dd = max(max_dd, peak - equity)
                if pnl < 0:
                    daily_loss += abs(pnl); weekly_loss += abs(pnl); session_losses += 1
                else:
                    session_losses = 0
                t["pnl"] = round(pnl, 2); t["win"] = 1 if pnl > 0 else 0
                closed_trades.append(t); closed_now.append(t)

        open_trades = [t for t in open_trades if t not in closed_now]

        if not in_sess: continue
        if daily_loss  >= equity * p["daily_loss_pct"]:  continue
        if session_losses >= p["max_consec"]:             continue
        if weekly_loss >= equity * p["weekly_dd_pct"]:   continue
        if sess_count  >= 20:                             continue
        if len(open_trades) >= p["max_concurrent"]:      continue

        spread_pts = bar.get("Spread", 25)
        if pd.isna(spread_pts): spread_pts = 25
        if spread_pts > 20: continue

        h4_bars = df_h4[df_h4.index <= bar_time].tail(20)
        h1_bars = df_h1[df_h1.index <= bar_time].tail(20)
        if len(h4_bars) < 10 or len(h1_bars) < 5: continue

        h1_last     = h1_bars.iloc[-1]
        h4_trend_up = h4_bars["Close"].iloc[-1] > h4_bars["Close"].iloc[-5]

        h1_rsi   = h1_last.get("RSI14", 50.0)    if hasattr(h1_last,"get") else 50.0
        h1_slope = h1_last.get("EMA20slope3",0.0) if hasattr(h1_last,"get") else 0.0
        if pd.isna(h1_rsi): h1_rsi = 50.0
        if pd.isna(h1_slope): h1_slope = 0.0

        h1_bull = (h1_rsi > p["h1_rsi_min"])    or (h1_slope > 0)
        h1_bear = (h1_rsi < p["h1_rsi_max"])    or (h1_slope < 0)

        stk_k = bar.get("StochK", 50.0); stk_d = bar.get("StochD", 50.0)
        if pd.isna(stk_k): stk_k=50.0
        if pd.isna(stk_d): stk_d=50.0
        prev   = df_m15.iloc[i-1]
        prev_k = prev.get("StochK", 50.0); prev_d = prev.get("StochD", 50.0)
        if pd.isna(prev_k): prev_k=50.0
        if pd.isna(prev_d): prev_d=50.0

        bull_cross = (stk_k > stk_d and prev_k <= prev_d and stk_k < p["stoch_bull_cap"])
        bear_cross = (stk_k < stk_d and prev_k >= prev_d and stk_k > p["stoch_bear_floor"])

        long_signal  = h4_trend_up       and h1_bull and bull_cross
        short_signal = (not h4_trend_up) and h1_bear and bear_cross
        if not (long_signal or short_signal): continue

        direction  = "LONG" if long_signal else "SHORT"
        open_dirs  = [t["direction"] for t in open_trades]
        if open_dirs.count(direction) >= 1: continue

        # v1.4 filters with configurable thresholds
        if direction == "LONG"  and rsi_slp3 >  p["rsi_slope_gate"]: continue
        if direction == "SHORT" and rsi_slp3 < -p["rsi_slope_gate"]: continue

        h1_ema5 = h1_last.get("EMA20slope5", 0.0) if hasattr(h1_last,"get") else 0.0
        if pd.isna(h1_ema5): h1_ema5 = 0.0
        if abs(h1_ema5) > p["ema_slope_cap"]: continue

        if vol_z < p["vol_min_z"]: continue

        conditions = {
            "elliott_bias":   h4_trend_up if direction=="LONG" else not h4_trend_up,
            "smc_structure":  h4_trend_up if direction=="LONG" else not h4_trend_up,
            "fvg_confluence": False,
            "h1_momentum":    h1_bull if direction=="LONG" else h1_bear,
            "m15_bos":        bull_cross if direction=="LONG" else bear_cross,
            "volume_confirm": vol_z > 0.5,
        }
        pw, _, grade = scorer.calculate(conditions)
        if grade == "SKIP": continue

        atr_pts = atr_val / POINT
        sl_pts  = int(atr_pts * p["sl_atr_mult"])
        sl_pts  = max(int(atr_pts * p["sl_min_mult"]),
                      min(int(atr_pts * p["sl_max_mult"]), sl_pts))
        sl_pts  = max(sl_pts, 80)
        tp_pts  = int(sl_pts * p["tp_rr"])
        entry   = bar["Close"]
        sl = entry - sl_pts*POINT if direction=="LONG" else entry + sl_pts*POINT
        tp = entry + tp_pts*POINT if direction=="LONG" else entry - tp_pts*POINT

        risk = equity * p["risk_pct"]
        lot  = round(risk / (sl_pts * CONTRACT_SIZE * POINT) / LOT_STEP) * LOT_STEP
        lot  = max(LOT_MIN, min(LOT_MAX, round(lot, 2)))
        if lot <= 0: continue

        # Day-of-week lot reduction
        dow = bar_time.weekday()
        if dow == 0: lot = max(LOT_MIN, round(round(lot*p["monday_lot"]/LOT_STEP)*LOT_STEP,2))
        if dow == 3: lot = max(LOT_MIN, round(round(lot*p["thursday_lot"]/LOT_STEP)*LOT_STEP,2))

        open_trades.append({
            "entry": entry, "sl": sl, "tp": tp, "lot": lot,
            "direction": direction, "spread": spread_pts,
        })
        sess_count += 1

    # Close remaining
    if open_trades:
        last = df_m15.iloc[-1]
        for t in open_trades:
            cp = last["Close"]
            if t["direction"] == "LONG":
                pnl = (cp - t["entry"]) / POINT * POINT * CONTRACT_SIZE * t["lot"]
            else:
                pnl = (t["entry"] - cp) / POINT * POINT * CONTRACT_SIZE * t["lot"]
            balance += pnl; equity = balance
            peak = max(peak, equity); max_dd = max(max_dd, peak - equity)
            t["pnl"] = round(pnl, 2); t["win"] = 1 if pnl > 0 else 0
            closed_trades.append(t)

    pnls = [t["pnl"] for t in closed_trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    n = len(pnls) or 1
    wr = len(wins) / n
    pf = sum(wins) / abs(sum(losses)) if losses else 999.0
    dd_pct = (max_dd / ACCOUNT_BALANCE) * 100

    # Composite score: rewards profit and trades, penalises DD
    score = 0.0
    if n >= 50 and max_dd > 0:
        score = (sum(pnls) / max_dd) * wr * np.sqrt(n)

    return {
        "trades": n, "win_rate": wr, "profit": sum(pnls),
        "max_dd_pct": dd_pct, "profit_factor": pf,
        "score": score,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
    }


# ================================================================
#  OPTUNA BAYESIAN OPTIMIZATION
# ================================================================

def run_optuna(df_h4, df_h1, df_m15, n_trials=200):
    print("\n" + "="*65)
    print("  OPTUNA BAYESIAN OPTIMIZATION")
    print(f"  Trials: {n_trials}  |  Objective: max composite score")
    print("="*65)

    best_results = []

    def objective(trial):
        p = {
            # SL/TP
            "sl_atr_mult":    trial.suggest_float("sl_atr_mult",   0.8,  2.5),
            "sl_min_mult":    trial.suggest_float("sl_min_mult",   0.5,  1.5),
            "sl_max_mult":    trial.suggest_float("sl_max_mult",   1.5,  3.5),
            "tp_rr":          trial.suggest_float("tp_rr",         1.2,  4.0),
            # Breakeven
            "be_trigger":     trial.suggest_float("be_trigger",    2.0,  8.0),
            "be_lock":        trial.suggest_float("be_lock",       0.5,  5.0),
            # Risk / Position
            "risk_pct":       trial.suggest_float("risk_pct",      0.003, 0.02),
            "max_concurrent": trial.suggest_int  ("max_concurrent",1, 4),
            # Circuit breakers
            "max_consec":     trial.suggest_int  ("max_consec",    2, 7),
            "daily_loss_pct": trial.suggest_float("daily_loss_pct",0.02, 0.08),
            "weekly_dd_pct":  trial.suggest_float("weekly_dd_pct", 0.05, 0.20),
            # Signal filters
            "rsi_slope_gate": trial.suggest_float("rsi_slope_gate",0.10, 2.0),
            "ema_slope_cap":  trial.suggest_float("ema_slope_cap", 0.2,  5.0),
            "vol_min_z":      trial.suggest_float("vol_min_z",    -1.0,  1.0),
            "h1_rsi_min":     trial.suggest_float("h1_rsi_min",   38.0, 52.0),
            "h1_rsi_max":     trial.suggest_float("h1_rsi_max",   48.0, 62.0),
            "stoch_bull_cap": trial.suggest_float("stoch_bull_cap",75.0, 92.0),
            "stoch_bear_floor":trial.suggest_float("stoch_bear_floor",8.0,25.0),
            # Sessions
            "lon_open":       trial.suggest_int  ("lon_open",     6, 9),
            "lon_close":      trial.suggest_int  ("lon_close",    10, 14),
            "ny_open":        trial.suggest_int  ("ny_open",      11, 14),
            "ny_close":       trial.suggest_int  ("ny_close",     16, 20),
            # Day of week
            "monday_lot":     trial.suggest_float("monday_lot",   0.50, 1.00),
            "thursday_lot":   trial.suggest_float("thursday_lot", 0.50, 1.00),
        }

        # Constraint: sl_min_mult < sl_atr_mult < sl_max_mult
        if p["sl_min_mult"] >= p["sl_atr_mult"] or p["sl_atr_mult"] >= p["sl_max_mult"]:
            return 0.0
        # Constraint: h1_rsi_min < h1_rsi_max
        if p["h1_rsi_min"] >= p["h1_rsi_max"]:
            return 0.0

        try:
            r = run_backtest_with_params(df_h4, df_h1, df_m15, p)
        except Exception:
            return 0.0

        # Penalise extreme DD or too few trades
        if r["max_dd_pct"] > 50 or r["trades"] < 100:
            return 0.0

        return r["score"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=True, n_jobs=1)

    best = study.best_params
    best_r = run_backtest_with_params(df_h4, df_h1, df_m15, best)

    print(f"\n  Best score   : {study.best_value:.3f}")
    print(f"  Best trial   : #{study.best_trial.number}")
    print(f"\n  BEST PARAMETERS:")
    for k, v in sorted(best.items()):
        print(f"    {k:<25} = {v}")

    print(f"\n  BEST BACKTEST RESULT:")
    print(f"    Trades       : {best_r['trades']}")
    print(f"    Win rate     : {best_r['win_rate']*100:.1f}%")
    print(f"    Net profit   : ${best_r['profit']:,.2f}")
    print(f"    Max DD       : {best_r['max_dd_pct']:.1f}%")
    print(f"    Profit factor: {best_r['profit_factor']:.2f}")
    print(f"    Avg win      : ${best_r['avg_win']:.2f}")
    print(f"    Avg loss     : ${best_r['avg_loss']:.2f}")

    # Top 10 trials
    print(f"\n  TOP 10 TRIALS:")
    trials_df = study.trials_dataframe()
    top = trials_df.nlargest(10, "value")[["number","value"] +
          [c for c in trials_df.columns if c.startswith("params_")]]
    top.columns = ["trial","score"] + [c.replace("params_","") for c in top.columns[2:]]
    print(top.to_string(index=False))

    # Save trials
    path = os.path.join(OUTPUT_DIR, "optuna_trials.csv")
    trials_df.to_csv(path, index=False)
    print(f"\n  All trials saved: {path}")

    return best, best_r, study


# ================================================================
#  WRITE OPTIMAL CONFIG
# ================================================================

def write_optimal_config(best_params: Dict, best_r: Dict):
    """Write the Optuna best parameters as config_v1.5.py"""

    b = best_params

    content = f'''# ================================================================
#  config_v1.5.py  --  GoldScalperBot  XAUUSD...  (MultiBank)
#  VERSION 1.5  --  Optuna Bayesian Optimization Result
#
#  Generated by: ml_optimizer.py
#  Date        : {datetime.now().strftime("%Y-%m-%d %H:%M")}
#  Optuna score: (see optuna_trials.csv)
#
#  Optimization result:
#    Trades    : {best_r["trades"]}
#    Win rate  : {best_r["win_rate"]*100:.1f}%
#    Net profit: ${best_r["profit"]:,.2f}
#    Max DD    : {best_r["max_dd_pct"]:.1f}%
#    PF        : {best_r["profit_factor"]:.2f}
# ================================================================

# ---- BROKER / SYMBOL ---------------------------------------
SYMBOL          = "XAUUSD..."
BROKER_SERVER   = "MEXGlobalFinancial-Demo"
POINT           = 0.01
DIGITS          = 2
CONTRACT_SIZE   = 100
PIP_VALUE_STD   = 1.0
MAX_SPREAD_PTS  = 20

# ---- ACCOUNT -----------------------------------------------
ACCOUNT_BALANCE = 2000.0
ACCOUNT_LEVERAGE= 500

# ---- RISK PER TRADE (Optuna-optimised) ---------------------
RISK_NEWS_PCT    = 0.005
RISK_NEWS_USD    = ACCOUNT_BALANCE * RISK_NEWS_PCT

# ---- CONCURRENT TRADES & DYNAMIC RISK ----------------------
MAX_CONCURRENT_TRADES  = {b.get("max_concurrent", 2)}
TOTAL_RISK_BUDGET_PCT  = {b.get("risk_pct", 0.01) * b.get("max_concurrent", 2):.4f}
RISK_NORMAL_PCT        = TOTAL_RISK_BUDGET_PCT / MAX_CONCURRENT_TRADES
RISK_NORMAL_USD        = ACCOUNT_BALANCE * RISK_NORMAL_PCT

# ---- SL/TP PARAMETERS (Optuna-optimised) -------------------
SL_ATR_MULTIPLIER   = {b.get("sl_atr_mult", 1.5):.3f}
SL_ATR_MIN_MULT     = {b.get("sl_min_mult", 1.0):.3f}
SL_ATR_MAX_MULT     = {b.get("sl_max_mult", 2.5):.3f}
TP_RR_RATIO         = {b.get("tp_rr", 2.0):.3f}
TP_TRAIL_ATR_MULT   = 1.5

# ---- LOT SIZING --------------------------------------------
LOT_MIN         = 0.01
LOT_MAX         = 10.0
LOT_STEP        = 0.01
MIN_RR          = 1.5

# ---- COMMISSION -- spread-only model -----------------------
COMMISSION_USD  = 0.0

# ---- BREAKEVEN PARAMETERS (Optuna-optimised) ---------------
BE_PROFIT_TRIGGER_MULT = {b.get("be_trigger", 3.5):.2f}
BE_SL_LOCK_MULT        = {b.get("be_lock", 3.0):.2f}

# ---- CIRCUIT BREAKERS (Optuna-optimised) -------------------
MAX_DAILY_LOSS_PCT   = {b.get("daily_loss_pct", 0.04):.4f}
MAX_DAILY_LOSS_USD   = ACCOUNT_BALANCE * MAX_DAILY_LOSS_PCT
MAX_CONSEC_LOSSES    = {b.get("max_consec", 4)}
MAX_TRADES_PER_SESS  = 20
WEEKLY_DD_PCT        = {b.get("weekly_dd_pct", 0.10):.4f}
WEEKLY_DD_USD        = ACCOUNT_BALANCE * WEEKLY_DD_PCT

# ---- SESSIONS (Optuna-optimised) ---------------------------
LONDON_OPEN_H    = {b.get("lon_open", 7)}
LONDON_OPEN_MIN  = 0
LONDON_CLOSE_H   = {b.get("lon_close", 12)}
NY_OPEN_H        = {b.get("ny_open", 12)}
NY_CLOSE_H       = {b.get("ny_close", 18)}

# ---- SIGNAL FILTERS (Optuna-optimised) ---------------------
FILTER_VOL_MIN_Z        = {b.get("vol_min_z", 0.0):.3f}
FILTER_LONDON_DELAY     = False
FILTER_MONDAY_LOT       = {b.get("monday_lot", 0.70):.3f}
FILTER_THURSDAY_LOT     = {b.get("thursday_lot", 0.80):.3f}

# RSI 3-bar slope gate
FILTER_RSI_SLOPE_3BAR   = {b.get("rsi_slope_gate", 0.5):.3f}

# EMA20 H1 5-bar slope cap
FILTER_EMA_SLOPE_MAX    = {b.get("ema_slope_cap", 1.0):.3f}

# H1 RSI thresholds
H1_RSI_BULL_MIN         = {b.get("h1_rsi_min", 45.0):.1f}
H1_RSI_BEAR_MAX         = {b.get("h1_rsi_max", 55.0):.1f}

# StochRSI thresholds
STOCH_BULL_CAP          = {b.get("stoch_bull_cap", 85.0):.1f}
STOCH_BEAR_FLOOR        = {b.get("stoch_bear_floor", 15.0):.1f}

# ---- LOGGING -----------------------------------------------
LOG_VERBOSE     = False
LOG_TRADES      = True
'''

    path = os.path.join("config_v1.5.py")
    with open(path, "w") as f:
        f.write(content)
    print(f"\n  Optimal config written: {path}")
    print("  To use: copy config_v1.5.py -> config.py and run quick_diagnostic.py")


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  GoldScalperBot -- ML Optimizer")
    print("  XGBoost + SHAP + Optuna Bayesian Optimization")
    print("=" * 65)
    print(f"  Symbol  : {SYMBOL}")
    print(f"  Period  : {INSAMPLE_START.date()} to {INSAMPLE_END.date()}")
    print("=" * 65)

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

    # Normalise
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

    # Pre-compute all indicators (shared across all phases)
    df_h4["ATR14"]       = Indicators.atr(df_h4["High"],df_h4["Low"],df_h4["Close"],14)
    df_h1["RSI14"]       = Indicators.rsi(df_h1["Close"],14)
    df_h1["EMA20"]       = Indicators.ema(df_h1["Close"],20)
    df_h1["EMA20slope3"] = df_h1["EMA20"].diff(3)
    df_h1["EMA20slope5"] = df_h1["EMA20"].diff(5)
    df_m15["RSI14"]      = Indicators.rsi(df_m15["Close"],14)
    df_m15["RSIslope"]   = Indicators.rsi_slope(df_m15["RSI14"],3)
    df_m15["RSIslope3"]  = df_m15["RSI14"].diff(3)
    df_m15["ATR14"]      = Indicators.atr(df_m15["High"],df_m15["Low"],df_m15["Close"],14)
    df_m15["VolZ"]       = Indicators.volume_zscore(df_m15["Volume"],20)
    stk, std             = Indicators.stoch_rsi(df_m15["Close"])
    df_m15["StochK"]     = stk; df_m15["StochD"] = std

    # ---- PHASE 1: Generate trades + features --------------------
    t0 = time.time()
    gen = TradeGenerator()
    df_trades = gen.generate(df_h4, df_h1, df_m15)
    print(f"  Feature generation: {time.time()-t0:.1f}s")

    # Save feature matrix
    feat_path = os.path.join(OUTPUT_DIR, "ml_full_features.csv")
    df_trades.to_csv(feat_path, index=False)
    print(f"  Full feature matrix saved: {feat_path}")

    # ---- PHASE 2: XGBoost + SHAP --------------------------------
    t1 = time.time()
    model, shap_df, X_te, y_te, feat_names, clean_df = train_xgboost(df_trades)
    print(f"\n  XGBoost + SHAP: {time.time()-t1:.1f}s")

    # ---- PHASE 3: Optuna optimization ---------------------------
    N_TRIALS = 200  # Increase to 500 for deeper search (takes ~60 mins)
    print(f"\n  Starting Optuna ({N_TRIALS} trials)...")
    print("  Progress bar below -- grab a coffee, ~15-25 mins")
    t2 = time.time()
    best_params, best_r, study = run_optuna(df_h4, df_h1, df_m15, n_trials=N_TRIALS)
    print(f"\n  Optuna completed: {time.time()-t2:.1f}s")

    # ---- PHASE 4: Write optimal config --------------------------
    write_optimal_config(best_params, best_r)

    mt5.shutdown()
    print("\n" + "="*65)
    print("  DONE. Next steps:")
    print("  1. Review config_v1.5.py")
    print("  2. Copy it to config.py")
    print("  3. Run: python quick_diagnostic.py")
    print("="*65)
