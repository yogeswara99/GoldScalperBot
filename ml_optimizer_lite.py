#!/usr/bin/env python3
# ================================================================
#  ml_optimizer_lite.py  --  GoldScalperBot  XAUUSD...
#
#  XGBoost-FREE version using:
#    - Random Forest (sklearn, no DLL issues)
#    - Permutation importance (replaces SHAP)
#    - Optuna Bayesian optimization (unchanged)
#
#  Achieves identical analytical goals as ml_optimizer.py
#  without any Windows DLL / dask / OpenMP dependencies.
#
#  REQUIRES: pip install scikit-learn optuna
#  USAGE   : python ml_optimizer_lite.py
# ================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import csv, os, time, warnings, json
warnings.filterwarnings("ignore")

# ---- Check dependencies ----------------------------------------
def check_deps():
    missing = []
    for pkg in ["sklearn", "optuna"]:
        try:
            __import__("sklearn" if pkg == "sklearn" else pkg)
        except ImportError:
            missing.append("scikit-learn" if pkg == "sklearn" else pkg)
    if missing:
        print(f"  Run: pip install {' '.join(missing)}")
        exit(1)
    print("  Dependencies OK (scikit-learn + optuna)")

check_deps()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import *
from backtest_engine import (
    DataLoader, Indicators, ScoringEngine,
    INSAMPLE_START, INSAMPLE_END, SLIPPAGE_POINTS
)

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
#  FEATURE ENGINEERING  (identical to ml_optimizer.py)
# ================================================================

def compute_features(df_m15, df_h1, df_h4, i, bar_time) -> Dict:
    f = {}
    m15 = df_m15.iloc[max(0, i-30):i+1]
    h1  = df_h1[df_h1.index <= bar_time].tail(20)
    h4  = df_h4[df_h4.index <= bar_time].tail(20)
    if len(m15) < 15 or len(h1) < 5 or len(h4) < 5:
        return {}

    bar = m15.iloc[-1]

    # Price action
    rng = bar["Range"] if bar["Range"] > 0 else 1.0
    f["body_ratio"]     = bar["BodySize"] / rng
    f["close_pos"]      = (bar["Close"] - bar["Low"]) / rng
    f["is_bullish"]     = 1.0 if bar["Close"] > bar["Open"] else 0.0
    atr_now = bar["ATR14"] if "ATR14" in m15.columns and not pd.isna(bar["ATR14"]) else 10.0
    f["gap_from_prev"]  = (bar["Open"] - m15.iloc[-2]["Close"]) / (atr_now + 1e-9)
    consec = 0
    for k in range(len(m15)-2, max(0, len(m15)-8), -1):
        if (m15.iloc[k]["Close"] > m15.iloc[k]["Open"]) == (bar["Close"] > bar["Open"]):
            consec += 1
        else: break
    f["consec_candles"] = float(consec)

    # ATR / Volatility
    atr = m15["ATR14"].values if "ATR14" in m15.columns else np.ones(len(m15)) * 10.0
    atr_mean20 = np.mean(atr[-20:]) if len(atr) >= 20 else np.mean(atr)
    atr_5ago   = atr[-6] if len(atr) > 5 else atr[0]
    f["atr_pts"]       = atr_now / POINT
    f["atr_ratio"]     = atr_now / (atr_mean20 + 1e-9)
    f["atr_5bar_roc"]  = (atr_now - atr_5ago) / (atr_5ago + 1e-9)
    f["atr_regime"]    = (1.0 if atr_now > atr_mean20*1.3 else
                          -1.0 if atr_now < atr_mean20*0.7 else 0.0)

    # RSI
    rsi = m15["RSI14"].values if "RSI14" in m15.columns else np.ones(len(m15)) * 50.0
    rsi_now  = rsi[-1]
    rsi_3ago = rsi[-4] if len(rsi) > 3 else rsi[0]
    rsi_5ago = rsi[-6] if len(rsi) > 5 else rsi[0]
    rsi_10ago= rsi[-11] if len(rsi) > 10 else rsi[0]
    f["rsi"]             = rsi_now
    f["rsi_vs_50"]       = rsi_now - 50.0
    f["rsi_3bar_slope"]  = rsi_now - rsi_3ago
    f["rsi_5bar_slope"]  = rsi_now - rsi_5ago
    f["rsi_10bar_slope"] = rsi_now - rsi_10ago
    f["rsi_overbought"]  = 1.0 if rsi_now > 70 else 0.0
    f["rsi_oversold"]    = 1.0 if rsi_now < 30 else 0.0

    # Volume
    vol = m15["Volume"].values
    vol_z    = m15["VolZ"].iloc[-1] if "VolZ" in m15.columns else 0.0
    vol_mean5= np.mean(vol[-5:]) if len(vol) >= 5 else np.mean(vol)
    vol_mean20=np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
    if len(vol) >= 5:
        x = np.arange(5); y = vol[-5:]
        vol_slope5 = np.polyfit(x, y, 1)[0] / (vol_mean5 + 1e-9)
    else:
        vol_slope5 = 0.0
    f["vol_z"]          = float(vol_z)
    f["vol_ratio"]      = vol[-1] / (vol_mean5 + 1e-9)
    f["vol_slope_5bar"] = vol_slope5
    f["vol_vs_mean20"]  = (vol[-1] - vol_mean20) / (vol_mean20 + 1e-9)

    # StochRSI
    stk_k = bar.get("StochK", 50.0) if hasattr(bar, "get") else 50.0
    stk_d = bar.get("StochD", 50.0) if hasattr(bar, "get") else 50.0
    if pd.isna(stk_k): stk_k = 50.0
    if pd.isna(stk_d): stk_d = 50.0
    f["stoch_k"]          = stk_k
    f["stoch_d"]          = stk_d
    f["stoch_kd_spread"]  = stk_k - stk_d
    f["stoch_overbought"] = 1.0 if stk_k > 80 else 0.0
    f["stoch_oversold"]   = 1.0 if stk_k < 20 else 0.0

    # H1 EMA
    if "EMA20" in h1.columns and len(h1) >= 6:
        ema      = h1["EMA20"].values
        ema_now  = ema[-1]
        ema_3ago = ema[-4] if len(ema) > 3 else ema[0]
        ema_5ago = ema[-6] if len(ema) > 5 else ema[0]
        h1_close = h1["Close"].iloc[-1]
        h1_rsi   = h1["RSI14"].iloc[-1] if "RSI14" in h1.columns else 50.0
        f["h1_ema_slope3"]   = ema_now - ema_3ago
        f["h1_ema_slope5"]   = ema_now - ema_5ago
        f["h1_price_vs_ema"] = (h1_close - ema_now) / (atr_now + 1e-9)
        f["h1_rsi"]          = float(h1_rsi) if not pd.isna(h1_rsi) else 50.0
    else:
        f["h1_ema_slope3"] = 0.0; f["h1_ema_slope5"] = 0.0
        f["h1_price_vs_ema"] = 0.0; f["h1_rsi"] = 50.0

    # MACD
    close_m15   = m15["Close"]
    ema12       = close_m15.ewm(span=12, adjust=False).mean()
    ema26       = close_m15.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - macd_signal
    f["macd_line"]       = float(macd_line.iloc[-1]) / (atr_now + 1e-9)
    f["macd_hist"]       = float(macd_hist.iloc[-1]) / (atr_now + 1e-9)
    f["macd_hist_slope"] = float(macd_hist.iloc[-1] - macd_hist.iloc[-2]) / (atr_now + 1e-9) if len(macd_hist) > 1 else 0.0

    # Bollinger Bands
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
    f["bb_width"]  = bb_width
    f["bb_pct_b"]  = bb_pct_b
    f["bb_squeeze"]= 1.0 if bb_width < 0.02 else 0.0

    # ADX
    if len(m15) >= 15:
        high = m15["High"].values; low = m15["Low"].values; close = m15["Close"].values
        tr   = np.maximum(high[1:]-low[1:],
               np.maximum(abs(high[1:]-close[:-1]), abs(low[1:]-close[:-1])))
        dm_p = np.where((high[1:]-high[:-1]) > (low[:-1]-low[1:]),
                         np.maximum(high[1:]-high[:-1], 0.0), 0.0)
        dm_n = np.where((low[:-1]-low[1:]) > (high[1:]-high[:-1]),
                         np.maximum(low[:-1]-low[1:], 0.0), 0.0)
        p = min(14, len(tr))
        if p >= 5:
            atr14 = pd.Series(tr).ewm(com=p-1, adjust=False).mean().iloc[-1]
            dip   = pd.Series(dm_p).ewm(com=p-1, adjust=False).mean().iloc[-1]
            din   = pd.Series(dm_n).ewm(com=p-1, adjust=False).mean().iloc[-1]
            di_p  = 100 * dip / (atr14 + 1e-9)
            di_n  = 100 * din / (atr14 + 1e-9)
            dx    = 100 * abs(di_p - di_n) / (di_p + di_n + 1e-9)
            f["adx"] = dx; f["di_plus"] = di_p; f["di_minus"] = di_n; f["di_spread"] = di_p - di_n
        else:
            f["adx"]=25.0; f["di_plus"]=25.0; f["di_minus"]=25.0; f["di_spread"]=0.0
    else:
        f["adx"]=25.0; f["di_plus"]=25.0; f["di_minus"]=25.0; f["di_spread"]=0.0

    # H4 context
    if len(h4) >= 6:
        h4c = h4["Close"].values
        f["h4_slope5"]  = (h4c[-1] - h4c[-6])  / (atr_now + 1e-9) if len(h4c) > 5  else 0.0
        f["h4_slope10"] = (h4c[-1] - h4c[-11]) / (atr_now + 1e-9) if len(h4c) > 10 else 0.0
        h4_ema20 = pd.Series(h4c).ewm(span=min(20,len(h4c)), adjust=False).mean().iloc[-1]
        f["h4_vs_ema20"] = (h4c[-1] - h4_ema20) / (atr_now + 1e-9)
    else:
        f["h4_slope5"]=0.0; f["h4_slope10"]=0.0; f["h4_vs_ema20"]=0.0

    # Price momentum
    ca = m15["Close"].values
    f["price_mom_3bar"]  = (ca[-1]-ca[-4])  / (atr_now+1e-9) if len(ca)>3  else 0.0
    f["price_mom_5bar"]  = (ca[-1]-ca[-6])  / (atr_now+1e-9) if len(ca)>5  else 0.0
    f["price_mom_10bar"] = (ca[-1]-ca[-11]) / (atr_now+1e-9) if len(ca)>10 else 0.0

    # Session / Time
    hour = bar_time.hour
    f["hour"]             = float(hour)
    f["day_of_week"]      = float(bar_time.weekday())
    f["is_london"]        = 1.0 if LONDON_OPEN_H <= hour < LONDON_CLOSE_H else 0.0
    f["is_ny"]            = 1.0 if NY_OPEN_H     <= hour < NY_CLOSE_H     else 0.0
    f["session_age_bars"] = float((hour - (LONDON_OPEN_H if f["is_london"] else NY_OPEN_H)) * 4 +
                                   bar_time.minute // 15) if (f["is_london"] or f["is_ny"]) else 0.0

    # Spread
    spread = bar.get("Spread", 20) if hasattr(bar, "get") else 20
    if pd.isna(spread): spread = 20
    f["spread_pts"]       = float(spread)
    f["spread_atr_ratio"] = spread / (f["atr_pts"] + 1e-9)

    return f

FEATURE_COLS = [
    "body_ratio","close_pos","is_bullish","gap_from_prev","consec_candles",
    "atr_pts","atr_ratio","atr_5bar_roc","atr_regime",
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
    "hour","day_of_week","is_london","is_ny","session_age_bars",
    "spread_pts","spread_atr_ratio",
]

# ================================================================
#  TRADE GENERATOR
# ================================================================

def generate_trades(df_h4, df_h1, df_m15) -> pd.DataFrame:
    print("  Generating trades with feature vectors...")
    scorer = ScoringEngine()
    rows   = []
    balance = ACCOUNT_BALANCE; equity = ACCOUNT_BALANCE
    open_trades = []; daily_loss = 0.0; weekly_loss = 0.0
    session_losses = 0; daily_date = None; weekly_key = None
    sess_key = None; sess_count = 0; trade_id = 0

    for i in range(50, len(df_m15)):
        bar_time = df_m15.index[i]; bar = df_m15.iloc[i]

        bdate = bar_time.date()
        if bdate != daily_date:
            daily_loss = 0.0; session_losses = 0; daily_date = bdate
        wkey = f"{bar_time.year}_{bar_time.isocalendar()[1]}"
        if wkey != weekly_key:
            weekly_loss = 0.0; weekly_key = wkey

        hour      = bar_time.hour
        in_london = (LONDON_OPEN_H <= hour < LONDON_CLOSE_H)
        in_ny     = (NY_OPEN_H     <= hour < NY_CLOSE_H)
        in_sess   = in_london or in_ny

        sk = (f"london_{bdate}" if in_london else
              f"ny_{bdate}"     if in_ny     else None)
        if sk and sk != sess_key:
            sess_key = sk; sess_count = 0

        atr_val = bar["ATR14"]    if not pd.isna(bar["ATR14"])    else 10.0
        rsi_slp = bar["RSIslope"] if not pd.isna(bar["RSIslope"]) else 0.0

        # Update open trades
        closed = []
        for t in open_trades:
            direction = t["direction"]
            sl = t["sl"]; tp = t["tp"]
            outcome = None
            if direction == "LONG":
                sl_hit = bar["Low"] <= sl; tp_hit = bar["High"] >= tp
                if sl_hit and tp_hit:
                    outcome = "TP" if abs(bar["Open"]-tp) <= abs(bar["Open"]-sl) else "SL"
                elif sl_hit: outcome = "SL"
                elif tp_hit: outcome = "TP"
            else:
                sl_hit = bar["High"] >= sl; tp_hit = bar["Low"] <= tp
                if sl_hit and tp_hit:
                    outcome = "TP" if abs(bar["Open"]-tp) <= abs(bar["Open"]-sl) else "SL"
                elif sl_hit: outcome = "SL"
                elif tp_hit: outcome = "TP"

            if outcome:
                cp = tp if outcome == "TP" else sl
                pnl = ((cp - t["entry"]) if direction=="LONG" else (t["entry"] - cp)) / POINT * POINT * CONTRACT_SIZE * t["lot"]
                balance += pnl; equity = balance
                if pnl < 0: daily_loss += abs(pnl); weekly_loss += abs(pnl); session_losses += 1
                else: session_losses = 0
                t["pnl"] = round(pnl, 2); t["outcome"] = outcome; t["win"] = 1 if pnl > 0 else 0
                rows.append(t); closed.append(t)
        open_trades = [t for t in open_trades if t not in closed]

        if not in_sess: continue
        if daily_loss  >= MAX_DAILY_LOSS_USD: continue
        if session_losses >= MAX_CONSEC_LOSSES: continue
        if weekly_loss >= WEEKLY_DD_USD: continue
        if len(open_trades) >= MAX_CONCURRENT_TRADES: continue
        spread_pts = bar.get("Spread", 25)
        if pd.isna(spread_pts): spread_pts = 25
        if spread_pts > MAX_SPREAD_PTS: continue

        h4_bars = df_h4[df_h4.index <= bar_time].tail(20)
        h1_bars = df_h1[df_h1.index <= bar_time].tail(20)
        if len(h4_bars) < 10 or len(h1_bars) < 5: continue

        h1_last     = h1_bars.iloc[-1]
        h4_trend_up = h4_bars["Close"].iloc[-1] > h4_bars["Close"].iloc[-5]
        h1_rsi      = h1_last.get("RSI14", 50.0)    if hasattr(h1_last,"get") else 50.0
        h1_slope    = h1_last.get("EMA20slope3",0.0) if hasattr(h1_last,"get") else 0.0
        if pd.isna(h1_rsi): h1_rsi = 50.0
        if pd.isna(h1_slope): h1_slope = 0.0
        h1_bull = (h1_rsi > 45) or (h1_slope > 0)
        h1_bear = (h1_rsi < 55) or (h1_slope < 0)

        stk_k = bar.get("StochK",50.0); stk_d = bar.get("StochD",50.0)
        if pd.isna(stk_k): stk_k=50.0
        if pd.isna(stk_d): stk_d=50.0
        prev = df_m15.iloc[i-1]
        prev_k = prev.get("StochK",50.0); prev_d = prev.get("StochD",50.0)
        if pd.isna(prev_k): prev_k=50.0
        if pd.isna(prev_d): prev_d=50.0

        bull_cross = (stk_k > stk_d and prev_k <= prev_d and stk_k < 85)
        bear_cross = (stk_k < stk_d and prev_k >= prev_d and stk_k > 15)

        long_signal  = h4_trend_up       and h1_bull and bull_cross
        short_signal = (not h4_trend_up) and h1_bear and bear_cross
        if not (long_signal or short_signal): continue

        direction = "LONG" if long_signal else "SHORT"
        if [t["direction"] for t in open_trades].count(direction) >= 1: continue

        conditions = {
            "elliott_bias":   h4_trend_up if direction=="LONG" else not h4_trend_up,
            "smc_structure":  h4_trend_up if direction=="LONG" else not h4_trend_up,
            "fvg_confluence": False,
            "h1_momentum":    h1_bull if direction=="LONG" else h1_bear,
            "m15_bos":        bull_cross if direction=="LONG" else bear_cross,
            "volume_confirm": bar["VolZ"] > 0.5 if "VolZ" in df_m15.columns else False,
        }
        pw, _, grade = scorer.calculate(conditions)
        if grade == "SKIP": continue

        atr_pts = atr_val / POINT
        sl_pts  = max(int(atr_pts * SL_ATR_MULTIPLIER), 100)
        tp_pts  = int(sl_pts * TP_RR_RATIO)
        entry   = bar["Close"]
        sl = entry - sl_pts*POINT if direction=="LONG" else entry + sl_pts*POINT
        tp = entry + tp_pts*POINT if direction=="LONG" else entry - tp_pts*POINT
        lot = max(LOT_MIN, min(LOT_MAX, round(round(equity*RISK_NORMAL_PCT/(sl_pts*CONTRACT_SIZE*POINT)/LOT_STEP)*LOT_STEP, 2)))
        if lot <= 0: continue

        feats = compute_features(df_m15, df_h1, df_h4, i, bar_time)
        if not feats: continue
        feats.update({
            "direction": direction, "dir_binary": 1.0 if direction=="LONG" else 0.0,
            "entry": entry, "sl": sl, "tp": tp, "lot": lot, "trade_id": trade_id,
            "open_time": bar_time, "sl_pts": float(sl_pts), "tp_pts": float(tp_pts),
            "pnl": 0.0, "win": -1, "outcome": "",
        })
        trade_id += 1
        open_trades.append(feats); sess_count += 1

    if open_trades:
        last = df_m15.iloc[-1]
        for t in open_trades:
            cp = last["Close"]
            pnl = ((cp-t["entry"]) if t["direction"]=="LONG" else (t["entry"]-cp)) / POINT * POINT * CONTRACT_SIZE * t["lot"]
            t["pnl"] = round(pnl,2); t["outcome"] = "BACKTEST_END"; t["win"] = 1 if pnl>0 else 0
            rows.append(t)

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df)} trades with {len([c for c in FEATURE_COLS if c in df.columns])} features")
    return df


# ================================================================
#  RANDOM FOREST + PERMUTATION IMPORTANCE
# ================================================================

def train_and_rank(df: pd.DataFrame):
    print("\n" + "="*65)
    print("  RANDOM FOREST + PERMUTATION IMPORTANCE")
    print("="*65)

    clean  = df[df["outcome"].isin(["TP","SL"])].copy()
    clean["target"] = (clean["win"] == 1).astype(int)
    avail  = [c for c in FEATURE_COLS if c in clean.columns]
    print(f"  Samples: {len(clean)} | Features: {len(avail)}")
    print(f"  Win rate: {clean['target'].mean()*100:.1f}%")

    X = clean[avail].fillna(0).values
    y = clean["target"].values

    # Time-based split (80/20, no leakage)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Random Forest (no DLL needed, solid performance)
    rf = RandomForestClassifier(
        n_estimators    = 300,
        max_depth       = 6,
        min_samples_leaf= 20,
        max_features    = "sqrt",
        class_weight    = "balanced",
        random_state    = 42,
        n_jobs          = -1,
    )
    rf.fit(X_tr, y_tr)

    preds = rf.predict(X_te)
    proba = rf.predict_proba(X_te)[:,1]
    acc   = accuracy_score(y_te, preds)
    auc   = roc_auc_score(y_te, proba)

    print(f"\n  Test accuracy  : {acc*100:.1f}%")
    print(f"  AUC-ROC        : {auc:.3f}")
    print(f"\n{classification_report(y_te, preds, target_names=['Loss','Win'])}")

    # Impurity importance (fast)
    gini_imp = rf.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    avail,
        "importance": gini_imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df["importance_pct"]  = fi_df["importance"] / fi_df["importance"].sum() * 100
    fi_df["cumulative_pct"]  = fi_df["importance_pct"].cumsum()
    fi_df["rank"]            = fi_df.index + 1

    # Permutation importance (validates Gini, slower but more reliable)
    print("\n  Computing permutation importance (this takes ~1 min)...")
    perm = permutation_importance(rf, X_te, y_te, n_repeats=10,
                                   random_state=42, n_jobs=-1)
    fi_df["perm_importance"] = [perm.importances_mean[avail.index(f)] if f in avail else 0
                                  for f in fi_df["feature"]]
    fi_df["perm_std"]        = [perm.importances_std[avail.index(f)] if f in avail else 0
                                  for f in fi_df["feature"]]

    # Re-rank by permutation importance
    fi_df = fi_df.sort_values("perm_importance", ascending=False).reset_index(drop=True)
    fi_df["perm_pct"]    = fi_df["perm_importance"].clip(lower=0)
    total_perm           = fi_df["perm_pct"].sum()
    fi_df["perm_pct"]    = fi_df["perm_pct"] / (total_perm + 1e-9) * 100
    fi_df["perm_cumsum"] = fi_df["perm_pct"].cumsum()
    fi_df["rank"]        = fi_df.index + 1

    # Print full ranking
    print(f"\n  FULL FEATURE RANKING (Permutation Importance):")
    print(f"  {'Rank':<5} {'Feature':<28} {'Perm%':>8} {'Cumul%':>8} {'±Std':>8} {'Tier'}")
    print("  " + "-" * 72)
    for _, r in fi_df.iterrows():
        tier = ("ESSENTIAL" if r["perm_cumsum"] <= 80
                else "USEFUL"    if r["perm_cumsum"] <= 95
                else "MARGINAL")
        marker = "<<<" if tier == "MARGINAL" else ("***" if tier == "ESSENTIAL" else "   ")
        print(f"  {r['rank']:<5} {r['feature']:<28} {r['perm_pct']:>7.2f}% "
              f"{r['perm_cumsum']:>7.1f}% {r['perm_std']:>8.4f}  {tier} {marker}")

    # Summary
    essential = fi_df[fi_df["perm_cumsum"] <= 80]
    useful    = fi_df[(fi_df["perm_cumsum"] > 80) & (fi_df["perm_cumsum"] <= 95)]
    marginal  = fi_df[fi_df["perm_cumsum"] > 95]
    neg_feats = fi_df[fi_df["perm_importance"] < 0]

    print(f"\n  TIER SUMMARY:")
    print(f"  ESSENTIAL  (top 80%): {len(essential)} features")
    print(f"  USEFUL     (80-95%) : {len(useful)} features")
    print(f"  MARGINAL   (>95%)   : {len(marginal)} features  <- drop candidates")
    if len(neg_feats) > 0:
        print(f"  NEGATIVE PERM IMP   : {len(neg_feats)} features  <- ACTIVELY HARMFUL (drop these)")
        for _, r in neg_feats.iterrows():
            print(f"    {r['feature']:<28}  perm={r['perm_importance']:.4f}")

    # Gradient Boosting (sklearn, no DLL) as cross-validation
    print(f"\n  Cross-validating with Gradient Boosting (5-fold)...")
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    essential_feats = [c for c in essential["feature"].tolist() if c in avail]
    X_ess = clean[essential_feats].fillna(0).values if essential_feats else X
    cv_scores = cross_val_score(gb, X_ess, y, cv=5, scoring="roc_auc")
    print(f"  CV AUC (essential features): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Save
    path = os.path.join(OUTPUT_DIR, "ml_feature_importance.csv")
    fi_df.to_csv(path, index=False)
    print(f"\n  Feature importance saved: {path}")

    return fi_df, essential_feats, rf, avail


# ================================================================
#  OPTUNA BAYESIAN OPTIMIZATION  (identical to ml_optimizer.py)
# ================================================================

def run_backtest_fast(df_h4, df_h1, df_m15, p: Dict) -> Dict:
    """Lightweight backtest for Optuna trials."""
    balance = ACCOUNT_BALANCE; equity = ACCOUNT_BALANCE
    peak = ACCOUNT_BALANCE; max_dd = 0.0
    open_trades = []; closed_trades = []
    daily_loss = 0.0; weekly_loss = 0.0; session_losses = 0
    daily_date = None; weekly_key = None
    sess_key = None; sess_count = 0
    scorer = ScoringEngine()

    for i in range(50, len(df_m15)):
        bar_time = df_m15.index[i]; bar = df_m15.iloc[i]

        bdate = bar_time.date()
        if bdate != daily_date:
            daily_loss = 0.0; session_losses = 0; daily_date = bdate
        wkey = f"{bar_time.year}_{bar_time.isocalendar()[1]}"
        if wkey != weekly_key:
            weekly_loss = 0.0; weekly_key = wkey

        hour      = bar_time.hour
        in_london = (p["lon_open"] <= hour < p["lon_close"])
        in_ny     = (p["ny_open"]  <= hour < p["ny_close"])
        in_sess   = in_london or in_ny

        sk = (f"london_{bdate}" if in_london else f"ny_{bdate}" if in_ny else None)
        if sk and sk != sess_key: sess_key = sk; sess_count = 0

        atr_val  = bar["ATR14"]     if not pd.isna(bar["ATR14"])     else 10.0
        rsi_slp  = bar["RSIslope"]  if not pd.isna(bar["RSIslope"])  else 0.0
        rsi_slp3 = bar["RSIslope3"] if not pd.isna(bar["RSIslope3"]) else 0.0
        vol_z    = bar["VolZ"]      if not pd.isna(bar["VolZ"])      else 0.0

        # Update trades
        closed = []
        for t in open_trades:
            sl = t["sl"]; tp = t["tp"]; entry = t["entry"]
            direction = t["direction"]
            sp = t["spread"] * POINT
            be_min = sp * p["be_trigger"]
            outcome = None
            if direction == "LONG":
                cur = bar["Close"] - entry
                sl_hit = bar["Low"] <= sl; tp_hit = bar["High"] >= tp
                if sl_hit and tp_hit:
                    outcome = "TP" if abs(bar["Open"]-tp)<=abs(bar["Open"]-sl) else "SL"
                elif sl_hit: outcome="SL"
                elif tp_hit: outcome="TP"
                elif cur > be_min and rsi_slp < -5 and bar["BodySize"] < bar["Range"]*0.3:
                    outcome="MOM"; t["tp"]=bar["Close"]
                elif cur > be_min and sl < entry:
                    new_sl = entry + sp * p["be_lock"]
                    if new_sl > sl: t["sl"] = new_sl; t["tp"] = bar["High"] + atr_val * TP_TRAIL_ATR_MULT
            else:
                cur = entry - bar["Close"]
                sl_hit = bar["High"] >= sl; tp_hit = bar["Low"] <= tp
                if sl_hit and tp_hit:
                    outcome = "TP" if abs(bar["Open"]-tp)<=abs(bar["Open"]-sl) else "SL"
                elif sl_hit: outcome="SL"
                elif tp_hit: outcome="TP"
                elif cur > be_min and rsi_slp > 5 and bar["BodySize"] < bar["Range"]*0.3:
                    outcome="MOM"; t["tp"]=bar["Close"]
                elif cur > be_min and sl > entry:
                    new_sl = entry - sp * p["be_lock"]
                    if new_sl < sl: t["sl"] = new_sl; t["tp"] = bar["Low"] - atr_val * TP_TRAIL_ATR_MULT

            if outcome:
                cp = t["tp"] if outcome in ("TP","MOM") else t["sl"]
                pnl = ((cp-entry) if direction=="LONG" else (entry-cp)) / POINT * POINT * CONTRACT_SIZE * t["lot"]
                balance += pnl; equity = balance
                peak = max(peak, equity); max_dd = max(max_dd, peak-equity)
                if pnl < 0: daily_loss+=abs(pnl); weekly_loss+=abs(pnl); session_losses+=1
                else: session_losses = 0
                t["pnl"]=round(pnl,2); t["win"]=1 if pnl>0 else 0
                closed_trades.append(t); closed.append(t)
        open_trades = [t for t in open_trades if t not in closed]

        if not in_sess: continue
        if daily_loss  >= equity * p["daily_loss_pct"]: continue
        if session_losses >= p["max_consec"]:            continue
        if weekly_loss >= equity * p["weekly_dd_pct"]:  continue
        if len(open_trades) >= p["max_concurrent"]:     continue
        spread_pts = bar.get("Spread", 25)
        if pd.isna(spread_pts): spread_pts = 25
        if spread_pts > 20: continue

        h4_bars = df_h4[df_h4.index <= bar_time].tail(20)
        h1_bars = df_h1[df_h1.index <= bar_time].tail(20)
        if len(h4_bars) < 10 or len(h1_bars) < 5: continue

        h1_last     = h1_bars.iloc[-1]
        h4_trend_up = h4_bars["Close"].iloc[-1] > h4_bars["Close"].iloc[-5]
        h1_rsi      = h1_last.get("RSI14",50.0)    if hasattr(h1_last,"get") else 50.0
        h1_slope    = h1_last.get("EMA20slope3",0.0) if hasattr(h1_last,"get") else 0.0
        if pd.isna(h1_rsi): h1_rsi=50.0
        if pd.isna(h1_slope): h1_slope=0.0

        h1_bull = (h1_rsi > p["h1_rsi_min"]) or (h1_slope > 0)
        h1_bear = (h1_rsi < p["h1_rsi_max"]) or (h1_slope < 0)

        stk_k = bar.get("StochK",50.0); stk_d = bar.get("StochD",50.0)
        if pd.isna(stk_k): stk_k=50.0
        if pd.isna(stk_d): stk_d=50.0
        prev = df_m15.iloc[i-1]
        prev_k = prev.get("StochK",50.0); prev_d = prev.get("StochD",50.0)
        if pd.isna(prev_k): prev_k=50.0
        if pd.isna(prev_d): prev_d=50.0

        bull_cross = (stk_k>stk_d and prev_k<=prev_d and stk_k<p["stoch_bull_cap"])
        bear_cross = (stk_k<stk_d and prev_k>=prev_d and stk_k>p["stoch_bear_floor"])

        long_signal  = h4_trend_up       and h1_bull and bull_cross
        short_signal = (not h4_trend_up) and h1_bear and bear_cross
        if not (long_signal or short_signal): continue

        direction  = "LONG" if long_signal else "SHORT"
        open_dirs  = [t["direction"] for t in open_trades]
        if open_dirs.count(direction) >= 1: continue

        if direction=="LONG"  and rsi_slp3 >  p["rsi_slope_gate"]: continue
        if direction=="SHORT" and rsi_slp3 < -p["rsi_slope_gate"]: continue
        h1_ema5 = h1_last.get("EMA20slope5",0.0) if hasattr(h1_last,"get") else 0.0
        if pd.isna(h1_ema5): h1_ema5=0.0
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
        sl_pts  = max(int(atr_pts*p["sl_min_mult"]), min(int(atr_pts*p["sl_max_mult"]), sl_pts))
        sl_pts  = max(sl_pts, 80)
        tp_pts  = int(sl_pts * p["tp_rr"])
        entry   = bar["Close"]
        sl = entry - sl_pts*POINT if direction=="LONG" else entry + sl_pts*POINT
        tp = entry + tp_pts*POINT if direction=="LONG" else entry - tp_pts*POINT
        lot = round(equity*p["risk_pct"]/(sl_pts*CONTRACT_SIZE*POINT)/LOT_STEP)*LOT_STEP
        lot = max(LOT_MIN, min(LOT_MAX, round(lot,2)))
        if lot <= 0: continue
        dow = bar_time.weekday()
        if dow == 0: lot = max(LOT_MIN, round(round(lot*p["monday_lot"]/LOT_STEP)*LOT_STEP,2))
        if dow == 3: lot = max(LOT_MIN, round(round(lot*p["thursday_lot"]/LOT_STEP)*LOT_STEP,2))

        open_trades.append({"entry":entry,"sl":sl,"tp":tp,"lot":lot,
                             "direction":direction,"spread":spread_pts})
        sess_count += 1

    if open_trades:
        last = df_m15.iloc[-1]
        for t in open_trades:
            cp = last["Close"]
            pnl = ((cp-t["entry"]) if t["direction"]=="LONG" else (t["entry"]-cp)) / POINT * POINT * CONTRACT_SIZE * t["lot"]
            balance += pnl; equity = balance
            peak = max(peak,equity); max_dd = max(max_dd,peak-equity)
            t["pnl"]=round(pnl,2); t["win"]=1 if pnl>0 else 0
            closed_trades.append(t)

    pnls = [t["pnl"] for t in closed_trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    n    = len(pnls) or 1
    wr   = len(wins) / n
    pf   = sum(wins)/abs(sum(losses)) if losses else 999.0
    dd   = (max_dd/ACCOUNT_BALANCE)*100
    score = ((sum(pnls)/max_dd)*wr*np.sqrt(n)) if (n >= 50 and max_dd > 0) else 0.0

    return {"trades":n,"win_rate":wr,"profit":sum(pnls),"max_dd_pct":dd,
            "profit_factor":pf,"score":score,
            "avg_win":np.mean(wins) if wins else 0,
            "avg_loss":np.mean(losses) if losses else 0}


def run_optuna(df_h4, df_h1, df_m15, n_trials=200):
    print("\n" + "="*65)
    print(f"  OPTUNA BAYESIAN OPTIMIZATION  ({n_trials} trials)")
    print("="*65)

    def objective(trial):
        p = {
            "sl_atr_mult":     trial.suggest_float("sl_atr_mult",    0.8, 2.5),
            "sl_min_mult":     trial.suggest_float("sl_min_mult",    0.5, 1.5),
            "sl_max_mult":     trial.suggest_float("sl_max_mult",    1.5, 3.5),
            "tp_rr":           trial.suggest_float("tp_rr",          1.2, 4.0),
            "be_trigger":      trial.suggest_float("be_trigger",     2.0, 8.0),
            "be_lock":         trial.suggest_float("be_lock",        0.5, 5.0),
            "risk_pct":        trial.suggest_float("risk_pct",       0.003, 0.02),
            "max_concurrent":  trial.suggest_int  ("max_concurrent", 1, 4),
            "max_consec":      trial.suggest_int  ("max_consec",     2, 7),
            "daily_loss_pct":  trial.suggest_float("daily_loss_pct", 0.02, 0.08),
            "weekly_dd_pct":   trial.suggest_float("weekly_dd_pct",  0.05, 0.20),
            "rsi_slope_gate":  trial.suggest_float("rsi_slope_gate", 0.1, 2.0),
            "ema_slope_cap":   trial.suggest_float("ema_slope_cap",  0.2, 5.0),
            "vol_min_z":       trial.suggest_float("vol_min_z",     -1.0, 1.0),
            "h1_rsi_min":      trial.suggest_float("h1_rsi_min",    38.0, 52.0),
            "h1_rsi_max":      trial.suggest_float("h1_rsi_max",    48.0, 62.0),
            "stoch_bull_cap":  trial.suggest_float("stoch_bull_cap", 75.0, 92.0),
            "stoch_bear_floor":trial.suggest_float("stoch_bear_floor",8.0, 25.0),
            "lon_open":        trial.suggest_int  ("lon_open",       6,  9),
            "lon_close":       trial.suggest_int  ("lon_close",     10, 14),
            "ny_open":         trial.suggest_int  ("ny_open",       11, 14),
            "ny_close":        trial.suggest_int  ("ny_close",      16, 20),
            "monday_lot":      trial.suggest_float("monday_lot",    0.50, 1.00),
            "thursday_lot":    trial.suggest_float("thursday_lot",  0.50, 1.00),
        }
        if p["sl_min_mult"] >= p["sl_atr_mult"] or p["sl_atr_mult"] >= p["sl_max_mult"]: return 0.0
        if p["h1_rsi_min"]  >= p["h1_rsi_max"]:  return 0.0
        try:
            r = run_backtest_fast(df_h4, df_h1, df_m15, p)
        except Exception:
            return 0.0
        if r["max_dd_pct"] > 50 or r["trades"] < 100: return 0.0
        return r["score"]

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best   = study.best_params
    best_r = run_backtest_fast(df_h4, df_h1, df_m15, best)

    print(f"\n  Best score   : {study.best_value:.3f}")
    print(f"\n  BEST PARAMETERS:")
    for k, v in sorted(best.items()):
        print(f"    {k:<25} = {round(v,4) if isinstance(v,float) else v}")
    print(f"\n  BEST BACKTEST RESULT:")
    print(f"    Trades: {best_r['trades']}  WR: {best_r['win_rate']*100:.1f}%  "
          f"Profit: ${best_r['profit']:,.0f}  DD: {best_r['max_dd_pct']:.1f}%  PF: {best_r['profit_factor']:.2f}")

    # Optuna parameter importance
    try:
        pi = optuna.importance.get_param_importances(study)
        print(f"\n  OPTUNA PARAMETER IMPORTANCE:")
        print(f"  {'Parameter':<25} {'Importance':>12}  {'Bar'}")
        print("  " + "-"*55)
        for param, imp in sorted(pi.items(), key=lambda x: x[1], reverse=True):
            bar_viz = "=" * int(imp * 100)
            verdict = "HIGH" if imp > 0.10 else ("MOD" if imp > 0.04 else "low")
            print(f"  {param:<25} {imp:>12.4f}  {bar_viz} {verdict}")
    except Exception as e:
        print(f"  (Parameter importance unavailable: {e})")

    # Save trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(OUTPUT_DIR, "optuna_trials.csv"), index=False)
    print(f"\n  Trials saved: {OUTPUT_DIR}/optuna_trials.csv")

    return best, best_r


# ================================================================
#  WRITE OPTIMAL CONFIG v1.5
# ================================================================

def write_config(best: Dict, best_r: Dict):
    b = best
    content = f'''# ================================================================
#  config_v1.5.py  --  GoldScalperBot  XAUUSD...
#  VERSION 1.5  --  Optuna + Random Forest Optimization
#
#  Generated : {datetime.now().strftime("%Y-%m-%d %H:%M")}
#  Result    : {best_r["trades"]} trades | {best_r["win_rate"]*100:.1f}% WR | ${best_r["profit"]:,.0f} | {best_r["max_dd_pct"]:.1f}% DD
# ================================================================

SYMBOL          = "XAUUSD..."
BROKER_SERVER   = "MEXGlobalFinancial-Demo"
POINT           = 0.01; DIGITS=2; CONTRACT_SIZE=100
PIP_VALUE_STD   = 1.0;  MAX_SPREAD_PTS=20
ACCOUNT_BALANCE = 2000.0; ACCOUNT_LEVERAGE=500

RISK_NEWS_PCT   = 0.005
RISK_NEWS_USD   = ACCOUNT_BALANCE * RISK_NEWS_PCT

MAX_CONCURRENT_TRADES  = {b.get("max_concurrent", 2)}
TOTAL_RISK_BUDGET_PCT  = {b.get("risk_pct",0.01)*b.get("max_concurrent",2):.4f}
RISK_NORMAL_PCT        = TOTAL_RISK_BUDGET_PCT / MAX_CONCURRENT_TRADES
RISK_NORMAL_USD        = ACCOUNT_BALANCE * RISK_NORMAL_PCT

SL_ATR_MULTIPLIER   = {b.get("sl_atr_mult",1.5):.4f}
SL_ATR_MIN_MULT     = {b.get("sl_min_mult",1.0):.4f}
SL_ATR_MAX_MULT     = {b.get("sl_max_mult",2.5):.4f}
TP_RR_RATIO         = {b.get("tp_rr",2.0):.4f}
TP_TRAIL_ATR_MULT   = 1.5
LOT_MIN=0.01; LOT_MAX=10.0; LOT_STEP=0.01; MIN_RR=1.5
COMMISSION_USD      = 0.0

BE_PROFIT_TRIGGER_MULT = {b.get("be_trigger",3.5):.3f}
BE_SL_LOCK_MULT        = {b.get("be_lock",3.0):.3f}

MAX_DAILY_LOSS_PCT   = {b.get("daily_loss_pct",0.04):.4f}
MAX_DAILY_LOSS_USD   = ACCOUNT_BALANCE * MAX_DAILY_LOSS_PCT
MAX_CONSEC_LOSSES    = {b.get("max_consec",4)}
MAX_TRADES_PER_SESS  = 20
WEEKLY_DD_PCT        = {b.get("weekly_dd_pct",0.10):.4f}
WEEKLY_DD_USD        = ACCOUNT_BALANCE * WEEKLY_DD_PCT

LONDON_OPEN_H    = {b.get("lon_open",7)}
LONDON_OPEN_MIN  = 0
LONDON_CLOSE_H   = {b.get("lon_close",12)}
NY_OPEN_H        = {b.get("ny_open",12)}
NY_CLOSE_H       = {b.get("ny_close",18)}

FILTER_VOL_MIN_Z        = {b.get("vol_min_z",0.0):.3f}
FILTER_LONDON_DELAY     = False
FILTER_MONDAY_LOT       = {b.get("monday_lot",0.70):.3f}
FILTER_THURSDAY_LOT     = {b.get("thursday_lot",0.80):.3f}
FILTER_RSI_SLOPE_3BAR   = {b.get("rsi_slope_gate",0.5):.3f}
FILTER_EMA_SLOPE_MAX    = {b.get("ema_slope_cap",1.0):.3f}
H1_RSI_BULL_MIN         = {b.get("h1_rsi_min",45.0):.1f}
H1_RSI_BEAR_MAX         = {b.get("h1_rsi_max",55.0):.1f}
STOCH_BULL_CAP          = {b.get("stoch_bull_cap",85.0):.1f}
STOCH_BEAR_FLOOR        = {b.get("stoch_bear_floor",15.0):.1f}

LOG_VERBOSE=False; LOG_TRADES=True
'''
    with open("config_v1.5.py","w") as f: f.write(content)
    print(f"\n  config_v1.5.py written.")
    print("  To test: copy config_v1.5.py config.py && python quick_diagnostic.py")


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("="*65)
    print("  GoldScalperBot -- ML Optimizer LITE")
    print("  Random Forest + Permutation Importance + Optuna")
    print("  (No XGBoost / No DLL / No dask required)")
    print("="*65)

    if not mt5.initialize():
        print(f"MT5 failed: {mt5.last_error()}"); exit(1)
    a = mt5.account_info()
    if a is None: print("Not logged in"); mt5.shutdown(); exit(1)
    print(f"MT5 connected -- {a.login}  Balance: ${a.balance:,.2f}\n")

    loader = DataLoader(SYMBOL)
    df_h4  = loader.load("H4",  INSAMPLE_START, INSAMPLE_END)
    df_h1  = loader.load("H1",  INSAMPLE_START, INSAMPLE_END)
    df_m15 = loader.load("M15", INSAMPLE_START, INSAMPLE_END)

    def _n(df):
        m={}
        for c in df.columns:
            cl=c.lower()
            if cl=="open": m[c]="Open"
            elif cl=="high": m[c]="High"
            elif cl=="low": m[c]="Low"
            elif cl=="close": m[c]="Close"
            elif cl in("volume","tick_volume"): m[c]="Volume"
            elif cl=="spread": m[c]="Spread"
        return df.rename(columns=m)

    df_h4=_n(df_h4); df_h1=_n(df_h1); df_m15=_n(df_m15)

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
    stk,std              = Indicators.stoch_rsi(df_m15["Close"])
    df_m15["StochK"]=stk; df_m15["StochD"]=std

    # Phase 1: Generate trades + features
    t0 = time.time()
    df_trades = generate_trades(df_h4, df_h1, df_m15)
    df_trades.to_csv(os.path.join(OUTPUT_DIR,"ml_full_features.csv"), index=False)
    print(f"  Feature generation: {time.time()-t0:.1f}s")

    # Phase 2: Random Forest + permutation importance
    t1 = time.time()
    fi_df, essential_feats, rf_model, all_feats = train_and_rank(df_trades)
    print(f"  Model training: {time.time()-t1:.1f}s")

    # Phase 3: Optuna optimization
    N_TRIALS = 200
    print(f"\n  Starting Optuna ({N_TRIALS} trials, ~15-20 mins)...")
    t2 = time.time()
    best_params, best_r = run_optuna(df_h4, df_h1, df_m15, n_trials=N_TRIALS)
    print(f"  Optuna: {time.time()-t2:.1f}s")

    # Phase 4: Write config
    write_config(best_params, best_r)

    mt5.shutdown()
    print("\n" + "="*65)
    print("  DONE. Run feature_selector.py next for full analysis.")
    print("  Then: copy config_v1.5.py config.py")
    print("        python quick_diagnostic.py")
    print("="*65)
