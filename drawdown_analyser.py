# ================================================================
#  drawdown_analyser.py
#  GoldScalperBot — Trade Condition Analyser
#
#  Analyses market conditions during:
#    - Large losing trades (drawdown events)
#    - Large winning trades (profit events)
#
#  For each trade fetches at the moment of entry:
#    - ATR ratio (current vs 50-bar average) — volatility spike
#    - Volume Z-score                         — abnormal activity
#    - Candle body ratio                      — momentum strength
#    - H4 trend direction vs trade direction  — counter-trend check
#    - RSI level                              — overbought/oversold
#    - Spread at entry                        — liquidity conditions
#    - Session (London/NY)
#    - Day of week
#    - Hour of day
#
#  Output:
#    - Console summary table
#    - CSV files for further analysis
#    - PDF report with charts
# ================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone, timedelta
import os
import csv
from config import *

# ----------------------------------------------------------------
#  SETTINGS
# ----------------------------------------------------------------

LOSS_THRESHOLD   = 80.0   # Trades with loss > this are "large losses"
WIN_THRESHOLD    = 150.0  # Trades with win  > this are "large wins"
OUTPUT_DIR       = "analysis_results"
BARS_CONTEXT     = 50     # How many bars before entry to analyse

# ----------------------------------------------------------------
#  INDICATOR HELPERS
# ----------------------------------------------------------------

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period-1, adjust=False).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def volume_zscore(volume, period=20):
    mean = volume.rolling(period).mean()
    std  = volume.rolling(period).std()
    return (volume - mean) / std.replace(0, np.nan)

# ----------------------------------------------------------------
#  DATA FETCHER
# ----------------------------------------------------------------

def get_bars_at(symbol, timeframe, dt, n_bars):
    """
    Fetch n_bars of OHLCV data ending at datetime dt.
    Returns DataFrame or None.
    """
    # Fetch extra bars to ensure we have enough after filtering
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 99999)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
        "spread": "Spread"
    }, inplace=True)

    # Filter to bars up to dt
    mask = df.index <= dt
    subset = df[mask]
    if len(subset) < n_bars:
        return None
    return subset.tail(n_bars)

# ----------------------------------------------------------------
#  CONDITION EXTRACTOR
# ----------------------------------------------------------------

def extract_conditions(row, m15_data, h4_data):
    """
    Given a trade row and market data at entry time,
    extract all relevant market conditions.
    Returns a dict of conditions.
    """
    conditions = {}

    # Basic trade info
    conditions["trade_id"]    = row["trade_id"]
    conditions["direction"]   = row["direction"]
    conditions["entry_price"] = row["entry_price"]
    conditions["exit_price"]  = row["exit_price"]
    conditions["pnl_usd"]     = row["pnl_usd"]
    conditions["exit_reason"] = row["exit_reason"]
    conditions["grade"]       = row["signal_grade"]

    # Time analysis
    open_time = pd.to_datetime(row["open_time"], utc=True)
    conditions["open_time"]   = open_time
    conditions["hour"]        = open_time.hour
    conditions["day_of_week"] = open_time.strftime("%A")
    conditions["session"]     = (
        "London" if 8 <= open_time.hour < 11 else
        "NY"     if 13 <= open_time.hour < 16 else
        "Other"
    )

    if m15_data is None or len(m15_data) < 20:
        return conditions

    # --- M15 indicators ---
    m15 = m15_data.copy()
    m15["ATR14"]   = calc_atr(m15["High"], m15["Low"], m15["Close"], 14)
    m15["ATR50avg"]= m15["ATR14"].rolling(50).mean()
    m15["RSI14"]   = calc_rsi(m15["Close"], 14)
    m15["VolZ"]    = volume_zscore(m15["Volume"], 20)
    m15["BodyRat"] = (m15["Close"] - m15["Open"]).abs() / (m15["High"] - m15["Low"]).replace(0, np.nan)
    m15["EMA20"]   = calc_ema(m15["Close"], 20)

    last_m15 = m15.iloc[-1]

    atr_now  = last_m15["ATR14"]
    atr_avg  = last_m15["ATR50avg"]

    conditions["m15_atr"]       = round(atr_now / POINT, 1) if not pd.isna(atr_now) else None
    conditions["m15_atr_ratio"] = round(atr_now / atr_avg, 2) if (not pd.isna(atr_now) and not pd.isna(atr_avg) and atr_avg > 0) else None
    conditions["m15_rsi"]       = round(last_m15["RSI14"], 1) if not pd.isna(last_m15["RSI14"]) else None
    conditions["m15_vol_z"]     = round(last_m15["VolZ"], 2) if not pd.isna(last_m15["VolZ"]) else None
    conditions["m15_body_ratio"]= round(last_m15["BodyRat"], 2) if not pd.isna(last_m15["BodyRat"]) else None
    conditions["m15_spread"]    = round(last_m15["Spread"], 0) if "Spread" in m15.columns and not pd.isna(last_m15["Spread"]) else None

    # ATR spike flag
    conditions["vol_spike"] = (
        conditions["m15_atr_ratio"] is not None and
        conditions["m15_atr_ratio"] > 2.0
    )

    # Volume spike flag
    conditions["vol_abnormal"] = (
        conditions["m15_vol_z"] is not None and
        abs(conditions["m15_vol_z"]) > 2.0
    )

    # --- H4 trend analysis ---
    if h4_data is not None and len(h4_data) >= 20:
        h4 = h4_data.copy()
        h4["EMA20"] = calc_ema(h4["Close"], 20)
        h4["EMA50"] = calc_ema(h4["Close"], 50)
        h4["ATR14"] = calc_atr(h4["High"], h4["Low"], h4["Close"], 14)
        h4["RSI14"] = calc_rsi(h4["Close"], 14)

        last_h4 = h4.iloc[-1]
        prev_h4 = h4.iloc[-2]

        h4_trend_up = last_h4["EMA20"] > last_h4["EMA50"]
        trade_long  = row["direction"] == "LONG"

        conditions["h4_trend"]          = "UP" if h4_trend_up else "DOWN"
        conditions["h4_ema20"]          = round(last_h4["EMA20"], 2) if not pd.isna(last_h4["EMA20"]) else None
        conditions["h4_rsi"]            = round(last_h4["RSI14"], 1) if not pd.isna(last_h4["RSI14"]) else None
        conditions["h4_atr"]            = round(last_h4["ATR14"] / POINT, 1) if not pd.isna(last_h4["ATR14"]) else None
        conditions["counter_trend"]     = (h4_trend_up != trade_long)
        conditions["h4_trend_strength"] = round(abs(last_h4["EMA20"] - last_h4["EMA50"]) / POINT, 1) if not pd.isna(last_h4["EMA20"]) else None

        # Consecutive same-direction H4 candles (momentum continuation)
        bearish_count = 0
        bullish_count = 0
        for i in range(len(h4) - 1, max(len(h4) - 8, -1), -1):
            if h4["Close"].iloc[i] < h4["Open"].iloc[i]:
                bearish_count += 1
            else:
                break
        for i in range(len(h4) - 1, max(len(h4) - 8, -1), -1):
            if h4["Close"].iloc[i] > h4["Open"].iloc[i]:
                bullish_count += 1
            else:
                break

        conditions["h4_consec_bear"] = bearish_count
        conditions["h4_consec_bull"] = bullish_count

        # Extended move detection: price extended beyond ATR from EMA20
        price_ext = abs(last_h4["Close"] - last_h4["EMA20"]) / last_h4["ATR14"] if last_h4["ATR14"] > 0 else 0
        conditions["price_extension"] = round(price_ext, 2) if not pd.isna(price_ext) else None

    return conditions


# ----------------------------------------------------------------
#  MAIN ANALYSER
# ----------------------------------------------------------------

class TradeConditionAnalyser:

    def __init__(self, symbol=SYMBOL):
        self.symbol = symbol

    def load_trades_from_diagnostic(self):
        """
        Re-run the backtest to get trade data.
        Imports the backtest runner directly.
        """
        print("Loading trade data from backtest...")
        from backtest_engine import (BacktestRunner, INSAMPLE_START,
                                      INSAMPLE_END, DataLoader)

        loader = DataLoader(self.symbol)
        df_h4  = loader.load("H4",  INSAMPLE_START, INSAMPLE_END)
        df_h1  = loader.load("H1",  INSAMPLE_START, INSAMPLE_END)
        df_m15 = loader.load("M15", INSAMPLE_START, INSAMPLE_END)

        runner = BacktestRunner()
        params = {
            "pivot_strength": 5,
            "min_wave_pts":   1200,
            "bos_method":     "single_close",
            "ehl_tolerance":  "atr_0.5"
        }
        result = runner._run_single(df_h4, df_h1, df_m15, params)
        return result.trades

    def analyse(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Load trades
        trades = self.load_trades_from_diagnostic()
        if not trades:
            print("No trades found")
            return

        print(f"Analysing {len(trades)} trades...")
        print(f"  Large losses (> ${LOSS_THRESHOLD}): "
              f"{len([t for t in trades if t.pnl_usd < -LOSS_THRESHOLD])}")
        print(f"  Large wins   (> ${WIN_THRESHOLD}):  "
              f"{len([t for t in trades if t.pnl_usd > WIN_THRESHOLD])}")
        print()

        # Separate into groups
        large_losses = [t for t in trades if t.pnl_usd < -LOSS_THRESHOLD]
        large_wins   = [t for t in trades if t.pnl_usd > WIN_THRESHOLD]
        normal       = [t for t in trades
                        if -LOSS_THRESHOLD <= t.pnl_usd <= WIN_THRESHOLD]

        # Analyse each group
        print("Fetching market conditions for large losses...")
        loss_conditions = self._analyse_group(large_losses, "LOSS")

        print("Fetching market conditions for large wins...")
        win_conditions  = self._analyse_group(large_wins, "WIN")

        print("Building summary statistics...")
        self._print_summary(loss_conditions, win_conditions)

        print("Saving CSV files...")
        self._save_csv(loss_conditions, "large_losses.csv")
        self._save_csv(win_conditions,  "large_wins.csv")

        print("Generating PDF report...")
        self._generate_report(loss_conditions, win_conditions, trades)

        print(f"\nAll analysis saved to: {OUTPUT_DIR}/")

    def _analyse_group(self, trade_list, label):
        """Fetch conditions for each trade in the list"""
        conditions_list = []
        total = len(trade_list)

        for i, trade in enumerate(trade_list):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {label}: {i+1}/{total} trades processed...")

            open_dt = trade.open_time
            if open_dt is None:
                continue

            # Ensure timezone aware
            if open_dt.tzinfo is None:
                open_dt = open_dt.replace(tzinfo=timezone.utc)

            # Fetch M15 and H4 data at entry time
            m15_data = get_bars_at(self.symbol, mt5.TIMEFRAME_M15,
                                    open_dt, BARS_CONTEXT + 20)
            h4_data  = get_bars_at(self.symbol, mt5.TIMEFRAME_H4,
                                    open_dt, 60)

            # Build trade row dict
            row = {
                "trade_id":    trade.trade_id,
                "direction":   trade.direction,
                "open_time":   open_dt,
                "entry_price": trade.entry_price,
                "exit_price":  trade.exit_price,
                "pnl_usd":     trade.pnl_usd,
                "exit_reason": trade.exit_reason,
                "signal_grade":trade.signal_grade,
            }

            cond = extract_conditions(row, m15_data, h4_data)
            cond["group"] = label
            conditions_list.append(cond)

        return conditions_list

    def _print_summary(self, loss_cond, win_cond):
        """Print comparison summary to console"""

        def safe_mean(lst, key):
            vals = [c[key] for c in lst
                    if c.get(key) is not None and not pd.isna(c[key])]
            return round(np.mean(vals), 2) if vals else "N/A"

        def safe_pct(lst, key):
            vals = [c[key] for c in lst if c.get(key) is not None]
            if not vals:
                return "N/A"
            return f"{sum(1 for v in vals if v) / len(vals) * 100:.0f}%"

        print()
        print("=" * 65)
        print("  MARKET CONDITIONS COMPARISON")
        print(f"  Large Losses (>{LOSS_THRESHOLD}) vs Large Wins (>{WIN_THRESHOLD})")
        print("=" * 65)
        print(f"  {'Metric':<30} {'LOSSES':>12} {'WINS':>12}")
        print("-" * 65)

        metrics = [
            ("Count",             len(loss_cond),          len(win_cond)),
            ("Avg P&L ($)",       safe_mean(loss_cond,"pnl_usd"),
                                  safe_mean(win_cond,"pnl_usd")),
            ("M15 ATR (pts)",     safe_mean(loss_cond,"m15_atr"),
                                  safe_mean(win_cond,"m15_atr")),
            ("M15 ATR ratio",     safe_mean(loss_cond,"m15_atr_ratio"),
                                  safe_mean(win_cond,"m15_atr_ratio")),
            ("M15 RSI",           safe_mean(loss_cond,"m15_rsi"),
                                  safe_mean(win_cond,"m15_rsi")),
            ("Volume Z-score",    safe_mean(loss_cond,"m15_vol_z"),
                                  safe_mean(win_cond,"m15_vol_z")),
            ("Body ratio",        safe_mean(loss_cond,"m15_body_ratio"),
                                  safe_mean(win_cond,"m15_body_ratio")),
            ("Spread (pts)",      safe_mean(loss_cond,"m15_spread"),
                                  safe_mean(win_cond,"m15_spread")),
            ("H4 ATR (pts)",      safe_mean(loss_cond,"h4_atr"),
                                  safe_mean(win_cond,"h4_atr")),
            ("H4 RSI",            safe_mean(loss_cond,"h4_rsi"),
                                  safe_mean(win_cond,"h4_rsi")),
            ("Price extension",   safe_mean(loss_cond,"price_extension"),
                                  safe_mean(win_cond,"price_extension")),
            ("Counter-trend %",   safe_pct(loss_cond,"counter_trend"),
                                  safe_pct(win_cond,"counter_trend")),
            ("Volatility spike %",safe_pct(loss_cond,"vol_spike"),
                                  safe_pct(win_cond,"vol_spike")),
            ("Volume abnormal %", safe_pct(loss_cond,"vol_abnormal"),
                                  safe_pct(win_cond,"vol_abnormal")),
        ]

        for label, l_val, w_val in metrics:
            print(f"  {label:<30} {str(l_val):>12} {str(w_val):>12}")

        print("-" * 65)

        # Session breakdown
        print()
        print("  SESSION BREAKDOWN:")
        for session in ["London", "NY"]:
            l_ct = sum(1 for c in loss_cond if c.get("session") == session)
            w_ct = sum(1 for c in win_cond  if c.get("session") == session)
            print(f"  {session:<30} {str(l_ct):>12} {str(w_ct):>12}")

        # Day of week breakdown
        print()
        print("  DAY OF WEEK BREAKDOWN:")
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        for day in days:
            l_ct = sum(1 for c in loss_cond if c.get("day_of_week") == day)
            w_ct = sum(1 for c in win_cond  if c.get("day_of_week") == day)
            if l_ct > 0 or w_ct > 0:
                print(f"  {day:<30} {str(l_ct):>12} {str(w_ct):>12}")

        # Hour of day breakdown
        print()
        print("  HOUR OF DAY BREAKDOWN (losses):")
        hour_losses = {}
        for c in loss_cond:
            h = c.get("hour")
            if h is not None:
                hour_losses[h] = hour_losses.get(h, 0) + 1
        for h in sorted(hour_losses):
            bar = "█" * hour_losses[h]
            print(f"  {h:02d}:00  {bar} ({hour_losses[h]})")

        print()
        print("  HOUR OF DAY BREAKDOWN (wins):")
        hour_wins = {}
        for c in win_cond:
            h = c.get("hour")
            if h is not None:
                hour_wins[h] = hour_wins.get(h, 0) + 1
        for h in sorted(hour_wins):
            bar = "█" * hour_wins[h]
            print(f"  {h:02d}:00  {bar} ({hour_wins[h]})")

        print("=" * 65)

    def _save_csv(self, conditions_list, filename):
        if not conditions_list:
            return
        path = os.path.join(OUTPUT_DIR, filename)
        keys = [k for k in conditions_list[0].keys()
                if k != "open_time"]
        keys = ["open_time"] + keys

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys,
                                     extrasaction="ignore")
            writer.writeheader()
            for c in conditions_list:
                row = {}
                for k in keys:
                    v = c.get(k, "")
                    if isinstance(v, (datetime, pd.Timestamp)):
                        row[k] = str(v)
                    else:
                        row[k] = v
                writer.writerow(row)
        print(f"  Saved: {path}")

    def _generate_report(self, loss_cond, win_cond, all_trades):
        """Generate a PDF report with charts"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                             Spacer, Image, PageBreak)
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.colors import HexColor
            import io

            path = os.path.join(OUTPUT_DIR, "trade_analysis_report.pdf")
            doc  = SimpleDocTemplate(path, pagesize=A4,
                                      leftMargin=2*cm, rightMargin=2*cm,
                                      topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story  = []

            def fig_img(fig, w=17, h=8):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120,
                            bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                buf.seek(0)
                plt.close(fig)
                return Image(buf, width=w*cm, height=h*cm)

            sty_h = styles["Heading1"]
            sty_b = styles["Normal"]

            # Cover
            story.append(Paragraph(
                "GoldScalperBot — Trade Condition Analysis", sty_h))
            story.append(Paragraph(
                f"Symbol: {SYMBOL}  |  "
                f"Large losses threshold: ${LOSS_THRESHOLD}  |  "
                f"Large wins threshold: ${WIN_THRESHOLD}", sty_b))
            story.append(Paragraph(
                f"Large losses: {len(loss_cond)}  |  "
                f"Large wins: {len(win_cond)}", sty_b))
            story.append(Spacer(1, 20))

            # ---- Chart 1: ATR ratio comparison ----
            story.append(Paragraph("ATR Ratio at Entry (current ATR / 50-bar average)", sty_h))
            story.append(Paragraph(
                "ATR ratio > 2.0 = volatility spike. "
                "High ratio during losses = entered during dangerous conditions.", sty_b))

            l_atr = [c["m15_atr_ratio"] for c in loss_cond
                     if c.get("m15_atr_ratio") is not None]
            w_atr = [c["m15_atr_ratio"] for c in win_cond
                     if c.get("m15_atr_ratio") is not None]

            fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                                      facecolor="#0f1117")
            for ax, data, label, color in [
                (axes[0], l_atr, f"Large Losses (n={len(l_atr)})", "#dc2626"),
                (axes[1], w_atr, f"Large Wins (n={len(w_atr)})",   "#16a34a")
            ]:
                ax.set_facecolor("#1a1a2e")
                if data:
                    ax.hist(data, bins=20, color=color, alpha=0.8)
                    ax.axvline(np.mean(data), color="white",
                               linestyle="--", linewidth=1,
                               label=f"Mean: {np.mean(data):.2f}")
                    ax.axvline(2.0, color="orange",
                               linestyle=":", linewidth=1.5,
                               label="Spike threshold (2.0)")
                    ax.legend(fontsize=8, labelcolor="white")
                ax.set_title(label, color="white", fontsize=9)
                ax.set_xlabel("ATR ratio", color="#94a3b8", fontsize=8)
                ax.tick_params(colors="#94a3b8")
            fig.tight_layout()
            story.append(fig_img(fig, w=17, h=6))
            story.append(Spacer(1, 10))

            # ---- Chart 2: Volume Z-score ----
            story.append(Paragraph("Volume Z-Score at Entry", sty_h))
            story.append(Paragraph(
                "Z-score > 2.0 = abnormal volume spike. "
                "Compare distribution between losses and wins.", sty_b))

            l_vz = [c["m15_vol_z"] for c in loss_cond
                    if c.get("m15_vol_z") is not None]
            w_vz = [c["m15_vol_z"] for c in win_cond
                    if c.get("m15_vol_z") is not None]

            fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                                      facecolor="#0f1117")
            for ax, data, label, color in [
                (axes[0], l_vz, f"Large Losses (n={len(l_vz)})", "#dc2626"),
                (axes[1], w_vz, f"Large Wins (n={len(w_vz)})",   "#16a34a")
            ]:
                ax.set_facecolor("#1a1a2e")
                if data:
                    ax.hist(data, bins=20, color=color, alpha=0.8)
                    ax.axvline(np.mean(data), color="white",
                               linestyle="--", linewidth=1,
                               label=f"Mean: {np.mean(data):.2f}")
                    ax.legend(fontsize=8, labelcolor="white")
                ax.set_title(label, color="white", fontsize=9)
                ax.set_xlabel("Volume Z-score", color="#94a3b8", fontsize=8)
                ax.tick_params(colors="#94a3b8")
            fig.tight_layout()
            story.append(fig_img(fig, w=17, h=6))

            story.append(PageBreak())

            # ---- Chart 3: Hour of day heatmap ----
            story.append(Paragraph("Hour of Day Distribution", sty_h))

            hours = list(range(6, 18))
            l_hours = [c.get("hour") for c in loss_cond
                       if c.get("hour") is not None]
            w_hours = [c.get("hour") for c in win_cond
                       if c.get("hour") is not None]

            l_counts = [l_hours.count(h) for h in hours]
            w_counts = [w_hours.count(h) for h in hours]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5),
                                            facecolor="#0f1117")
            for ax, counts, label, color in [
                (ax1, l_counts, "Large Losses by hour", "#dc2626"),
                (ax2, w_counts, "Large Wins by hour",   "#16a34a")
            ]:
                ax.set_facecolor("#1a1a2e")
                ax.bar(hours, counts, color=color, alpha=0.8)
                ax.set_xticks(hours)
                ax.set_xticklabels([f"{h:02d}:00" for h in hours],
                                    rotation=45, fontsize=7,
                                    color="#94a3b8")
                ax.set_title(label, color="white", fontsize=9)
                ax.tick_params(colors="#94a3b8")
                ax.axvline(11, color="yellow", linestyle=":",
                           linewidth=1, alpha=0.5, label="London close")
                ax.axvline(13, color="cyan",   linestyle=":",
                           linewidth=1, alpha=0.5, label="NY open")
                ax.legend(fontsize=7, labelcolor="white")
            fig.tight_layout()
            story.append(fig_img(fig, w=17, h=7))
            story.append(Spacer(1, 10))

            # ---- Chart 4: H4 trend direction ----
            story.append(Paragraph("Counter-Trend Entry Analysis", sty_h))

            l_ct = sum(1 for c in loss_cond if c.get("counter_trend"))
            l_wt = len(loss_cond) - l_ct
            w_ct = sum(1 for c in win_cond  if c.get("counter_trend"))
            w_wt = len(win_cond) - w_ct

            fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                                      facecolor="#0f1117")
            for ax, wt, ct, title in [
                (axes[0], l_wt, l_ct, "Large Losses"),
                (axes[1], w_wt, w_ct, "Large Wins")
            ]:
                ax.set_facecolor("#1a1a2e")
                if wt + ct > 0:
                    ax.pie([wt, ct],
                           labels=["With trend", "Counter trend"],
                           colors=["#16a34a", "#dc2626"],
                           autopct="%1.0f%%",
                           textprops={"color":"white","fontsize":9},
                           startangle=90)
                ax.set_title(title, color="white", fontsize=10)
            fig.tight_layout()
            story.append(fig_img(fig, w=17, h=5))

            story.append(PageBreak())

            # ---- Chart 5: Price extension ----
            story.append(Paragraph("Price Extension at Entry (distance from H4 EMA20 in ATR units)", sty_h))
            story.append(Paragraph(
                "High extension = entering when price is already far from equilibrium. "
                "A value > 2 means price is 2x ATR away from EMA20 — overextended.", sty_b))

            l_ext = [c["price_extension"] for c in loss_cond
                     if c.get("price_extension") is not None]
            w_ext = [c["price_extension"] for c in win_cond
                     if c.get("price_extension") is not None]

            fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                                      facecolor="#0f1117")
            for ax, data, label, color in [
                (axes[0], l_ext, f"Large Losses (n={len(l_ext)})", "#dc2626"),
                (axes[1], w_ext, f"Large Wins (n={len(w_ext)})",   "#16a34a")
            ]:
                ax.set_facecolor("#1a1a2e")
                if data:
                    ax.hist(data, bins=20, color=color, alpha=0.8)
                    ax.axvline(np.mean(data), color="white",
                               linestyle="--", linewidth=1,
                               label=f"Mean: {np.mean(data):.2f}")
                    ax.axvline(2.0, color="orange",
                               linestyle=":", linewidth=1.5,
                               label="Overextended threshold (2.0)")
                    ax.legend(fontsize=8, labelcolor="white")
                ax.set_title(label, color="white", fontsize=9)
                ax.set_xlabel("Extension (ATR units)", color="#94a3b8",
                              fontsize=8)
                ax.tick_params(colors="#94a3b8")
            fig.tight_layout()
            story.append(fig_img(fig, w=17, h=6))
            story.append(Spacer(1, 10))

            # ---- Chart 6: H4 consecutive candles ----
            story.append(Paragraph("H4 Consecutive Same-Direction Candles at Entry", sty_h))
            story.append(Paragraph(
                "Many consecutive candles in the same direction = momentum continuation risk. "
                "High count = entering into an already extended move.", sty_b))

            l_bear = [c.get("h4_consec_bear", 0) for c in loss_cond]
            l_bull = [c.get("h4_consec_bull", 0) for c in loss_cond]
            w_bear = [c.get("h4_consec_bear", 0) for c in win_cond]
            w_bull = [c.get("h4_consec_bull", 0) for c in win_cond]

            fig, axes = plt.subplots(2, 2, figsize=(11, 6),
                                      facecolor="#0f1117")
            pairs = [
                (axes[0,0], l_bull, "Losses: Bull H4 candles", "#dc2626"),
                (axes[0,1], l_bear, "Losses: Bear H4 candles", "#dc2626"),
                (axes[1,0], w_bull, "Wins: Bull H4 candles",   "#16a34a"),
                (axes[1,1], w_bear, "Wins: Bear H4 candles",   "#16a34a"),
            ]
            for ax, data, title, color in pairs:
                ax.set_facecolor("#1a1a2e")
                if data:
                    counts = [data.count(i) for i in range(8)]
                    ax.bar(range(8), counts, color=color, alpha=0.8)
                    ax.set_xticks(range(8))
                    ax.set_title(title, color="white", fontsize=8)
                    ax.tick_params(colors="#94a3b8", labelsize=7)
            fig.tight_layout()
            story.append(fig_img(fig, w=17, h=7))

            # Build PDF
            doc.build(story)
            print(f"  PDF saved: {path}")

        except Exception as e:
            print(f"  PDF generation error: {e}")
            print("  (CSV files still saved)")


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  GoldScalperBot — Trade Condition Analyser")
    print("=" * 55)
    print(f"  Symbol: {SYMBOL}")
    print(f"  Loss threshold : > ${LOSS_THRESHOLD}")
    print(f"  Win  threshold : > ${WIN_THRESHOLD}")
    print("=" * 55)

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        print("Make sure MT5 is open and connected")
        exit(1)

    account = mt5.account_info()
    if account is None:
        print("MT5 not logged in")
        mt5.shutdown()
        exit(1)

    print(f"MT5 connected — {account.login}")
    print()

    analyser = TradeConditionAnalyser(SYMBOL)
    analyser.analyse()

    mt5.shutdown()
    print("Done.")
