# ================================================================
#  trade_analyser_excel.py  v1.0
#  GoldScalperBot XAUUSD... — Excel Trade Analysis Report
#
#  Generates a multi-sheet Excel workbook:
#    Sheet 1 — Summary dashboard (key metrics + KPI cards)
#    Sheet 2 — Full trade log (all columns including duration)
#    Sheet 3 — Monthly performance breakdown
#    Sheet 4 — Hourly/session performance heatmap
#    Sheet 5 — Charts (equity curve, monthly P&L, win rate by hour)
#    Sheet 6 — Duration analysis (trade lifetime insights)
#
#  Usage:
#    python trade_analyser_excel.py
#  Output:
#    backtest_results/trade_analysis_v1.0.xlsx
# ================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, os
from datetime import datetime, timezone
from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                               GradientFill)
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.series import DataPoint
from openpyxl.drawing.image import Image as XLImage

from backtest_engine import (BacktestRunner, INSAMPLE_START, INSAMPLE_END,
                              DataLoader, ResultAnalyser)
from config import *

OUTPUT_DIR  = "backtest_results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "trade_analysis_v1.0.xlsx")

# ── COLOURS ──────────────────────────────────────────────────────
C_HEADER_DARK  = "1A1A2E"   # Dark navy
C_HEADER_MID   = "0F3460"   # Deep blue
C_GOLD         = "D97706"   # Gold accent
C_GREEN_DARK   = "166534"
C_GREEN_LIGHT  = "DCFCE7"
C_RED_DARK     = "991B1B"
C_RED_LIGHT    = "FEE2E2"
C_AMBER_LIGHT  = "FEF3C7"
C_AMBER_DARK   = "92400E"
C_LIGHT_BG     = "F8FAFC"
C_BORDER       = "E2E8F0"
C_WHITE        = "FFFFFF"

def fill(hex_color):
    return PatternFill("solid", start_color=hex_color, fgColor=hex_color)

def font(bold=False, size=10, color="000000", name="Arial"):
    return Font(bold=bold, size=size, color=color, name=name)

def border_all(color=C_BORDER):
    s = Side(style="thin", color=color)
    return Border(left=s, right=s, top=s, bottom=s)

def center():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)

def right_align():
    return Alignment(horizontal="right", vertical="center")

# ── MATPLOTLIB CHARTS ────────────────────────────────────────────
CHART_STYLE = {
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#334155", "axes.labelcolor": "#94a3b8",
    "axes.grid": True, "grid.color": "#1e293b", "grid.linewidth": 0.5,
    "text.color": "#e2e8f0", "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8", "lines.linewidth": 1.5, "font.size": 9,
}

def fig_to_xl_image(fig, width_cm=22, height_cm=10):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    img = XLImage(buf)
    img.width  = int(width_cm * 37.795)
    img.height = int(height_cm * 37.795)
    return img

def build_equity_chart(equity_curve, starting_balance):
    with plt.style.context(CHART_STYLE):
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(14, 6), facecolor="#0f1117")
        gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        eq  = np.array(equity_curve); x = range(len(eq))
        ax1.fill_between(x, starting_balance, eq,
                         where=(eq >= starting_balance), color="#16a34a", alpha=0.3)
        ax1.fill_between(x, starting_balance, eq,
                         where=(eq < starting_balance),  color="#dc2626", alpha=0.3)
        ax1.plot(x, eq, color="#d97706", linewidth=1.5)
        ax1.axhline(starting_balance, color="#64748b", linewidth=0.8,
                    linestyle="--", alpha=0.6)
        ax1.set_ylabel("Equity ($)", color="#94a3b8", fontsize=9)
        ax1.set_title("Equity curve", color="#e2e8f0", fontsize=10, pad=8)
        ax1.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        ax1.set_xlim(0, max(1, len(eq)-1))
        ax1.tick_params(labelbottom=False)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        peak = np.maximum.accumulate(eq)
        dd   = ((eq - peak) / peak) * 100
        ax2.fill_between(x, dd, 0, color="#dc2626", alpha=0.4)
        ax2.plot(x, dd, color="#dc2626", linewidth=1.0)
        ax2.set_ylabel("DD %", color="#94a3b8", fontsize=9)
        ax2.set_xlabel("Trade #", color="#94a3b8", fontsize=9)
        ax2.set_xlim(0, max(1, len(eq)-1))
        fig.tight_layout(pad=1.2)
    return fig_to_xl_image(fig, 22, 9)

def build_monthly_chart(df):
    df["month"] = pd.to_datetime(df["close_time"]).dt.strftime("%Y-%m")
    mnl = df.groupby("month")["pnl_usd"].sum().reset_index()
    with plt.style.context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f1117")
        colors  = ["#16a34a" if v >= 0 else "#dc2626" for v in mnl["pnl_usd"]]
        ax.bar(mnl["month"], mnl["pnl_usd"], color=colors, width=0.7, zorder=3)
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.set_title("Monthly P&L ($)", color="#e2e8f0", fontsize=10, pad=8)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
        fig.tight_layout(pad=1.2)
    return fig_to_xl_image(fig, 22, 7)

def build_hourly_chart(df):
    hours  = sorted(df["hour"].unique())
    wr_by_h= df.groupby("hour").apply(
        lambda x: (x["pnl_usd"]>0).mean()*100).reindex(hours, fill_value=0)
    ct_by_h= df.groupby("hour").size().reindex(hours, fill_value=0)
    with plt.style.context(CHART_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
        colors = ["#16a34a" if w>=50 else "#dc2626" for w in wr_by_h]
        ax1.bar([f"{h:02d}:00" for h in hours], wr_by_h, color=colors, zorder=3)
        ax1.axhline(50, color="#64748b", linewidth=0.8, linestyle="--")
        ax1.set_title("Win rate % by entry hour", color="#e2e8f0", fontsize=9, pad=8)
        ax1.set_ylabel("Win rate %", color="#94a3b8", fontsize=8)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
        ax2.bar([f"{h:02d}:00" for h in hours], ct_by_h, color="#d97706", zorder=3)
        ax2.set_title("Trade count by entry hour", color="#e2e8f0", fontsize=9, pad=8)
        ax2.set_ylabel("Trades", color="#94a3b8", fontsize=8)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
        fig.tight_layout(pad=1.5)
    return fig_to_xl_image(fig, 22, 7)

def build_duration_chart(df):
    df["duration_h"] = df["duration_min"] / 60
    with plt.style.context(CHART_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f1117")
        wins   = df[df["pnl_usd"]>0]["duration_h"]
        losses = df[df["pnl_usd"]<=0]["duration_h"]
        axes[0].hist(wins,   bins=30, color="#16a34a", alpha=0.8, label="Wins")
        axes[0].hist(losses, bins=30, color="#dc2626", alpha=0.8, label="Losses")
        axes[0].set_title("Duration distribution", color="#e2e8f0", fontsize=9, pad=6)
        axes[0].set_xlabel("Hours", color="#94a3b8", fontsize=8)
        axes[0].legend(fontsize=8)
        axes[1].scatter(df[df["pnl_usd"]>0]["duration_h"],
                         df[df["pnl_usd"]>0]["pnl_usd"],
                         color="#16a34a", alpha=0.5, s=15, label="Wins")
        axes[1].scatter(df[df["pnl_usd"]<=0]["duration_h"],
                         df[df["pnl_usd"]<=0]["pnl_usd"],
                         color="#dc2626", alpha=0.5, s=15, label="Losses")
        axes[1].axhline(0, color="#64748b", linewidth=0.8)
        axes[1].set_title("Duration vs P&L", color="#e2e8f0", fontsize=9, pad=6)
        axes[1].set_xlabel("Hours alive", color="#94a3b8", fontsize=8)
        axes[1].set_ylabel("P&L ($)", color="#94a3b8", fontsize=8)
        axes[1].legend(fontsize=8)
        dur_buckets = pd.cut(df["duration_h"],
                              bins=[0,1,2,4,8,24,999],
                              labels=["<1h","1-2h","2-4h","4-8h","8-24h",">24h"])
        bucket_pnl = df.groupby(dur_buckets, observed=True)["pnl_usd"].mean()
        colors_b   = ["#16a34a" if v>=0 else "#dc2626" for v in bucket_pnl]
        axes[2].bar(bucket_pnl.index.astype(str), bucket_pnl, color=colors_b, zorder=3)
        axes[2].axhline(0, color="#64748b", linewidth=0.8)
        axes[2].set_title("Avg P&L by duration bucket", color="#e2e8f0", fontsize=9, pad=6)
        axes[2].set_ylabel("Avg P&L ($)", color="#94a3b8", fontsize=8)
        for ax in axes: ax.set_facecolor("#1a1a2e"); ax.tick_params(colors="#94a3b8", labelsize=7)
        fig.tight_layout(pad=1.5)
    return fig_to_xl_image(fig, 24, 7)

# ── SHEET BUILDERS ───────────────────────────────────────────────

def set_col_width(ws, col, width):
    ws.column_dimensions[get_column_letter(col)].width = width

def header_row(ws, row, headers, col_widths=None, bg=C_HEADER_MID):
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font      = font(bold=True, size=9, color=C_WHITE)
        cell.fill      = fill(bg)
        cell.alignment = center()
        cell.border    = border_all()
        if col_widths and c-1 < len(col_widths):
            set_col_width(ws, c, col_widths[c-1])

def data_cell(ws, row, col, value, fmt=None, bold=False,
              bg=C_WHITE, fg="000000", align="left"):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font      = font(bold=bold, size=9, color=fg)
    cell.fill      = fill(bg)
    cell.border    = border_all()
    cell.alignment = Alignment(horizontal=align, vertical="center")
    if fmt: cell.number_format = fmt
    return cell


def build_summary_sheet(wb, result, df):
    ws = wb.create_sheet("Summary")
    ws.sheet_view.showGridLines = False

    # Title band
    ws.merge_cells("A1:L1")
    c = ws["A1"]
    c.value = "GoldScalperBot v1.0 — XAUUSD...  Trade Analysis Report"
    c.font  = font(bold=True, size=14, color=C_WHITE)
    c.fill  = fill(C_HEADER_DARK)
    c.alignment = center()
    ws.row_dimensions[1].height = 30

    ws.merge_cells("A2:L2")
    c = ws["A2"]
    c.value = (f"In-sample: Feb 2022 – Dec 2024  |  "
               f"Total trades: {result.total_trades}  |  "
               f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.font  = font(size=9, color="94a3b8")
    c.fill  = fill(C_HEADER_DARK)
    c.alignment = center()
    ws.row_dimensions[2].height = 18

    # KPI cards — row 4+
    ws.row_dimensions[3].height = 10

    kpis = [
        ("Total Trades",    result.total_trades,          None,     C_WHITE,       "000000"),
        ("Win Rate",        result.win_rate,               "0.0%",   C_GREEN_LIGHT, C_GREEN_DARK),
        ("Net Profit",      result.net_profit_usd,         '$#,##0.00', C_GREEN_LIGHT, C_GREEN_DARK),
        ("Max Drawdown",    result.max_drawdown_pct/100,   "0.0%",   C_RED_LIGHT,   C_RED_DARK),
        ("Profit Factor",   result.profit_factor,          "0.00",   C_AMBER_LIGHT, C_AMBER_DARK),
        ("Score",           result.score,                  "0.000",  C_WHITE,       "000000"),
        ("Avg Win ($)",     result.avg_win_usd,            '$#,##0.00', C_GREEN_LIGHT, C_GREEN_DARK),
        ("Avg Loss ($)",    abs(result.avg_loss_usd),      '$#,##0.00', C_RED_LIGHT,   C_RED_DARK),
        ("Sharpe Ratio",    result.sharpe_ratio,           "0.00",   C_WHITE,       "000000"),
        ("Total Winning",   result.winning_trades,         None,     C_GREEN_LIGHT, C_GREEN_DARK),
        ("Total Losing",    result.losing_trades,          None,     C_RED_LIGHT,   C_RED_DARK),
        ("Avg Trade ($)",   result.net_profit_usd / max(1, result.total_trades), '$#,##0.00', C_WHITE, "000000"),
    ]

    col = 1
    for label, value, fmt, bg, fg in kpis:
        ws.merge_cells(start_row=4, start_column=col,
                        end_row=4,   end_column=col)
        ws.merge_cells(start_row=5, start_column=col,
                        end_row=5,   end_column=col)

        lc = ws.cell(row=4, column=col, value=label)
        lc.font = font(size=8, color="475569"); lc.fill = fill(C_LIGHT_BG)
        lc.alignment = center(); lc.border = border_all()

        vc = ws.cell(row=5, column=col, value=value)
        vc.font = font(bold=True, size=12, color=fg); vc.fill = fill(bg)
        vc.alignment = center(); vc.border = border_all()
        if fmt: vc.number_format = fmt

        set_col_width(ws, col, 14)
        ws.row_dimensions[4].height = 18
        ws.row_dimensions[5].height = 28
        col += 1

    # Session summary
    ws.row_dimensions[6].height = 10
    sessions = ["London (08:00-11:00)", "NY (13:00-16:00)"]
    session_data = []
    for s in sessions:
        h_start = 8 if "London" in s else 13
        h_end   = 11 if "London" in s else 16
        sub = df[(df["hour"] >= h_start) & (df["hour"] < h_end)]
        wins = (sub["pnl_usd"]>0).sum()
        wr   = wins/len(sub) if len(sub)>0 else 0
        pnl  = sub["pnl_usd"].sum()
        session_data.append([s, len(sub), f"{wr*100:.1f}%",
                              f"${pnl:,.2f}", f"${sub['pnl_usd'].mean():,.2f}"])

    header_row(ws, 7, ["Session","Trades","Win Rate","Total P&L","Avg P&L"],
               [22,10,12,14,12], bg=C_HEADER_MID)
    for r, row in enumerate(session_data, 8):
        for c, val in enumerate(row, 1):
            data_cell(ws, r, c, val, align="center",
                      bg=C_GREEN_LIGHT if "London" in row[0] else C_LIGHT_BG)

    # Day of week summary
    ws.row_dimensions[10].height = 10
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    header_row(ws, 11,
               ["Day","Trades","Wins","Losses","Win Rate","Avg P&L ($)","Total P&L ($)"],
               [14,10,8,8,12,14,14], bg=C_HEADER_MID)
    df["dow"] = pd.to_datetime(df["open_time"]).dt.day_name()
    for r, day in enumerate(days, 12):
        sub  = df[df["dow"]==day]
        wins = (sub["pnl_usd"]>0).sum()
        wr   = wins/len(sub) if len(sub)>0 else 0
        bg   = C_GREEN_LIGHT if wr>=0.5 else C_RED_LIGHT
        data_cell(ws, r, 1, day,           align="center", bg=bg)
        data_cell(ws, r, 2, len(sub),      align="center", bg=bg)
        data_cell(ws, r, 3, wins,          align="center", bg=bg)
        data_cell(ws, r, 4, len(sub)-wins, align="center", bg=bg)
        data_cell(ws, r, 5, round(wr,3),   fmt="0.0%", align="center", bg=bg)
        data_cell(ws, r, 6, round(sub["pnl_usd"].mean(),2) if len(sub)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=bg)
        data_cell(ws, r, 7, round(sub["pnl_usd"].sum(),2),
                  fmt="$#,##0.00", align="right", bg=bg)

    # Grade breakdown
    ws.row_dimensions[17].height = 10
    header_row(ws, 18,
               ["Grade","Trades","Win Rate","Avg Win ($)","Avg Loss ($)","Total P&L ($)"],
               [10,10,12,14,14,14], bg=C_HEADER_DARK)
    for r, grade in enumerate(["A+","B"], 19):
        sub  = df[df["signal_grade"]==grade]
        wins = sub[sub["pnl_usd"]>0]; losses = sub[sub["pnl_usd"]<=0]
        wr   = len(wins)/len(sub) if len(sub)>0 else 0
        bg   = C_AMBER_LIGHT if grade=="A+" else C_LIGHT_BG
        data_cell(ws, r, 1, grade, bold=True, align="center", bg=bg)
        data_cell(ws, r, 2, len(sub),  align="center", bg=bg)
        data_cell(ws, r, 3, round(wr,3), fmt="0.0%", align="center", bg=bg)
        data_cell(ws, r, 4, round(wins["pnl_usd"].mean(),2) if len(wins)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=bg)
        data_cell(ws, r, 5, round(losses["pnl_usd"].mean(),2) if len(losses)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=bg)
        data_cell(ws, r, 6, round(sub["pnl_usd"].sum(),2),
                  fmt="$#,##0.00", align="right", bg=bg)

    return ws


def build_trade_log_sheet(wb, df):
    ws = wb.create_sheet("Trade Log")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A2"

    cols = [
        ("#",            6),  ("Direction",      10),
        ("Open Time",    18), ("Close Time",      18),
        ("Duration",     12), ("Entry",           10),
        ("Exit",         10), ("SL",              10),
        ("TP",           10), ("Lots",             8),
        ("Grade",         8), ("P&L ($)",         12),
        ("Exit Reason",  14), ("Cumulative P&L",  16),
        ("DD from Peak", 14),
    ]
    header_row(ws, 1, [c[0] for c in cols],
               [c[1] for c in cols], bg=C_HEADER_DARK)

    cum_pnl = 0.0
    peak    = ACCOUNT_BALANCE

    for i, row in df.iterrows():
        r    = i + 2
        pnl  = row["pnl_usd"]
        cum_pnl += pnl
        peak     = max(peak, ACCOUNT_BALANCE + cum_pnl)
        dd_pct   = ((peak - (ACCOUNT_BALANCE + cum_pnl)) / peak) * 100

        bg = C_GREEN_LIGHT if pnl > 0 else C_RED_LIGHT if pnl < 0 else C_WHITE
        fg = C_GREEN_DARK  if pnl > 0 else C_RED_DARK  if pnl < 0 else "000000"

        data_cell(ws, r,  1, row["trade_id"],    align="center")
        data_cell(ws, r,  2, row["direction"],   align="center",
                  bg=("E0F2FE" if row["direction"]=="LONG" else "FEF2F2"))
        data_cell(ws, r,  3, str(row["open_time"])[:16], align="center")
        data_cell(ws, r,  4, str(row["close_time"])[:16] if row["close_time"] else "", align="center")
        data_cell(ws, r,  5, row["duration_str"], align="center")
        data_cell(ws, r,  6, row["entry_price"],  fmt="0.00", align="right")
        data_cell(ws, r,  7, row["exit_price"],   fmt="0.00", align="right")
        data_cell(ws, r,  8, row["sl_price"],     fmt="0.00", align="right")
        data_cell(ws, r,  9, row["tp_price"],     fmt="0.00", align="right")
        data_cell(ws, r, 10, row["lot_size"],     fmt="0.00", align="center")
        data_cell(ws, r, 11, row["signal_grade"], align="center",
                  bg=C_AMBER_LIGHT if row["signal_grade"]=="A+" else C_WHITE)
        data_cell(ws, r, 12, pnl, fmt="$#,##0.00", align="right",
                  bold=True, bg=bg, fg=fg)
        data_cell(ws, r, 13, row["exit_reason"],  align="center")
        data_cell(ws, r, 14, round(cum_pnl, 2),   fmt="$#,##0.00", align="right",
                  bg=C_GREEN_LIGHT if cum_pnl>0 else C_RED_LIGHT)
        data_cell(ws, r, 15, round(dd_pct, 2),    fmt="0.00%", align="right",
                  bg=C_RED_LIGHT if dd_pct>5 else C_WHITE)

    # Auto filter
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(df)+1}"
    return ws


def build_monthly_sheet(wb, df):
    ws = wb.create_sheet("Monthly P&L")
    ws.sheet_view.showGridLines = False

    df["ym"] = pd.to_datetime(df["open_time"]).dt.strftime("%Y-%m")
    months   = sorted(df["ym"].unique())

    header_row(ws, 1,
               ["Month","Trades","Wins","Losses","Win Rate",
                "Total P&L","Avg P&L","Best Trade","Worst Trade",
                "London Trades","NY Trades"],
               [12,8,8,8,10,14,14,14,14,14,10], bg=C_HEADER_DARK)

    for r, m in enumerate(months, 2):
        sub  = df[df["ym"]==m]
        wins = sub[sub["pnl_usd"]>0]
        wr   = len(wins)/len(sub) if len(sub)>0 else 0
        pnl  = sub["pnl_usd"].sum()
        bg   = C_GREEN_LIGHT if pnl>=0 else C_RED_LIGHT

        lon = sub[(sub["hour"]>=8)&(sub["hour"]<11)]
        ny  = sub[(sub["hour"]>=13)&(sub["hour"]<16)]

        data_cell(ws, r, 1, m, bold=True, align="center", bg=bg)
        data_cell(ws, r, 2, len(sub), align="center", bg=bg)
        data_cell(ws, r, 3, len(wins), align="center", bg=bg)
        data_cell(ws, r, 4, len(sub)-len(wins), align="center", bg=bg)
        data_cell(ws, r, 5, round(wr,3), fmt="0.0%", align="center", bg=bg)
        data_cell(ws, r, 6, round(pnl,2), fmt="$#,##0.00",
                  align="right", bold=True, bg=bg)
        data_cell(ws, r, 7, round(sub["pnl_usd"].mean(),2) if len(sub)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=bg)
        data_cell(ws, r, 8, round(sub["pnl_usd"].max(),2),
                  fmt="$#,##0.00", align="right", bg=C_GREEN_LIGHT)
        data_cell(ws, r, 9, round(sub["pnl_usd"].min(),2),
                  fmt="$#,##0.00", align="right", bg=C_RED_LIGHT)
        data_cell(ws, r,10, len(lon), align="center", bg=bg)
        data_cell(ws, r,11, len(ny),  align="center", bg=bg)

    return ws


def build_hourly_sheet(wb, df):
    ws = wb.create_sheet("Hourly Analysis")
    ws.sheet_view.showGridLines = False

    hours = sorted(df["hour"].unique())
    header_row(ws, 1,
               ["Hour (UTC)","Session","Trades","Wins","Losses","Win Rate",
                "Total P&L","Avg P&L","Avg Win","Avg Loss","Avg Duration (h)"],
               [12,10,8,8,8,10,14,14,12,12,16], bg=C_HEADER_DARK)

    for r, h in enumerate(hours, 2):
        sub    = df[df["hour"]==h]
        wins   = sub[sub["pnl_usd"]>0]; losses = sub[sub["pnl_usd"]<=0]
        wr     = len(wins)/len(sub) if len(sub)>0 else 0
        session= ("London" if 8<=h<11 else "NY" if 13<=h<16 else "Off-hours")
        bg     = C_GREEN_LIGHT if wr>=0.55 else C_RED_LIGHT if wr<0.45 else C_WHITE
        dur_h  = sub["duration_min"].mean()/60 if len(sub)>0 else 0

        data_cell(ws, r, 1, f"{h:02d}:00", align="center", bold=True)
        data_cell(ws, r, 2, session, align="center",
                  bg=("E0F2FE" if session=="London" else
                      "FEF9E7" if session=="NY" else C_LIGHT_BG))
        data_cell(ws, r, 3, len(sub), align="center", bg=bg)
        data_cell(ws, r, 4, len(wins), align="center", bg=bg)
        data_cell(ws, r, 5, len(losses), align="center", bg=bg)
        data_cell(ws, r, 6, round(wr,3), fmt="0.0%", align="center",
                  bold=True, bg=bg)
        data_cell(ws, r, 7, round(sub["pnl_usd"].sum(),2),
                  fmt="$#,##0.00", align="right", bg=bg)
        data_cell(ws, r, 8, round(sub["pnl_usd"].mean(),2) if len(sub)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=bg)
        data_cell(ws, r, 9, round(wins["pnl_usd"].mean(),2) if len(wins)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=C_GREEN_LIGHT)
        data_cell(ws, r,10, round(losses["pnl_usd"].mean(),2) if len(losses)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=C_RED_LIGHT)
        data_cell(ws, r,11, round(dur_h,2), fmt="0.00", align="center")

    return ws


def build_duration_sheet(wb, df):
    ws = wb.create_sheet("Duration Analysis")
    ws.sheet_view.showGridLines = False

    buckets = [
        ("<30 min",   0,   30),
        ("30m–1h",    30,  60),
        ("1h–2h",     60,  120),
        ("2h–4h",    120,  240),
        ("4h–8h",    240,  480),
        ("8h–24h",   480, 1440),
        (">24h",    1440, 99999),
    ]

    header_row(ws, 1,
               ["Duration Bucket","Trades","Wins","Win Rate",
                "Avg P&L","Total P&L","Avg Win","Avg Loss","Max Win","Max Loss"],
               [14,8,8,10,12,14,12,12,12,12], bg=C_HEADER_DARK)

    for r, (label, lo, hi) in enumerate(buckets, 2):
        sub  = df[(df["duration_min"]>=lo)&(df["duration_min"]<hi)]
        if len(sub)==0:
            data_cell(ws, r, 1, label, align="center")
            for c in range(2, 11):
                data_cell(ws, r, c, 0, align="center")
            continue
        wins   = sub[sub["pnl_usd"]>0]; losses = sub[sub["pnl_usd"]<=0]
        wr     = len(wins)/len(sub)
        bg     = C_GREEN_LIGHT if wr>=0.55 else C_RED_LIGHT if wr<0.45 else C_WHITE

        data_cell(ws, r, 1, label, bold=True, align="center", bg=bg)
        data_cell(ws, r, 2, len(sub), align="center", bg=bg)
        data_cell(ws, r, 3, len(wins), align="center", bg=bg)
        data_cell(ws, r, 4, round(wr,3), fmt="0.0%", bold=True, align="center", bg=bg)
        data_cell(ws, r, 5, round(sub["pnl_usd"].mean(),2),
                  fmt="$#,##0.00", align="right", bg=bg)
        data_cell(ws, r, 6, round(sub["pnl_usd"].sum(),2),
                  fmt="$#,##0.00", align="right", bold=True, bg=bg)
        data_cell(ws, r, 7, round(wins["pnl_usd"].mean(),2) if len(wins)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=C_GREEN_LIGHT)
        data_cell(ws, r, 8, round(losses["pnl_usd"].mean(),2) if len(losses)>0 else 0,
                  fmt="$#,##0.00", align="right", bg=C_RED_LIGHT)
        data_cell(ws, r, 9, round(sub["pnl_usd"].max(),2),
                  fmt="$#,##0.00", align="right", bg=C_GREEN_LIGHT)
        data_cell(ws, r,10, round(sub["pnl_usd"].min(),2),
                  fmt="$#,##0.00", align="right", bg=C_RED_LIGHT)

    # Stats summary below
    ws.row_dimensions[10].height = 12
    ws["A11"] = "Duration Statistics"
    ws["A11"].font = font(bold=True, size=11)

    stats = [
        ("Median trade duration",  f"{df['duration_min'].median():.0f} min  ({df['duration_min'].median()/60:.1f}h)"),
        ("Mean trade duration",    f"{df['duration_min'].mean():.0f} min  ({df['duration_min'].mean()/60:.1f}h)"),
        ("Fastest winner",         f"{df[df['pnl_usd']>0]['duration_min'].min():.0f} min"),
        ("Longest winner",         f"{df[df['pnl_usd']>0]['duration_min'].max():.0f} min  ({df[df['pnl_usd']>0]['duration_min'].max()/60:.1f}h)"),
        ("Fastest loser",          f"{df[df['pnl_usd']<=0]['duration_min'].min():.0f} min"),
        ("Longest loser",          f"{df[df['pnl_usd']<=0]['duration_min'].max():.0f} min  ({df[df['pnl_usd']<=0]['duration_min'].max()/60:.1f}h)"),
        ("Trades < 1 hour",        f"{(df['duration_min']<60).sum()} ({(df['duration_min']<60).mean()*100:.1f}%)"),
        ("Trades > 8 hours",       f"{(df['duration_min']>480).sum()} ({(df['duration_min']>480).mean()*100:.1f}%)"),
    ]
    for r, (label, value) in enumerate(stats, 12):
        ws.cell(row=r, column=1, value=label).font = font(size=9, color="475569")
        ws.cell(row=r, column=2, value=value).font = font(bold=True, size=9)

    return ws


def build_charts_sheet(wb, result, df):
    ws = wb.create_sheet("Charts")
    ws.sheet_view.showGridLines = False

    ws["A1"] = "Equity Curve & Drawdown"
    ws["A1"].font = font(bold=True, size=12)
    ws.row_dimensions[1].height = 20

    img1 = build_equity_chart(result.equity_curve, ACCOUNT_BALANCE)
    ws.add_image(img1, "A2")
    ws.row_dimensions[28].height = 20

    ws["A29"] = "Monthly P&L"
    ws["A29"].font = font(bold=True, size=12)
    img2 = build_monthly_chart(df.copy())
    ws.add_image(img2, "A30")
    ws.row_dimensions[53].height = 20

    ws["A54"] = "Win Rate & Trade Count by Entry Hour"
    ws["A54"].font = font(bold=True, size=12)
    img3 = build_hourly_chart(df.copy())
    ws.add_image(img3, "A55")
    ws.row_dimensions[78].height = 20

    ws["A79"] = "Trade Duration Analysis"
    ws["A79"].font = font(bold=True, size=12)
    img4 = build_duration_chart(df.copy())
    ws.add_image(img4, "A80")

    return ws


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*55)
    print("  GoldScalperBot — Excel Trade Analyser v1.0")
    print("="*55)

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}"); exit(1)
    a = mt5.account_info()
    if a is None: print("MT5 not logged in"); mt5.shutdown(); exit(1)
    print(f"MT5 connected — {a.login}  Balance: ${a.balance:,.2f}\n")

    loader = DataLoader(SYMBOL)
    df_h4  = loader.load("H4",  INSAMPLE_START, INSAMPLE_END)
    df_h1  = loader.load("H1",  INSAMPLE_START, INSAMPLE_END)
    df_m15 = loader.load("M15", INSAMPLE_START, INSAMPLE_END)

    print("Running backtest...")
    runner = BacktestRunner()
    params = {"pivot_strength":5,"min_wave_pts":1200,
              "bos_method":"single_close","ehl_tolerance":"atr_0.5"}
    result = runner._run_single(df_h4, df_h1, df_m15, params)

    print(f"\nBacktest complete — {result.total_trades} trades")
    print(f"  Win rate  : {result.win_rate*100:.1f}%")
    print(f"  Net profit: ${result.net_profit_usd:,.2f}")
    print(f"  Max DD    : {result.max_drawdown_pct:.1f}%")

    # Build trade DataFrame with duration columns
    rows = []
    for t in result.trades:
        ot  = t.open_time
        ct  = t.close_time
        dur_min = 0
        dur_str = "—"
        if ot and ct:
            delta   = ct - ot
            dur_min = delta.total_seconds() / 60
            h = int(dur_min // 60); m = int(dur_min % 60)
            dur_str = f"{h}h {m:02d}m"
        rows.append({
            "trade_id":    t.trade_id,
            "direction":   t.direction,
            "open_time":   ot,
            "close_time":  ct,
            "duration_min":round(dur_min, 1),
            "duration_str":dur_str,
            "hour":        ot.hour if ot else 0,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "sl_price":    t.sl_price,
            "tp_price":    t.tp_price,
            "lot_size":    t.lot_size,
            "signal_grade":t.signal_grade,
            "pnl_usd":     t.pnl_usd,
            "exit_reason": t.exit_reason,
        })
    df = pd.DataFrame(rows)

    print("\nBuilding Excel workbook...")
    wb = Workbook()
    wb.remove(wb.active)   # Remove default sheet

    print("  → Summary sheet")
    build_summary_sheet(wb, result, df.copy())
    print("  → Trade log sheet")
    build_trade_log_sheet(wb, df.copy())
    print("  → Monthly P&L sheet")
    build_monthly_sheet(wb, df.copy())
    print("  → Hourly analysis sheet")
    build_hourly_sheet(wb, df.copy())
    print("  → Duration analysis sheet")
    build_duration_sheet(wb, df.copy())
    print("  → Charts sheet (takes ~30s)")
    build_charts_sheet(wb, result, df.copy())

    wb.save(OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")
    mt5.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
