# ================================================================
#  report_generator.py
#  GoldScalperBot — Backtest PDF Report Generator
#
#  Generates a professional multi-page PDF report from backtest
#  results including:
#    Page 1  : Cover — summary stats + key metrics
#    Page 2  : Equity curve + drawdown chart
#    Page 3  : Monthly P&L bar chart
#    Page 4  : Trade distribution charts (4 panels)
#    Page 5  : Parameter comparison table (top 10 combos)
#    Page 6  : Full trade log table
#
#  Usage:
#    from report_generator import ReportGenerator
#    gen = ReportGenerator()
#    gen.generate(results, best_result, output_path="backtest_results/report.pdf")
#
#  Or run standalone to generate a sample report:
#    python report_generator.py
# ================================================================

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime, timezone
from typing import List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import (HexColor, white, black,
                                   lightgrey, darkgrey)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image, PageBreak,
                                 HRFlowable, KeepTogether)
from reportlab.platypus.flowables import HRFlowable

from config import SYMBOL, ACCOUNT_BALANCE, ACCOUNT_LEVERAGE


# ----------------------------------------------------------------
#  COLOUR PALETTE
# ----------------------------------------------------------------
C_DARK      = HexColor("#1a1a2e")   # Dark navy — headers
C_PRIMARY   = HexColor("#0f3460")   # Deep blue — section titles
C_ACCENT    = HexColor("#e94560")   # Red accent — highlights
C_GREEN     = HexColor("#16a34a")   # Win colour
C_RED       = HexColor("#dc2626")   # Loss colour
C_GOLD      = HexColor("#d97706")   # Gold accent — fitting for XAUUSD
C_LIGHT     = HexColor("#f8fafc")   # Light background
C_BORDER    = HexColor("#e2e8f0")   # Table borders
C_MUTED     = HexColor("#64748b")   # Muted text


# ----------------------------------------------------------------
#  MATPLOTLIB STYLE
# ----------------------------------------------------------------
CHART_STYLE = {
    "figure.facecolor":   "#0f1117",
    "axes.facecolor":     "#1a1a2e",
    "axes.edgecolor":     "#334155",
    "axes.labelcolor":    "#94a3b8",
    "axes.grid":          True,
    "grid.color":         "#1e293b",
    "grid.linewidth":     0.5,
    "text.color":         "#e2e8f0",
    "xtick.color":        "#94a3b8",
    "ytick.color":        "#94a3b8",
    "lines.linewidth":    1.5,
    "font.size":          9,
}


# ----------------------------------------------------------------
#  HELPER: chart to reportlab Image
# ----------------------------------------------------------------
def fig_to_image(fig, width_cm: float = 17.0,
                 height_cm: float = 8.0) -> Image:
    """Convert a matplotlib figure to a ReportLab Image object"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


# ----------------------------------------------------------------
#  CHART BUILDERS
# ----------------------------------------------------------------

def build_equity_drawdown_chart(equity_curve: List[float],
                                 starting_balance: float) -> Image:
    """Equity curve + drawdown subplot"""
    with plt.style.context(CHART_STYLE):
        fig = plt.figure(figsize=(11, 5.5), facecolor="#0f1117")
        gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1],
                       hspace=0.08)

        # --- Equity curve ---
        ax1 = fig.add_subplot(gs[0])
        eq  = np.array(equity_curve)
        x   = range(len(eq))

        ax1.fill_between(x, starting_balance, eq,
                         where=(eq >= starting_balance),
                         color="#16a34a", alpha=0.25)
        ax1.fill_between(x, starting_balance, eq,
                         where=(eq < starting_balance),
                         color="#dc2626", alpha=0.25)
        ax1.plot(x, eq, color="#d97706", linewidth=1.5, zorder=3)
        ax1.axhline(starting_balance, color="#64748b",
                    linewidth=0.8, linestyle="--", alpha=0.6)
        ax1.set_ylabel("Account equity ($)", color="#94a3b8", fontsize=9)
        ax1.set_title("Equity curve", color="#e2e8f0",
                      fontsize=10, pad=8)
        ax1.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax1.set_xlim(0, max(1, len(eq) - 1))
        ax1.tick_params(labelbottom=False)

        # --- Drawdown ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        peak = np.maximum.accumulate(eq)
        dd   = ((eq - peak) / peak) * 100
        ax2.fill_between(x, dd, 0, color="#dc2626", alpha=0.4)
        ax2.plot(x, dd, color="#dc2626", linewidth=1.0)
        ax2.set_ylabel("Drawdown %", color="#94a3b8", fontsize=9)
        ax2.set_xlabel("Trade number", color="#94a3b8", fontsize=9)
        ax2.set_xlim(0, max(1, len(eq) - 1))

        fig.tight_layout(pad=1.2)
    return fig_to_image(fig, width_cm=17, height_cm=8)


def build_monthly_pnl_chart(trades: list) -> Image:
    """Monthly P&L bar chart"""
    if not trades:
        return None

    rows = []
    for t in trades:
        if t.close_time:
            rows.append({"month": t.close_time.strftime("%Y-%m"),
                         "pnl":   t.pnl_usd})
    if not rows:
        return None

    df  = pd.DataFrame(rows)
    mnl = df.groupby("month")["pnl"].sum().reset_index()
    mnl["color"] = mnl["pnl"].apply(
        lambda v: "#16a34a" if v >= 0 else "#dc2626")

    with plt.style.context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(11, 4), facecolor="#0f1117")
        ax.bar(mnl["month"], mnl["pnl"], color=mnl["color"],
               width=0.7, zorder=3)
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.set_title("Monthly P&L ($)", color="#e2e8f0",
                     fontsize=10, pad=8)
        ax.set_xlabel("Month", color="#94a3b8", fontsize=9)
        ax.set_ylabel("P&L ($)", color="#94a3b8", fontsize=9)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=45, ha="right", fontsize=7)
        fig.tight_layout(pad=1.2)
    return fig_to_image(fig, width_cm=17, height_cm=6)


def build_distribution_charts(trades: list) -> Image:
    """2x2 panel: win/loss pie, P&L histogram, exit reasons, grade breakdown"""
    if not trades:
        return None

    pnls    = [t.pnl_usd for t in trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]
    exits   = {}
    grades  = {}
    for t in trades:
        exits[t.exit_reason]  = exits.get(t.exit_reason, 0)  + 1
        grades[t.signal_grade]= grades.get(t.signal_grade, 0) + 1

    with plt.style.context(CHART_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                                 facecolor="#0f1117")
        fig.suptitle("Trade distribution analysis",
                     color="#e2e8f0", fontsize=11, y=0.98)

        # 1. Win/Loss pie
        ax = axes[0, 0]
        if wins or losses:
            ax.pie([len(wins), len(losses)],
                   labels=["Wins", "Losses"],
                   colors=["#16a34a", "#dc2626"],
                   autopct="%1.1f%%",
                   textprops={"color": "#e2e8f0", "fontsize": 9},
                   startangle=90)
        ax.set_title("Win / loss ratio", color="#e2e8f0",
                     fontsize=9, pad=6)

        # 2. P&L distribution histogram
        ax = axes[0, 1]
        if pnls:
            bins = min(30, max(10, len(pnls) // 5))
            ax.hist([p for p in pnls if p > 0],
                    bins=bins, color="#16a34a", alpha=0.7,
                    label="Wins", zorder=3)
            ax.hist([p for p in pnls if p <= 0],
                    bins=bins, color="#dc2626", alpha=0.7,
                    label="Losses", zorder=3)
            ax.axvline(0, color="#64748b", linewidth=0.8)
            ax.legend(fontsize=8)
        ax.set_title("P&L distribution", color="#e2e8f0",
                     fontsize=9, pad=6)
        ax.set_xlabel("P&L ($)", color="#94a3b8", fontsize=8)

        # 3. Exit reason bar chart
        ax = axes[1, 0]
        if exits:
            clrs = {"TP": "#16a34a", "SL": "#dc2626",
                    "TRAIL": "#d97706", "SESSION_END": "#6366f1",
                    "MOMENTUM": "#06b6d4"}
            names = list(exits.keys())
            vals  = list(exits.values())
            colors= [clrs.get(n, "#94a3b8") for n in names]
            ax.bar(names, vals, color=colors, zorder=3)
            ax.set_title("Exit reasons", color="#e2e8f0",
                         fontsize=9, pad=6)
            ax.set_ylabel("Count", color="#94a3b8", fontsize=8)
            plt.setp(ax.xaxis.get_majorticklabels(),
                     rotation=20, ha="right", fontsize=8)

        # 4. Grade breakdown
        ax = axes[1, 1]
        if grades:
            gclrs = {"A+": "#d97706", "B": "#6366f1",
                     "C": "#06b6d4", "SKIP": "#94a3b8"}
            gnames = list(grades.keys())
            gvals  = list(grades.values())
            gcolor = [gclrs.get(g, "#94a3b8") for g in gnames]
            ax.bar(gnames, gvals, color=gcolor, zorder=3)
            ax.set_title("Signal grade breakdown",
                         color="#e2e8f0", fontsize=9, pad=6)
            ax.set_ylabel("Count", color="#94a3b8", fontsize=8)

        fig.tight_layout(pad=1.5, rect=[0, 0, 1, 0.96])
    return fig_to_image(fig, width_cm=17, height_cm=10)


def build_parameter_comparison_chart(results: list) -> Image:
    """Horizontal bar chart of top 10 parameter sets by score"""
    if not results:
        return None

    top10 = results[:10]
    labels = [str(r.params.get("bos_method", "?"))[:12] + " / " +
              str(r.params.get("pivot_strength", "?"))
              for r in top10]
    scores = [r.score for r in top10]
    wrates = [r.win_rate * 100 for r in top10]

    with plt.style.context(CHART_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                        facecolor="#0f1117")
        y = range(len(top10))

        # Score bars
        bars = ax1.barh(y, scores, color="#d97706", alpha=0.8, zorder=3)
        ax1.set_yticks(y)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_title("Score (top 10 combos)",
                      color="#e2e8f0", fontsize=9, pad=6)
        ax1.set_xlabel("Score", color="#94a3b8", fontsize=8)
        ax1.invert_yaxis()
        for bar, val in zip(bars, scores):
            ax1.text(bar.get_width() + max(scores) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center",
                     color="#e2e8f0", fontsize=7)

        # Win rate bars
        bars2 = ax2.barh(y, wrates,
                          color=["#16a34a" if w >= 50 else "#dc2626"
                                 for w in wrates],
                          alpha=0.8, zorder=3)
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_title("Win rate % (top 10)",
                      color="#e2e8f0", fontsize=9, pad=6)
        ax2.set_xlabel("Win rate %", color="#94a3b8", fontsize=8)
        ax2.axvline(50, color="#64748b", linewidth=0.8,
                    linestyle="--", alpha=0.6)
        ax2.invert_yaxis()

        fig.tight_layout(pad=1.5)
    return fig_to_image(fig, width_cm=17, height_cm=7)


# ----------------------------------------------------------------
#  REPORT GENERATOR CLASS
# ----------------------------------------------------------------

class ReportGenerator:

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Define custom paragraph styles"""
        s = self.styles

        self.sty_title = ParagraphStyle(
            "ReportTitle",
            parent=s["Title"],
            fontSize=24, textColor=white,
            spaceAfter=6, alignment=TA_CENTER,
            fontName="Helvetica-Bold"
        )
        self.sty_subtitle = ParagraphStyle(
            "Subtitle",
            parent=s["Normal"],
            fontSize=11, textColor=HexColor("#94a3b8"),
            spaceAfter=4, alignment=TA_CENTER
        )
        self.sty_h1 = ParagraphStyle(
            "H1",
            parent=s["Heading1"],
            fontSize=13, textColor=C_GOLD,
            spaceBefore=14, spaceAfter=6,
            fontName="Helvetica-Bold",
            borderPad=4
        )
        self.sty_h2 = ParagraphStyle(
            "H2",
            parent=s["Heading2"],
            fontSize=11, textColor=C_PRIMARY,
            spaceBefore=10, spaceAfter=4,
            fontName="Helvetica-Bold"
        )
        self.sty_body = ParagraphStyle(
            "Body",
            parent=s["Normal"],
            fontSize=9, textColor=black,
            spaceAfter=4, leading=14
        )
        self.sty_small = ParagraphStyle(
            "Small",
            parent=s["Normal"],
            fontSize=8, textColor=HexColor("#475569"),
            spaceAfter=2
        )
        self.sty_metric_val = ParagraphStyle(
            "MetricVal",
            parent=s["Normal"],
            fontSize=18, fontName="Helvetica-Bold",
            textColor=C_DARK, alignment=TA_CENTER,
            spaceAfter=2
        )
        self.sty_metric_lbl = ParagraphStyle(
            "MetricLbl",
            parent=s["Normal"],
            fontSize=8, textColor=C_MUTED,
            alignment=TA_CENTER
        )

    def _metric_cell(self, value: str, label: str,
                     color: HexColor = C_DARK) -> list:
        """Creates a [value, label] pair for a summary metric table"""
        val_style = ParagraphStyle(
            "mv", parent=self.sty_metric_val, textColor=color)
        return [
            Paragraph(value, val_style),
            Paragraph(label, self.sty_metric_lbl)
        ]

    def _cover_page(self, result, story: list, run_date: str):
        """Page 1 — Cover with key metrics"""

        # Dark header band via coloured table
        header_data = [[Paragraph(
            "GoldScalperBot", ParagraphStyle(
                "ct", parent=self.sty_title,
                textColor=C_GOLD, fontSize=26))],
            [Paragraph(
                "Backtest Performance Report",
                self.sty_subtitle)],
            [Paragraph(
                f"Symbol: {SYMBOL}  |  Account: ${ACCOUNT_BALANCE:,.0f}  "
                f"|  Leverage: 1:{ACCOUNT_LEVERAGE}  |  Generated: {run_date}",
                self.sty_subtitle)]
        ]
        header_tbl = Table(header_data, colWidths=[17 * cm])
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
            ("TOPPADDING",    (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
            ("LEFTPADDING",   (0, 0), (-1, -1), 20),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
            ("ROUNDEDCORNERS", [8]),
        ]))
        story.append(header_tbl)
        story.append(Spacer(1, 20))

        # Key metric cards — 4 per row
        t  = result.total_trades
        wr = result.win_rate
        np_ = result.net_profit_usd
        dd  = result.max_drawdown_pct
        pf  = result.profit_factor
        sc  = result.score
        sh  = result.sharpe_ratio

        profit_color = C_GREEN if np_ >= 0 else C_RED
        dd_color     = C_GREEN if dd < 5 else (C_GOLD if dd < 15 else C_RED)

        row1 = [
            self._metric_cell(f"{t}", "total trades"),
            self._metric_cell(f"{wr*100:.1f}%",
                               "win rate",
                               C_GREEN if wr >= 0.5 else C_RED),
            self._metric_cell(f"${np_:,.2f}",
                               "net profit", profit_color),
            self._metric_cell(f"{dd:.1f}%",
                               "max drawdown", dd_color),
        ]
        row2 = [
            self._metric_cell(f"{pf:.2f}", "profit factor"),
            self._metric_cell(f"{sh:.2f}", "sharpe ratio"),
            self._metric_cell(f"${result.avg_win_usd:.2f}",
                               "avg win", C_GREEN),
            self._metric_cell(f"${abs(result.avg_loss_usd):.2f}",
                               "avg loss", C_RED),
        ]
        row3 = [
            self._metric_cell(f"{sc:.3f}", "overall score", C_GOLD),
            self._metric_cell(
                f"${result.max_drawdown_usd:.2f}", "max DD ($)"),
            self._metric_cell(
                f"{result.winning_trades}", "winning trades", C_GREEN),
            self._metric_cell(
                f"{result.losing_trades}", "losing trades", C_RED),
        ]

        col_w = [4.15 * cm] * 4
        for row_data in [row1, row2, row3]:
            tbl = Table([row_data], colWidths=col_w, rowHeights=[1.4 * cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), C_LIGHT),
                ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
                ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
                ("TOPPADDING",    (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 6))

        # Best parameter set
        story.append(Spacer(1, 10))
        story.append(Paragraph("Best parameter combination", self.sty_h1))
        param_rows = [[k, str(v)] for k, v in result.params.items()]
        param_tbl  = Table(
            [["Parameter", "Value"]] + param_rows,
            colWidths=[8 * cm, 9 * cm]
        )
        param_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_PRIMARY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("BACKGROUND",    (0, 1), (-1, -1), C_LIGHT),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, C_LIGHT]),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(param_tbl)

    def _charts_page(self, result, story: list):
        """Page 2 — Equity curve + drawdown"""
        story.append(PageBreak())
        story.append(Paragraph("Equity curve and drawdown", self.sty_h1))
        story.append(Paragraph(
            "The equity curve shows account balance progression trade by trade. "
            "Drawdown shows percentage decline from the running peak equity. "
            "The strategy's risk control target is max drawdown below 15%.",
            self.sty_body))
        story.append(Spacer(1, 8))

        img = build_equity_drawdown_chart(
            result.equity_curve, ACCOUNT_BALANCE)
        if img:
            story.append(img)

        story.append(Spacer(1, 14))
        story.append(Paragraph("Monthly P&L breakdown", self.sty_h1))
        story.append(Paragraph(
            "Each bar shows net profit or loss for that calendar month. "
            "Green = profitable month, red = losing month. "
            "Consistency across months indicates a robust strategy.",
            self.sty_body))
        story.append(Spacer(1, 8))

        img2 = build_monthly_pnl_chart(result.trades)
        if img2:
            story.append(img2)

    def _distribution_page(self, result, story: list):
        """Page 3 — Trade distribution charts"""
        story.append(PageBreak())
        story.append(Paragraph("Trade distribution analysis", self.sty_h1))
        story.append(Paragraph(
            "Four-panel breakdown: win/loss ratio, P&L distribution histogram, "
            "exit reason frequency, and signal grade breakdown. "
            "A healthy distribution shows exits concentrated on TP and TRAIL "
            "rather than SL and SESSION_END.",
            self.sty_body))
        story.append(Spacer(1, 8))

        img = build_distribution_charts(result.trades)
        if img:
            story.append(img)

    def _parameter_comparison_page(self, results: list, story: list):
        """Page 4 — Top parameter set comparison"""
        story.append(PageBreak())
        story.append(Paragraph("Parameter set comparison — top 10", self.sty_h1))
        story.append(Paragraph(
            "All parameter combinations ranked by score: "
            "(Net Profit / Max Drawdown) x Win Rate x sqrt(Trade Count). "
            "Higher score = better risk-adjusted performance with statistical significance.",
            self.sty_body))
        story.append(Spacer(1, 8))

        img = build_parameter_comparison_chart(results)
        if img:
            story.append(img)

        story.append(Spacer(1, 12))

        # Table of top 10
        if not results:
            return

        top10 = results[:10]
        headers = ["Rank", "Score", "Trades", "Win%", "Net P&L",
                   "Max DD%", "PF", "Pivot", "BOS", "Tolerance"]
        rows = [headers]
        for i, r in enumerate(top10):
            rows.append([
                str(i + 1),
                f"{r.score:.3f}",
                str(r.total_trades),
                f"{r.win_rate*100:.1f}%",
                f"${r.net_profit_usd:,.0f}",
                f"{r.max_drawdown_pct:.1f}%",
                f"{r.profit_factor:.2f}",
                str(r.params.get("pivot_strength", "-")),
                str(r.params.get("bos_method", "-")),
                str(r.params.get("ehl_tolerance", "-")),
            ])

        col_w = [1.2, 1.5, 1.3, 1.3, 2.0,
                 1.5, 1.2, 1.2, 2.8, 2.0]
        col_w = [w * cm for w in col_w]

        tbl = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_PRIMARY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, C_LIGHT]),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            # Highlight best row
            ("BACKGROUND",    (0, 1), (-1, 1), HexColor("#fef3c7")),
            ("FONTNAME",      (0, 1), (-1, 1), "Helvetica-Bold"),
        ]))
        story.append(tbl)

    def _trade_log_page(self, result, story: list):
        """Page 5 — Trade log (last 50 trades for space)"""
        story.append(PageBreak())
        story.append(Paragraph("Trade log (most recent 50 trades)", self.sty_h1))
        story.append(Paragraph(
            f"Full trade log contains {result.total_trades} trades. "
            "Showing most recent 50 for space. "
            "Complete log available in trades_*.csv in backtest_results/.",
            self.sty_body))
        story.append(Spacer(1, 8))

        trades = result.trades[-50:] if len(result.trades) > 50 else result.trades
        if not trades:
            story.append(Paragraph("No trades to display.", self.sty_body))
            return

        headers = ["#", "Dir", "Open time", "Entry", "Exit",
                   "SL", "Lots", "Grade", "P&L ($)", "Exit"]
        rows    = [headers]
        for t in reversed(trades):
            open_str = (t.open_time.strftime("%m-%d %H:%M")
                        if t.open_time else "-")
            pnl_str  = f"${t.pnl_usd:+.2f}"
            rows.append([
                str(t.trade_id),
                t.direction[:1],
                open_str,
                f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}",
                f"{t.sl_price:.2f}",
                f"{t.lot_size:.2f}",
                t.signal_grade,
                pnl_str,
                t.exit_reason[:6],
            ])

        col_w = [0.8, 0.8, 2.2, 1.8, 1.8,
                 1.8, 1.1, 1.2, 1.8, 1.5]
        col_w = [w * cm for w in col_w]

        # Colour P&L column green/red
        tbl_style = [
            ("BACKGROUND",    (0, 0), (-1, 0), C_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, C_LIGHT]),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 3),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ]
        for i, t in enumerate(reversed(trades), start=1):
            clr = C_GREEN if t.pnl_usd >= 0 else C_RED
            tbl_style.append(("TEXTCOLOR", (8, i), (8, i), clr))
            tbl_style.append(("FONTNAME",  (8, i), (8, i),
                               "Helvetica-Bold"))

        tbl = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(TableStyle(tbl_style))
        story.append(tbl)

    def generate(self, results: list,
                 best_result,
                 output_path: str = "backtest_results/report.pdf"):
        """
        Generate the complete PDF report.

        results:     List[BacktestResult] — all parameter combinations ranked
        best_result: BacktestResult       — the top-scoring combination
        output_path: str                  — where to save the PDF
        """
        os.makedirs(os.path.dirname(output_path)
                    if os.path.dirname(output_path) else ".",
                    exist_ok=True)

        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title="GoldScalperBot Backtest Report",
            author="GoldScalperBot",
            subject=f"XAUUSD.. backtest — {run_date}"
        )

        story = []

        print(f"Generating PDF report...")
        self._cover_page(best_result, story, run_date)
        self._charts_page(best_result, story)
        self._distribution_page(best_result, story)
        self._parameter_comparison_page(results, story)
        self._trade_log_page(best_result, story)

        doc.build(story)
        print(f"Report saved to: {output_path}")
        return output_path


# ================================================================
#  STANDALONE TEST — generates a sample report with random data
# ================================================================

if __name__ == "__main__":
    from backtest_engine import BacktestResult, BacktestTrade
    from datetime import timedelta
    import random

    print("Generating sample PDF report with synthetic data...")

    # Create synthetic trade data
    random.seed(42)
    np.random.seed(42)

    trades = []
    balance  = ACCOUNT_BALANCE
    equity   = [balance]
    base_time = datetime(2022, 1, 15, 9, 0, tzinfo=timezone.utc)

    for i in range(120):
        direction = random.choice(["LONG", "SHORT"])
        grade     = random.choices(["A+", "B", "C"],
                                    weights=[0.3, 0.5, 0.2])[0]
        pnl = random.gauss(
            12 if grade == "A+" else 6 if grade == "B" else 1,
            25
        )
        balance  += pnl
        equity.append(round(balance, 2))
        open_t = base_time + timedelta(hours=i * 6 +
                                        random.randint(0, 3))
        exit_r = random.choices(
            ["TP", "SL", "TRAIL", "SESSION_END", "MOMENTUM"],
            weights=[0.30, 0.25, 0.25, 0.10, 0.10]
        )[0]

        t = BacktestTrade(
            trade_id       = i + 1,
            direction      = direction,
            open_time      = open_t,
            close_time     = open_t + timedelta(hours=random.randint(1, 8)),
            entry_price    = 2100.0 + random.uniform(-50, 50),
            exit_price     = 2100.0 + random.uniform(-50, 50),
            sl_price       = 2100.0 - 50,
            tp_price       = 2100.0 + 80,
            lot_size       = round(random.uniform(0.10, 0.40), 2),
            spread_at_entry= random.randint(15, 30),
            signal_score   = random.randint(3, 6),
            signal_grade   = grade,
            prob_weight    = round(random.uniform(0.5, 1.0), 2),
            pnl_usd        = round(pnl, 2),
            pnl_points     = round(pnl / 0.01 / 100, 1),
            exit_reason    = exit_r,
            bos_method     = "single_close",
            choch_scope    = "H4_ONLY",
            ehl_tolerance  = "atr_0.5",
            pivot_strength = 5,
            is_open        = False
        )
        trades.append(t)

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    net    = sum(t.pnl_usd for t in trades)
    peak   = ACCOUNT_BALANCE
    max_dd = 0.0
    for e in equity:
        peak   = max(peak, e)
        max_dd = max(max_dd, peak - e)

    best = BacktestResult(
        params         = {"pivot_strength": 5, "min_wave_pts": 1200,
                          "bos_method": "single_close",
                          "ehl_tolerance": "atr_0.5"},
        trades         = trades,
        net_profit_usd = round(net, 2),
        max_drawdown_usd = round(max_dd, 2),
        max_drawdown_pct = round(max_dd / ACCOUNT_BALANCE * 100, 2),
        win_rate       = len(wins) / len(trades),
        total_trades   = len(trades),
        winning_trades = len(wins),
        losing_trades  = len(losses),
        avg_win_usd    = round(np.mean([t.pnl_usd for t in wins]), 2) if wins else 0,
        avg_loss_usd   = round(np.mean([t.pnl_usd for t in losses]), 2) if losses else 0,
        profit_factor  = (abs(sum(t.pnl_usd for t in wins)) /
                          abs(sum(t.pnl_usd for t in losses)) + 1e-9),
        sharpe_ratio   = round(np.mean([t.pnl_usd for t in trades]) /
                               (np.std([t.pnl_usd for t in trades]) + 1e-9) *
                               np.sqrt(252), 2),
        equity_curve   = equity,
    )
    if best.max_drawdown_usd > 0 and best.total_trades >= 10:
        best.score = ((best.net_profit_usd / best.max_drawdown_usd) *
                      best.win_rate * np.sqrt(best.total_trades))

    # Create a few more synthetic results for comparison table
    all_results = [best]
    for j in range(9):
        r2 = BacktestResult(
            params={"pivot_strength": random.choice([3, 5, 8]),
                    "min_wave_pts":   random.choice([800, 1200, 1600]),
                    "bos_method":     random.choice(["single_close",
                                                      "double_close",
                                                      "body_close"]),
                    "ehl_tolerance":  random.choice(["fixed_100",
                                                      "atr_0.3",
                                                      "atr_0.5",
                                                      "pct_0.02"])},
            total_trades   = random.randint(40, 150),
            win_rate       = random.uniform(0.40, 0.65),
            net_profit_usd = random.uniform(-100, 400),
            max_drawdown_usd = random.uniform(30, 200),
            max_drawdown_pct = random.uniform(2, 15),
            profit_factor  = random.uniform(0.8, 2.5),
            sharpe_ratio   = random.uniform(-0.5, 2.0),
            equity_curve   = equity,
            trades         = []
        )
        if r2.max_drawdown_usd > 0 and r2.total_trades >= 10:
            r2.score = ((r2.net_profit_usd / r2.max_drawdown_usd) *
                        r2.win_rate * np.sqrt(r2.total_trades))
        else:
            r2.score = 0.0
        all_results.append(r2)

    all_results.sort(key=lambda r: r.score, reverse=True)

    # Generate report
    gen  = ReportGenerator()
    path = gen.generate(all_results, best,
                        output_path="backtest_results/sample_report.pdf")
    print(f"\nSample report generated: {path}")
    print("Open it to preview the report layout before running the real backtest.")
