# ================================================================
#  quick_diagnostic.py  --  GoldScalperBot  v1.5
#  Usage: python quick_diagnostic.py
# ================================================================
import MetaTrader5 as mt5
from backtest_engine import (BacktestRunner, INSAMPLE_START, INSAMPLE_END,
                              DataLoader, ResultAnalyser)
from config import *

if not mt5.initialize(): print(f"MT5 failed: {mt5.last_error()}"); exit(1)
a = mt5.account_info()
if a is None: print("Not logged in"); mt5.shutdown(); exit(1)
print(f"MT5 connected -- {a.login}  Balance: ${a.balance:,.2f}\n")

loader = DataLoader(SYMBOL)
df_h4  = loader.load("H4",  INSAMPLE_START, INSAMPLE_END)
df_h1  = loader.load("H1",  INSAMPLE_START, INSAMPLE_END)
df_m15 = loader.load("M15", INSAMPLE_START, INSAMPLE_END)
print(f"\nH4:{len(df_h4)} | H1:{len(df_h1)} | M15:{len(df_m15)}\n")

runner = BacktestRunner()
print("Running v1.5 diagnostic...")
result = runner._run_single(df_h4, df_h1, df_m15)

trading_days = 756

print(f"\n{'='*70}")
print(f"  RESULTS -- Full version history")
print(f"{'='*70}")
print(f"  {'Metric':<25} {'v1.5':>12}   {'v1.3':>10}   {'v1.0':>10}")
print(f"  {'-'*60}")
print(f"  {'Total trades':<25} {result.total_trades:>12}   {'1,821':>10}   {'785':>10}")
print(f"  {'Trades/day':<25} {result.total_trades/trading_days:>12.1f}   {'2.4':>10}   {'1.0':>10}")
print(f"  {'Win rate':<25} {result.win_rate*100:>11.1f}%   {'73.8%':>10}   {'72.9%':>10}")
print(f"  {'Net profit':<25} ${result.net_profit_usd:>11,.2f}   {'$13,352':>10}   {'$13,659':>10}")
print(f"  {'Max drawdown':<25} {result.max_drawdown_pct:>11.1f}%   {'26.1%':>10}   {'37.8%':>10}")
print(f"  {'Profit factor':<25} {result.profit_factor:>12.2f}   {'1.82':>10}   {'2.04':>10}")
print(f"  {'Sharpe ratio':<25} {result.sharpe_ratio:>12.2f}")
print(f"  {'Avg win':<25} ${result.avg_win_usd:>11.2f}   {'$22.08':>10}   {'$46.78':>10}")
print(f"  {'Avg loss':<25} ${result.avg_loss_usd:>11.2f}   {'-$34.11':>10}   {'-$61.49':>10}")
print(f"  {'Score':<25} {result.score:>12.3f}   {'804.081':>10}   {'369.299':>10}")
print(f"{'='*70}")

by_year = {}
for t in result.trades:
    if t.open_time:
        y = t.open_time.year; by_year[y] = by_year.get(y,0) + 1
print(f"\n  Trades by year:")
for y in sorted(by_year):
    print(f"    {y}: {by_year[y]:>5}  (~{by_year[y]/252:.1f}/day)")

reasons = {}
for t in result.trades: reasons[t.exit_reason] = reasons.get(t.exit_reason,0) + 1
print(f"\n  Exit reasons:")
for r,c in sorted(reasons.items(), key=lambda x:-x[1]):
    pct = c / result.total_trades * 100 if result.total_trades > 0 else 0
    print(f"    {r:<15}: {c:>5}  ({pct:.1f}%)")

print(f"\n  [v1.5 ML-derived signal engine]")
print(f"    Primary signal    : H1 EMA 3-bar slope (ML rank 1: 17.03%)")
print(f"    Confirmation      : H1 EMA 5-bar slope (ML rank 3: 8.67%)")
print(f"    Timing trigger    : StochRSI cross")
print(f"    Filters applied   : ATR min, BB width, volume, price momentum, close pos")
print(f"    REMOVED           : H4 5-bar slope (most harmful: -0.023)")
print(f"    REMOVED           : NY session (harmful: -0.0003)")
print(f"    REMOVED           : RSI raw/5bar (harmful)")
print(f"    BE trigger        : {BE_PROFIT_TRIGGER_MULT:.1f}x spread (Optuna #1 param)")
print(f"    BE lock           : entry + {BE_SL_LOCK_MULT:.2f}x spread")
print(f"    SL multiplier     : {SL_ATR_MULTIPLIER:.2f}x ATR")
print(f"    TP R:R            : {TP_RR_RATIO:.2f}")
print(f"    Risk per trade    : {RISK_NORMAL_PCT*100:.2f}%")
print(f"    Concurrent trades : {MAX_CONCURRENT_TRADES}")
print(f"    Session           : London {LONDON_OPEN_H:02d}:00-{LONDON_CLOSE_H:02d}:00 UTC only")

mt5.shutdown()
print("\nDone.")
