# quick_diagnostic_v1.0.py
import MetaTrader5 as mt5
from backtest_engine import (BacktestRunner, INSAMPLE_START, INSAMPLE_END, DataLoader, ResultAnalyser)
from config import *

if not mt5.initialize(): print(f"MT5 failed: {mt5.last_error()}"); exit(1)
a = mt5.account_info()
if a is None: print("Not logged in"); mt5.shutdown(); exit(1)
print(f"MT5 connected — {a.login}  Balance: ${a.balance:,.2f}\n")

loader = DataLoader(SYMBOL)
df_h4  = loader.load("H4",  INSAMPLE_START, INSAMPLE_END)
df_h1  = loader.load("H1",  INSAMPLE_START, INSAMPLE_END)
df_m15 = loader.load("M15", INSAMPLE_START, INSAMPLE_END)
print(f"\nH4:{len(df_h4)} | H1:{len(df_h1)} | M15:{len(df_m15)}\n")

runner = BacktestRunner()
params = {"pivot_strength":5,"min_wave_pts":1200,
          "bos_method":"single_close","ehl_tolerance":"atr_0.5"}

print(f"Running v1.0 diagnostic...")
result = runner._run_single(df_h4, df_h1, df_m15, params)

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Total trades  : {result.total_trades}")
print(f"  Win rate      : {result.win_rate*100:.1f}%")
print(f"  Net profit    : ${result.net_profit_usd:,.2f}")
print(f"  Max drawdown  : {result.max_drawdown_pct:.1f}%")
print(f"  Score         : {result.score:.3f}")
print(f"  Profit factor : {result.profit_factor:.2f}")
print(f"  Avg win       : ${result.avg_win_usd:.2f}")
print(f"  Avg loss      : ${result.avg_loss_usd:.2f}")

by_year={}
for t in result.trades:
    if t.open_time:
        y=t.open_time.year; by_year[y]=by_year.get(y,0)+1
print(f"\n  Trades by year:")
for y in sorted(by_year):
    print(f"    {y}: {by_year[y]:>5}  (~{by_year[y]/252:.1f}/day)")

reasons={}
for t in result.trades: reasons[t.exit_reason]=reasons.get(t.exit_reason,0)+1
print(f"\n  Exit reasons:")
for r,c in sorted(reasons.items(),key=lambda x:-x[1]):
    print(f"    {r:<15}: {c}")

mt5.shutdown(); print("\nDone.")
