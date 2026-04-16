import pandas as pd
import os

# Find whatever trades CSV was saved
folder = "backtest_results"
files = [f for f in os.listdir(folder) if f.startswith("trades_")]

if files:
    df = pd.read_csv(os.path.join(folder, files[0]))
    print(f"File: {files[0]}")
    print(f"Total trades: {len(df)}")
    print()
    print(df[["direction","open_time","exit_reason","pnl_usd","signal_grade","lot_size"]].to_string())
else:
    print("No trades file found yet")