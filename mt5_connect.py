# ============================================================
#  mt5_connect.py
#  Step 1: Connect to MT5 and pull live XAUUSD. data
#  Run this first to confirm the entire pipeline works
# ============================================================

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# ---- CONFIGURATION -----------------------------------------
SYMBOL   = "XAUUSD..."   # Your exact MultiBank symbol
TIMEZONE = "Etc/UTC"   # MT5 always uses UTC
# ------------------------------------------------------------

def connect_mt5():
    """Initialize and connect to MT5 terminal"""

    print("=" * 50)
    print("  GoldScalperBot — MT5 Connection Test")
    print("=" * 50)

    # Step 1: Initialize MT5
    if not mt5.initialize():
        print(f"ERROR: MT5 initialize() failed")
        print(f"Error code : {mt5.last_error()}")
        print("")
        print("Make sure:")
        print("  1. MT5 is open and running")
        print("  2. You are logged into your MultiBank demo account")
        return False

    print("MT5 initialized successfully")

    # Step 2: Print terminal info
    terminal = mt5.terminal_info()
    if terminal:
        print(f"Terminal   : {terminal.name}")
        print(f"Path       : {terminal.path}")
        print(f"Connected  : {terminal.connected}")

    # Step 3: Print account info
    account = mt5.account_info()
    if account:
        print(f"\nAccount Info:")
        print(f"  Login    : {account.login}")
        print(f"  Server   : {account.server}")
        print(f"  Balance  : ${account.balance:,.2f}")
        print(f"  Equity   : ${account.equity:,.2f}")
        print(f"  Currency : {account.currency}")
        print(f"  Leverage : 1:{account.leverage}")
    else:
        print("WARNING: Could not read account info")

    return True


def check_symbol():
    """Verify XAUUSD.. symbol is available and get its properties"""

    print(f"\nChecking symbol: {SYMBOL}")
    print("-" * 40)

    # Check if symbol exists
    symbol_info = mt5.symbol_info(SYMBOL)

    if symbol_info is None:
        print(f"ERROR: Symbol '{SYMBOL}' not found")
        print("")
        print("Checking available gold symbols on your broker...")

        # Search for any gold-related symbols
        all_symbols = mt5.symbols_get()
        gold_symbols = [s.name for s in all_symbols
                       if "XAU" in s.name or "GOLD" in s.name.upper()]

        if gold_symbols:
            print(f"Found these gold symbols: {gold_symbols}")
            print(f"Update SYMBOL variable at top of this file with the correct name")
        else:
            print("No gold symbols found — check Market Watch in MT5")
        return False

    # Make sure symbol is visible in Market Watch
    if not symbol_info.visible:
        print(f"Symbol found but not visible — enabling...")
        mt5.symbol_select(SYMBOL, True)

    print(f"Symbol     : {symbol_info.name}")
    print(f"Description: {symbol_info.description}")
    print(f"Digits     : {symbol_info.digits}")
    print(f"Point      : {symbol_info.point}")
    print(f"Spread     : {symbol_info.spread} points")
    print(f"Min Lot    : {symbol_info.volume_min}")
    print(f"Max Lot    : {symbol_info.volume_max}")
    print(f"Lot Step   : {symbol_info.volume_step}")

    # Live price
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick:
        print(f"\nLive Price:")
        print(f"  Bid      : {tick.bid:.2f}")
        print(f"  Ask      : {tick.ask:.2f}")
        print(f"  Spread   : {(tick.ask - tick.bid) / symbol_info.point:.1f} points")
        print(f"  Time     : {datetime.utcfromtimestamp(tick.time)} UTC")

    return True


def get_h4_data(bars=100):
    """Pull H4 OHLCV data — this is our wave detection timeframe"""

    print(f"\nPulling H4 data ({bars} bars)...")
    print("-" * 40)

    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, bars)

    if rates is None or len(rates) == 0:
        print(f"ERROR: Could not get H4 data for {SYMBOL}")
        print(f"MT5 error: {mt5.last_error()}")
        return None

    # Convert to pandas DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open":  "Open",
        "high":  "High",
        "low":   "Low",
        "close": "Close",
        "tick_volume": "Volume"
    }, inplace=True)

    print(f"Bars received  : {len(df)}")
    print(f"First bar      : {df.index[0]}")
    print(f"Last bar       : {df.index[-1]}")
    print(f"Last close     : {df['Close'].iloc[-1]:.2f}")
    print(f"Last high      : {df['High'].iloc[-1]:.2f}")
    print(f"Last low       : {df['Low'].iloc[-1]:.2f}")

    print(f"\nLast 5 H4 candles:")
    print(df[["Open","High","Low","Close","Volume"]].tail(5).to_string())

    return df


def get_all_timeframes():
    """Pull data for all timeframes we need in the bot"""

    timeframes = {
        "M5" : mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1" : mt5.TIMEFRAME_H1,
        "H4" : mt5.TIMEFRAME_H4,
        "D1" : mt5.TIMEFRAME_D1,
    }

    print(f"\nChecking all timeframes...")
    print("-" * 40)

    results = {}
    for name, tf in timeframes.items():
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, 10)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            results[name] = df
            print(f"  {name:4s} : OK — last close: {rates[-1]['close']:.2f}  "
                  f"({len(rates)} bars received)")
        else:
            print(f"  {name:4s} : FAILED — {mt5.last_error()}")

    return results


def disconnect_mt5():
    """Clean shutdown"""
    mt5.shutdown()
    print("\nMT5 connection closed cleanly")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    # 1. Connect
    if not connect_mt5():
        print("\nConnection failed — fix the errors above and try again")
        exit(1)

    # 2. Check symbol
    if not check_symbol():
        mt5.shutdown()
        print("\nSymbol check failed — fix the errors above and try again")
        exit(1)

    # 3. Pull H4 data
    df_h4 = get_h4_data(bars=100)
    if df_h4 is None:
        mt5.shutdown()
        exit(1)

    # 4. Check all timeframes
    all_data = get_all_timeframes()

    # 5. Summary
    print("\n" + "=" * 50)
    print("  CONNECTION TEST COMPLETE")
    print("=" * 50)
    print(f"Symbol     : {SYMBOL}")
    print(f"H4 bars    : {len(df_h4)}")
    print(f"TFs working: {list(all_data.keys())}")
    print("")
    print("Pipeline confirmed — ready to build the bot")
    print("=" * 50)

    # 6. Disconnect
    disconnect_mt5()
