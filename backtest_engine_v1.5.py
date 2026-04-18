# ================================================================
#  backtest_engine_v1.5.py  --  GoldScalperBot  XAUUSD...
#  VERSION 1.5  --  ML-Optimized signal engine
#
#  Signal completely rebuilt from Random Forest permutation importance:
#
#  ESSENTIAL FEATURES (top 80% importance) now drive ALL entry logic:
#    1. h1_ema_slope3   17.03%  -- H1 EMA 3-bar slope (NEW primary)
#    2. price_mom_3bar   8.67%  -- 3-bar price momentum
#    3. h1_ema_slope5    8.67%  -- H1 EMA 5-bar slope (confirmation)
#    4. vol_vs_mean20    8.67%  -- volume vs 20-bar mean
#    5. price_mom_10bar  8.05%  -- 10-bar price momentum
#    6. atr_pts          7.74%  -- absolute ATR size
#    7. close_pos        5.57%  -- candle close position
#    8. bb_width         5.26%  -- Bollinger band width
#    9. is_london        4.95%  -- London session flag
#   10. atr_5bar_roc     4.02%  -- ATR rate of change
#
#  ACTIVELY HARMFUL features REMOVED:
#    - H4 5-bar slope  (-0.023)  WAS our core trend filter
#    - H1 price vs EMA (-0.013)
#    - bb_pct_b        (-0.011)
#    - RSI 5-bar slope (-0.011)
#    - H4 vs EMA20     (-0.010)
#    - StochRSI K      (-0.008)
#    - ATR ratio       (-0.006)
#    - body_ratio      (-0.005)
#    - ADX             (-0.003)
#    - NY session      (-0.0003)
#
#  Optuna results (200 trials):
#    be_trigger=7.42 | max_consec=7 | sl_atr=2.39 | tp_rr=2.76
#    risk=0.51% | max_concurrent=1 | London 09:00-14:00 UTC only
#
#  v1.3 baseline: $13,352 | 73.8% WR | 1,821 trades | 26.1% DD
#  Optuna target: $1,852  | 80.6% WR |   475 trades |  2.8% DD
#  Goal of v1.5:  Higher profit by applying ML signal to full data
#                 while keeping the optimal parameter set
#
#  DO NOT MODIFY -- locked reference for v1.5
# ================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import itertools, csv, os, time
from config import *

INSAMPLE_START  = datetime(2022, 2, 1,  tzinfo=timezone.utc)
INSAMPLE_END    = datetime(2024, 12, 31, tzinfo=timezone.utc)
OUTSAMPLE_START = datetime(2025, 1, 1,  tzinfo=timezone.utc)
OUTSAMPLE_END   = datetime(2026, 4, 8,  tzinfo=timezone.utc)
SLIPPAGE_POINTS = 3
RESULTS_DIR     = "backtest_results"
SUMMARY_FILE    = "summary.csv"

@dataclass
class BacktestTrade:
    trade_id: int; direction: str; open_time: datetime
    close_time: Optional[datetime]; entry_price: float
    exit_price: float = 0.0; sl_price: float = 0.0
    tp_price: float = 0.0; lot_size: float = 0.0
    spread_at_entry: float = 0.0; signal_score: int = 0
    signal_grade: str = "B"; prob_weight: float = 1.0
    pnl_usd: float = 0.0; pnl_points: float = 0.0
    exit_reason: str = ""; pivot_strength: int = 0
    is_open: bool = True

@dataclass
class BacktestResult:
    params: dict
    trades: List[BacktestTrade] = field(default_factory=list)
    net_profit_usd: float = 0.0; max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0; win_rate: float = 0.0
    total_trades: int = 0; winning_trades: int = 0
    losing_trades: int = 0; avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0; profit_factor: float = 0.0
    sharpe_ratio: float = 0.0; score: float = 0.0
    equity_curve: List[float] = field(default_factory=list)

class DataLoader:
    def __init__(self, symbol=SYMBOL):
        self.symbol = symbol
        self._cache: Dict[str, pd.DataFrame] = {}

    def load(self, tf_str, start, end):
        key = f"{tf_str}_{start.date()}_{end.date()}"
        if key in self._cache: return self._cache[key]
        tf_map = {
            "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(tf_str)
        if tf is None: raise ValueError(f"Unknown timeframe: {tf_str}")
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, 99999)
        if rates is None or len(rates) == 0:
            print(f"DataLoader WARNING: No {tf_str} data -- {mt5.last_error()}")
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"open":"Open","high":"High","low":"Low",
                            "close":"Close","tick_volume":"Volume",
                            "spread":"Spread"}, inplace=True)
        start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_utc   = end   if end.tzinfo   else end.replace(tzinfo=timezone.utc)
        df = df[(df.index >= start_utc) & (df.index <= end_utc)]
        if len(df) == 0:
            print(f"DataLoader WARNING: {tf_str} no bars {start.date()} to {end.date()}")
            return pd.DataFrame()
        df["BodySize"] = abs(df["Close"] - df["Open"])
        df["Range"]    = df["High"] - df["Low"]
        df["IsBullish"]= df["Close"] > df["Open"]
        print(f"  {tf_str:4s} loaded: {len(df):>6} bars "
              f"({df.index[0].date()} to {df.index[-1].date()})")
        self._cache[key] = df
        return df

class Indicators:
    @staticmethod
    def ema(s, p): return s.ewm(span=p, adjust=False).mean()
    @staticmethod
    def rsi(close, period=14):
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period-1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
        return 100 - (100 / (1 + g / l.replace(0, np.nan)))
    @staticmethod
    def atr(h, l, c, period=14):
        tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        return tr.ewm(com=period-1, adjust=False).mean()
    @staticmethod
    def rsi_slope(rsi, lb=3): return rsi.diff(lb)
    @staticmethod
    def stoch_rsi(close, rp=14, sp=14, kp=3, dp=3):
        rsi = Indicators.rsi(close, rp)
        lo  = rsi.rolling(sp).min(); hi = rsi.rolling(sp).max()
        k   = (100*(rsi-lo)/(hi-lo).replace(0, np.nan)).rolling(kp).mean()
        return k, k.rolling(dp).mean()
    @staticmethod
    def volume_zscore(vol, period=20):
        m = vol.rolling(period).mean()
        s = vol.rolling(period).std()
        return (vol-m)/s.replace(0, np.nan)
    @staticmethod
    def bollinger(close, period=20, std_dev=2.0):
        mid   = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        width = (upper - lower) / (mid.replace(0, np.nan))
        return mid, upper, lower, width

class TradeSimulator:
    def __init__(self, starting_balance=ACCOUNT_BALANCE):
        self.balance       = starting_balance
        self.equity        = starting_balance
        self.peak_equity   = starting_balance
        self.open_trades:   List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []
        self.trade_counter = 0
        self.daily_loss    = 0.0
        self.weekly_loss   = 0.0
        self.session_losses= 0
        self.equity_curve  = [starting_balance]
        self.session_trade_count = 0
        self.current_session_key = None

    def reset(self, b=ACCOUNT_BALANCE): self.__init__(b)

    def calc_lot_size(self, sl_pts):
        risk = self.equity * RISK_NORMAL_PCT
        base = risk / (sl_pts * CONTRACT_SIZE * POINT)
        lot  = round(base / LOT_STEP) * LOT_STEP
        return max(LOT_MIN, min(LOT_MAX, round(lot, 2)))

    def open_trade(self, direction, entry_price, sl_price, tp_price,
                   lot_size, spread, open_time):
        self.trade_counter += 1
        sp = spread * POINT
        if direction == "LONG":
            adj_e  = entry_price + SLIPPAGE_POINTS * POINT
            adj_sl = sl_price  - sp
            adj_tp = tp_price  + sp
        else:
            adj_e  = entry_price - SLIPPAGE_POINTS * POINT
            adj_sl = sl_price  + sp
            adj_tp = tp_price  - sp
        t = BacktestTrade(
            trade_id=self.trade_counter, direction=direction,
            open_time=open_time, close_time=None,
            entry_price=adj_e, sl_price=adj_sl, tp_price=adj_tp,
            lot_size=lot_size, spread_at_entry=spread, is_open=True)
        self.open_trades.append(t)
        self.session_trade_count += 1
        return t

    def update_trades(self, bar, bar_time, atr_val, rsi_slope_val):
        to_close = []
        for t in self.open_trades:
            cp = None; reason = ""
            sp      = t.spread_at_entry * POINT
            # v1.5: BE trigger from Optuna (7.4x spread)
            be_min  = sp * BE_PROFIT_TRIGGER_MULT

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
                elif cur > be_min and rsi_slope_val < -5 and bar["BodySize"] < bar["Range"]*0.3:
                    cp, reason = bar["Close"], "MOMENTUM"
                elif cur > be_min and t.sl_price < t.entry_price:
                    # v1.5: BE lock = entry + 1.41x spread (Optuna be_lock)
                    new_sl = t.entry_price + sp * BE_SL_LOCK_MULT
                    if new_sl > t.sl_price:
                        t.sl_price = new_sl
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
                elif cur > be_min and rsi_slope_val > 5 and bar["BodySize"] < bar["Range"]*0.3:
                    cp, reason = bar["Close"], "MOMENTUM"
                elif cur > be_min and t.sl_price > t.entry_price:
                    new_sl = t.entry_price - sp * BE_SL_LOCK_MULT
                    if new_sl < t.sl_price:
                        t.sl_price = new_sl
                        t.tp_price = bar["Low"] - atr_val * TP_TRAIL_ATR_MULT

            if cp is not None:
                to_close.append((t, cp, reason, bar_time))
        for t, cp, r, ct in to_close:
            self._close_trade(t, cp, r, ct)

    def _close_trade(self, t, cp, reason, ct):
        pnl_pts = (cp-t.entry_price)/POINT if t.direction=="LONG" \
                  else (t.entry_price-cp)/POINT
        pnl_usd = pnl_pts * POINT * CONTRACT_SIZE * t.lot_size
        t.exit_price=cp; t.close_time=ct
        t.pnl_usd=round(pnl_usd,2); t.pnl_points=round(pnl_pts,1)
        t.exit_reason=reason; t.is_open=False
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
        if self.daily_loss    >= MAX_DAILY_LOSS_USD:          return False, "Daily limit"
        if self.session_losses >= MAX_CONSEC_LOSSES:           return False, "Consec losses"
        if self.weekly_loss   >= WEEKLY_DD_USD:                return False, "Weekly DD"
        if self.session_trade_count >= MAX_TRADES_PER_SESS:    return False, "Session limit"
        if self.equity <= self.balance * 0.50:                 return False, "Equity 50%"
        return True, "OK"

class ResultAnalyser:
    @staticmethod
    def analyse(sim, params):
        t = sim.closed_trades; r = BacktestResult(params=params)
        r.trades=t; r.equity_curve=sim.equity_curve; r.total_trades=len(t)
        if r.total_trades==0: r.score=0.0; return r
        pnls = [x.pnl_usd for x in t]
        r.net_profit_usd = sum(pnls)
        wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
        r.winning_trades=len(wins); r.losing_trades=len(losses)
        r.win_rate = len(wins)/r.total_trades
        r.avg_win_usd  = np.mean(wins)   if wins   else 0.0
        r.avg_loss_usd = np.mean(losses) if losses else 0.0
        gp=sum(wins); gl=abs(sum(losses))
        r.profit_factor = gp/gl if gl>0 else 999.0
        eq=sim.equity_curve; peak=eq[0]; max_dd=0.0
        for e in eq: peak=max(peak,e); max_dd=max(max_dd,peak-e)
        r.max_drawdown_usd = max_dd
        r.max_drawdown_pct = (max_dd/ACCOUNT_BALANCE)*100
        if len(pnls)>1:
            s=np.std(pnls); m=np.mean(pnls)
            r.sharpe_ratio = m/s*np.sqrt(252) if s>0 else 0.0
        if r.max_drawdown_usd>0 and r.total_trades>=10:
            r.score = (r.net_profit_usd/r.max_drawdown_usd)*r.win_rate*np.sqrt(r.total_trades)
        return r

    @staticmethod
    def print_summary(r):
        print("="*55)
        print(f"  Params : {r.params}")
        print(f"  Trades : {r.total_trades} | WR: {r.win_rate*100:.1f}%")
        print(f"  Profit : ${r.net_profit_usd:.2f}")
        print(f"  Max DD : ${r.max_drawdown_usd:.2f} ({r.max_drawdown_pct:.1f}%)")
        print(f"  PF     : {r.profit_factor:.2f} | Sharpe: {r.sharpe_ratio:.2f}")
        print(f"  SCORE  : {r.score:.3f}")
        print("="*55)

class BacktestLogger:
    def __init__(self, output_dir=RESULTS_DIR):
        os.makedirs(output_dir, exist_ok=True); self.output_dir=output_dir
    def save_trades(self, result, run_id):
        path=os.path.join(self.output_dir,f"trades_{run_id}.csv")
        fields=["trade_id","direction","open_time","close_time","entry_price",
                "exit_price","sl_price","tp_price","lot_size","spread_at_entry",
                "pnl_usd","pnl_points","exit_reason"]
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for t in result.trades: w.writerow({k:getattr(t,k,"") for k in fields})

class BacktestRunner:
    def __init__(self):
        self.loader    = DataLoader(SYMBOL)
        self.simulator = TradeSimulator()
        self.logger    = BacktestLogger()

    def _run_single(self, df_h4, df_h1, df_m15, params={}):
        self.simulator.reset()
        daily_reset_date = None
        weekly_reset_key = None

        df_h4=df_h4.copy(); df_h1=df_h1.copy(); df_m15=df_m15.copy()

        def _norm(df):
            m={}
            for c in df.columns:
                cl=c.lower()
                if   cl=="open":                   m[c]="Open"
                elif cl=="high":                   m[c]="High"
                elif cl=="low":                    m[c]="Low"
                elif cl=="close":                  m[c]="Close"
                elif cl in("volume","tick_volume"):m[c]="Volume"
                elif cl=="spread":                 m[c]="Spread"
            return df.rename(columns=m)
        df_h4=_norm(df_h4); df_h1=_norm(df_h1); df_m15=_norm(df_m15)

        # ---- Pre-compute indicators --------------------------
        # H1: EMA slopes (features #1 and #3 -- most important)
        df_h1["EMA20"]        = Indicators.ema(df_h1["Close"], 20)
        df_h1["EMA20slope3"]  = df_h1["EMA20"].diff(3)   # 3-bar slope (rank 1)
        df_h1["EMA20slope5"]  = df_h1["EMA20"].diff(5)   # 5-bar slope (rank 3)
        df_h1["RSI14"]        = Indicators.rsi(df_h1["Close"], 14)

        # M15: price momentum, ATR, volume, Bollinger, RSI slope
        df_m15["ATR14"]       = Indicators.atr(df_m15["High"],df_m15["Low"],df_m15["Close"],14)
        df_m15["RSI14"]       = Indicators.rsi(df_m15["Close"], 14)
        df_m15["RSIslope3"]   = df_m15["RSI14"].diff(3)   # 3-bar RSI slope (rank 15)
        df_m15["RSIslope"]    = df_m15["RSI14"].diff(3)   # for momentum exit
        df_m15["VolZ"]        = Indicators.volume_zscore(df_m15["Volume"], 20)
        df_m15["VolMean20"]   = df_m15["Volume"].rolling(20).mean()
        df_m15["VolVsMean20"] = (df_m15["Volume"] - df_m15["VolMean20"]) / (df_m15["VolMean20"] + 1e-9)

        # Bollinger bands (rank 8: width is essential, pct_b is harmful)
        _, bb_upper, bb_lower, bb_width = Indicators.bollinger(df_m15["Close"], 20, 2.0)
        df_m15["BBWidth"]     = bb_width

        # ATR 5-bar rate of change (rank 10: 4.02%)
        df_m15["ATR5ROC"]     = df_m15["ATR14"].pct_change(5)

        # StochRSI (for entry confirmation -- stoch_oversold rank 14)
        stk, std              = Indicators.stoch_rsi(df_m15["Close"])
        df_m15["StochK"]      = stk
        df_m15["StochD"]      = std

        for i in range(50, len(df_m15)):
            bar_time = df_m15.index[i]
            bar      = df_m15.iloc[i]

            # Daily / weekly resets
            bar_date = bar_time.date()
            if bar_date != daily_reset_date:
                self.simulator.daily_loss     = 0.0
                self.simulator.session_losses = 0
                daily_reset_date = bar_date

            week_key = f"{bar_time.year}_{bar_time.isocalendar()[1]}"
            if week_key != weekly_reset_key:
                self.simulator.weekly_loss = 0.0
                weekly_reset_key = week_key

            hour = bar_time.hour

            # v1.5: London ONLY (NY removed -- harmful per ML)
            in_london = (LONDON_OPEN_H <= hour < LONDON_CLOSE_H)
            in_session = in_london

            if in_london:
                sess_key = f"london_{bar_date}"
            else:
                sess_key = None

            if sess_key and sess_key != self.simulator.current_session_key:
                self.simulator.current_session_key = sess_key
                self.simulator.session_trade_count = 0

            # Get bar values safely
            atr_val    = bar["ATR14"]     if not pd.isna(bar["ATR14"])     else 10.0
            rsi_slp    = bar["RSIslope"]  if not pd.isna(bar["RSIslope"])  else 0.0
            rsi_slp3   = bar["RSIslope3"] if not pd.isna(bar["RSIslope3"]) else 0.0
            vol_z      = bar["VolZ"]      if not pd.isna(bar["VolZ"])      else 0.0
            vol_vs_mean= bar["VolVsMean20"] if not pd.isna(bar["VolVsMean20"]) else 0.0
            bb_w       = bar["BBWidth"]   if not pd.isna(bar["BBWidth"])   else 0.05
            atr_5roc   = bar["ATR5ROC"]   if not pd.isna(bar["ATR5ROC"])   else 0.0

            # Update open trades on every bar
            self.simulator.update_trades(bar, bar_time, atr_val, rsi_slp)

            if not in_session: continue

            # Circuit breaker check
            allowed, _ = self.simulator.is_trading_allowed()
            if not allowed: continue

            # v1.5: single trade maximum (Optuna: max_concurrent=1)
            if len(self.simulator.open_trades) >= MAX_CONCURRENT_TRADES: continue

            # Spread filter
            spread_pts = bar.get("Spread", 25)
            if pd.isna(spread_pts): spread_pts = 25
            if spread_pts > MAX_SPREAD_PTS: continue

            # Get H1 context (last 10 bars)
            h1_bars = df_h1[df_h1.index <= bar_time].tail(10)
            if len(h1_bars) < 5: continue
            h1_last = h1_bars.iloc[-1]

            # ====================================================
            #  v1.5 SIGNAL ENGINE (ML-derived, feature-ranked)
            # ====================================================

            # --- Feature 1: H1 EMA 3-bar slope (rank 1: 17.03%) ---
            # The most important feature. Direction of EMA acceleration.
            h1_ema_slope3 = h1_last.get("EMA20slope3", 0.0) if hasattr(h1_last,"get") else 0.0
            if pd.isna(h1_ema_slope3): h1_ema_slope3 = 0.0

            # --- Feature 3: H1 EMA 5-bar slope (rank 3: 8.67%) ---
            # Confirmation of slope direction over longer window
            h1_ema_slope5 = h1_last.get("EMA20slope5", 0.0) if hasattr(h1_last,"get") else 0.0
            if pd.isna(h1_ema_slope5): h1_ema_slope5 = 0.0

            # --- Feature 6: ATR points minimum (rank 6: 7.74%) ---
            # Skip if market is too quiet
            atr_pts = atr_val / POINT
            if atr_pts < ATR_PTS_MIN: continue

            # --- Feature 8: Bollinger band width (rank 8: 5.26%) ---
            # Skip if bands too narrow (no momentum) or too wide (late)
            if not (BB_WIDTH_MIN <= bb_w <= BB_WIDTH_MAX): continue

            # --- Feature 4: Volume vs 20-bar mean (rank 4: 8.67%) ---
            # Require above-average volume (loose threshold from Optuna)
            if vol_vs_mean < FILTER_VOL_VS_MEAN_MIN: continue

            # --- DIRECTION DECISION ---
            # Primary: H1 EMA slope direction (feature #1)
            # Confirmation: EMA slope5 agrees (feature #3)
            # Entry: StochRSI cross (oversold rank 14 / pattern)

            stk_k = bar.get("StochK", 50.0)
            stk_d = bar.get("StochD", 50.0)
            if pd.isna(stk_k): stk_k = 50.0
            if pd.isna(stk_d): stk_d = 50.0
            prev   = df_m15.iloc[i-1]
            prev_k = prev.get("StochK", 50.0)
            prev_d = prev.get("StochD", 50.0)
            if pd.isna(prev_k): prev_k = 50.0
            if pd.isna(prev_d): prev_d = 50.0

            bull_cross = (stk_k > stk_d and prev_k <= prev_d and stk_k < STOCH_BULL_CAP)
            bear_cross = (stk_k < stk_d and prev_k >= prev_d and stk_k > STOCH_BEAR_FLOOR)

            # LONG signal:
            #   H1 EMA 3-bar slope positive AND accelerating (slope5 also positive)
            #   StochRSI bullish cross as timing trigger
            long_signal = (h1_ema_slope3 > H1_EMA_SLOPE3_MIN_LONG and
                           h1_ema_slope5 > 0 and
                           bull_cross)

            # SHORT signal:
            #   H1 EMA 3-bar slope negative AND decelerating (slope5 also negative)
            #   StochRSI bearish cross as timing trigger
            short_signal = (h1_ema_slope3 < H1_EMA_SLOPE3_MAX_SHORT and
                            h1_ema_slope5 < 0 and
                            bear_cross)

            if not (long_signal or short_signal): continue

            direction = "LONG" if long_signal else "SHORT"

            # --- Feature 2: Price 3-bar momentum (rank 2: 8.67%) ---
            # Confirm momentum direction matches signal
            close_arr = df_m15["Close"].values
            if i >= 3:
                price_mom_3bar = (close_arr[i] - close_arr[i-3]) / (atr_val + 1e-9)
            else:
                price_mom_3bar = 0.0

            if direction == "LONG"  and price_mom_3bar < -2.0: continue  # don't buy into strong selloff
            if direction == "SHORT" and price_mom_3bar >  2.0: continue  # don't sell into strong rally

            # --- Feature 5: Price 10-bar momentum (rank 5: 8.05%) ---
            # Confirm medium-term momentum
            if i >= 10:
                price_mom_10bar = (close_arr[i] - close_arr[i-10]) / (atr_val + 1e-9)
            else:
                price_mom_10bar = 0.0

            if direction == "LONG"  and price_mom_10bar < -3.0: continue
            if direction == "SHORT" and price_mom_10bar >  3.0: continue

            # --- Feature 7: Close position in candle (rank 7: 5.57%) ---
            # For LONG: prefer close in upper half of candle
            # For SHORT: prefer close in lower half of candle
            rng = bar["Range"] if bar["Range"] > 0 else 1.0
            close_pos = (bar["Close"] - bar["Low"]) / rng
            if direction == "LONG"  and close_pos < 0.25: continue  # closing near lows -- skip
            if direction == "SHORT" and close_pos > 0.75: continue  # closing near highs -- skip

            # --- Feature 10: ATR 5-bar ROC (rank 10: 4.02%) ---
            # Skip if volatility is collapsing (ATR falling fast)
            if direction == "LONG"  and atr_5roc < -0.30: continue  # volatility dying
            if direction == "SHORT" and atr_5roc < -0.30: continue

            # --- Feature 15: RSI 3-bar slope (rank 15: 2.17%) ---
            # Mild filter -- not too extreme
            if direction == "LONG"  and rsi_slp3 > RSI_3BAR_SLOPE_MIN_LONG:  pass  # ok
            if direction == "SHORT" and rsi_slp3 < RSI_3BAR_SLOPE_MAX_SHORT: pass  # ok

            # ====================================================
            #  POSITION SIZING & EXECUTION
            # ====================================================

            # ATR-based SL/TP (Optuna: wider SL, better R:R)
            sl_pts = int(atr_pts * SL_ATR_MULTIPLIER)
            sl_pts = max(int(atr_pts*SL_ATR_MIN_MULT),
                         min(int(atr_pts*SL_ATR_MAX_MULT), sl_pts))
            sl_pts = max(sl_pts, 120)  # minimum 120 pts on Gold
            tp_pts = int(sl_pts * TP_RR_RATIO)

            entry = bar["Close"]
            sl = entry - sl_pts*POINT if direction=="LONG" else entry + sl_pts*POINT
            tp = entry + tp_pts*POINT if direction=="LONG" else entry - tp_pts*POINT

            lot = self.simulator.calc_lot_size(sl_pts)
            if lot <= 0: continue

            # Day-of-week lot scaling (Optuna found near-full is optimal)
            dow = bar_time.weekday()
            if dow == 0:
                lot = max(LOT_MIN, round(round(lot*FILTER_MONDAY_LOT/LOT_STEP)*LOT_STEP, 2))
            elif dow == 3:
                lot = max(LOT_MIN, round(round(lot*FILTER_THURSDAY_LOT/LOT_STEP)*LOT_STEP, 2))

            self.simulator.open_trade(
                direction, entry, sl, tp, lot, spread_pts, bar_time)

        # Close any remaining open trades at backtest end
        if self.simulator.open_trades:
            last_bar = df_m15.iloc[-1]
            for t in list(self.simulator.open_trades):
                self.simulator._close_trade(
                    t, last_bar["Close"], "BACKTEST_END", df_m15.index[-1])

        return ResultAnalyser.analyse(self.simulator, params)

if __name__ == "__main__":
    print("="*60)
    print("  GoldScalperBot v1.5  --  XAUUSD...")
    print("  ML-Optimized Signal Engine")
    print("="*60)
    if not mt5.initialize():
        print(f"MT5 failed: {mt5.last_error()}"); exit(1)
    a = mt5.account_info()
    if a is None:
        print("MT5 not logged in"); mt5.shutdown(); exit(1)
    print(f"MT5 connected -- {a.login}  Balance: ${a.balance:,.2f}\n")

    runner = BacktestRunner()
    df_h4  = runner.loader.load("H4",  INSAMPLE_START, INSAMPLE_END)
    df_h1  = runner.loader.load("H1",  INSAMPLE_START, INSAMPLE_END)
    df_m15 = runner.loader.load("M15", INSAMPLE_START, INSAMPLE_END)

    print("\nRunning v1.5 in-sample backtest...")
    result = runner._run_single(df_h4, df_h1, df_m15)
    ResultAnalyser.print_summary(result)

    runner.logger.save_trades(result, f"v1.5_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    mt5.shutdown(); print("Done.")
