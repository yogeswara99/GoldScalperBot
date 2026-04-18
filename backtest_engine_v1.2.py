# ================================================================
#  backtest_engine_v1.2.py  —  GoldScalperBot  XAUUSD...
#  VERSION 1.2  —  High-frequency spread-only model
#
#  Changes from v1.1:
#    1. Spread-only cost model — COMMISSION_USD=0, MAX_SPREAD_PTS=20
#       filter blocks entry if spread > 20 pts at signal bar
#    2. Up to 3 concurrent trades (was 1) — primary frequency lever
#    3. H4 trend reverted to 5-bar close slope (v1.0 method)
#       EMA20/EMA50 was too restrictive, cutting ~30% of signals
#    4. H1 condition loosened: RSI>45 OR slope>0 (was AND both)
#       Captures pullback entries where one condition lags
#    5. StochRSI thresholds widened: bull<85, bear>15 (was 80/20)
#    6. Asian session added: 01:00–06:00 UTC
#    7. London extended: 07:00–12:00 UTC (was 08:30–11:00)
#    8. NY extended: 12:00–17:00 UTC (was 13:00–16:00)
#    9. London open delay removed (07:00 is already past the gap)
#   10. Volume filter: 0.0 (reverted to v1.0 — 0.5 was too aggressive)
#   11. Weekly DD enforced at 12% (v1.1 fix kept)
#   12. Session trade limit raised to 20 (effectively removed)
#
#  Baseline (v1.0): $13,659 | 72.9% WR | 785 trades | 37.8% DD
#  v1.1 result:     $3,936  | 70.2% WR | 584 trades | 19.8% DD
#  Target  (v1.2):  profit > v1.0 | DD <= 30% | trades 5-6/day
#
#  DO NOT MODIFY — locked reference for v1.2
# ================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import itertools, csv, os, time
from config import *

# ---- BACKTEST CONFIG ----------------------------------------
INSAMPLE_START  = datetime(2022, 2, 1,  tzinfo=timezone.utc)
INSAMPLE_END    = datetime(2024, 12, 31, tzinfo=timezone.utc)
OUTSAMPLE_START = datetime(2025, 1, 1,  tzinfo=timezone.utc)
OUTSAMPLE_END   = datetime(2026, 4, 8,  tzinfo=timezone.utc)
SLIPPAGE_POINTS = 3
RESULTS_DIR     = "backtest_results"
TRADES_FILE     = "trades_log.csv"
SUMMARY_FILE    = "summary.csv"

# ---- DATA CLASSES -------------------------------------------
@dataclass
class BacktestTrade:
    trade_id: int; direction: str; open_time: datetime
    close_time: Optional[datetime]; entry_price: float
    exit_price: float = 0.0; sl_price: float = 0.0
    tp_price: float = 0.0; lot_size: float = 0.0
    spread_at_entry: float = 0.0; signal_score: int = 0
    signal_grade: str = "B"; prob_weight: float = 1.0
    pnl_usd: float = 0.0; pnl_points: float = 0.0
    exit_reason: str = ""; bos_method: str = ""
    choch_scope: str = ""; ehl_tolerance: str = ""
    pivot_strength: int = 0; is_open: bool = True

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

# ---- DATA LOADER --------------------------------------------
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
            print(f"DataLoader WARNING: No {tf_str} data — {mt5.last_error()}")
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

# ---- INDICATORS ---------------------------------------------
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

# ---- SCORING ENGINE -----------------------------------------
class ScoringEngine:
    def __init__(self, weights=None):
        self.weights = weights or {
            "elliott_bias":  0.20, "smc_structure": 0.25,
            "fvg_confluence":0.15, "h1_momentum":   0.15,
            "m15_bos":       0.15, "volume_confirm": 0.10,
        }
    def calculate(self, conditions):
        ws = sum(w for k,w in self.weights.items() if conditions.get(k, False))
        rc = sum(1 for k in self.weights if conditions.get(k, False))
        grade = "A+" if ws>=0.85 else "B" if ws>=0.65 else "C" if ws>=0.45 else "SKIP"
        return ws, rc, grade
    def lot_multiplier(self, pw):
        return 1.00 if pw>=0.85 else 0.75 if pw>=0.65 else 0.50 if pw>=0.45 else 0.00

# ---- TRADE SIMULATOR ----------------------------------------
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
        self.session_trade_count  = 0
        self.current_session_key  = None

    def reset(self, b=ACCOUNT_BALANCE): self.__init__(b)

    def calc_lot_size(self, sl_pts, pw, scorer):
        risk = self.equity * RISK_NORMAL_PCT
        base = risk / (sl_pts * CONTRACT_SIZE * POINT)
        raw  = base * scorer.lot_multiplier(pw)
        lot  = round(raw / LOT_STEP) * LOT_STEP
        return max(LOT_MIN, min(LOT_MAX, round(lot, 2)))

    def open_trade(self, direction, entry_price, sl_price, tp_price,
                   lot_size, spread, signal_score, grade,
                   prob_weight, open_time, params):
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
            lot_size=lot_size, spread_at_entry=spread,
            signal_score=signal_score, signal_grade=grade,
            prob_weight=prob_weight,
            bos_method=params.get("bos_method",""),
            choch_scope=params.get("choch_scope",""),
            ehl_tolerance=params.get("ehl_tolerance",""),
            pivot_strength=params.get("pivot_strength",5), is_open=True)
        self.open_trades.append(t)
        self.session_trade_count += 1
        return t

    def update_trades(self, bar, bar_time, atr_val, rsi_val, rsi_slope, vol_z):
        to_close = []
        for t in self.open_trades:
            cp = None; reason = ""
            sp  = t.spread_at_entry * POINT
            min_profit = sp * 3.5
            if t.direction == "LONG":
                cur    = bar["Close"] - t.entry_price
                sl_hit = bar["Low"]  <= t.sl_price
                tp_hit = bar["High"] >= t.tp_price
                if sl_hit and tp_hit:
                    # v1.1+ fix: distance from open resolves conflict
                    dist_sl = abs(bar["Open"] - t.sl_price)
                    dist_tp = abs(bar["Open"] - t.tp_price)
                    cp, reason = (t.tp_price,"TP") if dist_tp<=dist_sl else (t.sl_price,"SL")
                elif sl_hit: cp, reason = t.sl_price, "SL"
                elif tp_hit: cp, reason = t.tp_price, "TP"
                elif cur>min_profit and rsi_slope<-5 and bar["BodySize"]<bar["Range"]*0.3:
                    cp, reason = bar["Close"], "MOMENTUM"
                elif cur>min_profit and t.sl_price < t.entry_price:
                    t.sl_price = t.entry_price + sp
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
                elif cur>min_profit and rsi_slope>5 and bar["BodySize"]<bar["Range"]*0.3:
                    cp, reason = bar["Close"], "MOMENTUM"
                elif cur>min_profit and t.sl_price > t.entry_price:
                    t.sl_price = t.entry_price - sp
                    t.tp_price = bar["Low"] - atr_val * TP_TRAIL_ATR_MULT
            if cp is not None:
                to_close.append((t, cp, reason, bar_time))
        for t, cp, r, ct in to_close:
            self._close_trade(t, cp, r, ct)

    def _close_trade(self, t, cp, reason, ct):
        pnl_pts = (cp-t.entry_price)/POINT if t.direction=="LONG" \
                  else (t.entry_price-cp)/POINT
        # v1.2: spread-only model — no fixed commission
        pnl_usd = pnl_pts * POINT * CONTRACT_SIZE * t.lot_size
        t.exit_price=cp; t.close_time=ct
        t.pnl_usd=round(pnl_usd,2); t.pnl_points=round(pnl_pts,1)
        t.exit_reason=reason; t.is_open=False
        self.balance  += pnl_usd
        self.equity    = self.balance
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(round(self.equity, 2))
        if pnl_usd < 0:
            self.daily_loss   += abs(pnl_usd)
            self.weekly_loss  += abs(pnl_usd)
            self.session_losses += 1
        else:
            self.session_losses = 0
        self.open_trades.remove(t)
        self.closed_trades.append(t)

    def is_trading_allowed(self):
        if self.daily_loss   >= MAX_DAILY_LOSS_USD:          return False, "Daily limit"
        if self.session_losses >= MAX_CONSEC_LOSSES:          return False, "Consec losses"
        if self.weekly_loss  >= WEEKLY_DD_USD:                return False, "Weekly DD"
        if self.session_trade_count >= MAX_TRADES_PER_SESS:   return False, "Session limit"
        if self.equity <= self.balance * 0.50:                return False, "Equity 50%"
        return True, "OK"

# ---- RESULT ANALYSER ----------------------------------------
class ResultAnalyser:
    @staticmethod
    def analyse(sim, params):
        t = sim.closed_trades; r = BacktestResult(params=params)
        r.trades=t; r.equity_curve=sim.equity_curve; r.total_trades=len(t)
        if r.total_trades==0: r.score=0.0; return r
        pnls = [x.pnl_usd for x in t]
        r.net_profit_usd = sum(pnls)
        wins   = [p for p in pnls if p>0]
        losses = [p for p in pnls if p<=0]
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

# ---- CSV LOGGER ---------------------------------------------
class BacktestLogger:
    def __init__(self, output_dir=RESULTS_DIR):
        os.makedirs(output_dir, exist_ok=True); self.output_dir=output_dir
    def save_trades(self, result, run_id):
        path=os.path.join(self.output_dir,f"trades_{run_id}.csv")
        fields=["trade_id","direction","open_time","close_time","entry_price",
                "exit_price","sl_price","tp_price","lot_size","spread_at_entry",
                "signal_score","signal_grade","prob_weight","pnl_usd","pnl_points",
                "exit_reason","bos_method","choch_scope","ehl_tolerance","pivot_strength"]
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for t in result.trades: w.writerow({k:getattr(t,k,"") for k in fields})
    def save_summary(self, results):
        if not results: return
        path=os.path.join(self.output_dir,SUMMARY_FILE)
        pk=list(results[0].params.keys())
        mk=["total_trades","win_rate","net_profit_usd","max_drawdown_usd",
            "max_drawdown_pct","profit_factor","sharpe_ratio","score"]
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=pk+mk); w.writeheader()
            for r in results:
                row={k:r.params.get(k,"") for k in pk}
                for m in mk: row[m]=round(getattr(r,m,0),4)
                w.writerow(row)
        print(f"\nSummary: {path}")

# ---- TIER 1 GRID -------------------------------------------
TIER1_GRID = {
    "pivot_strength": [3, 5, 8],
    "min_wave_pts":   [800, 1200, 1600],
    "bos_method":     ["single_close", "double_close", "body_close"],
    "ehl_tolerance":  ["fixed_100", "atr_0.3", "atr_0.5", "pct_0.02"],
}

# ---- BACKTEST RUNNER ----------------------------------------
class BacktestRunner:
    def __init__(self):
        self.loader    = DataLoader(SYMBOL)
        self.scorer    = ScoringEngine()
        self.simulator = TradeSimulator()
        self.logger    = BacktestLogger()

    def _generate_param_combos(self, grid):
        keys=list(grid.keys()); vals=list(grid.values())
        return [dict(zip(keys,c)) for c in itertools.product(*vals)]

    def run_tier1(self, period="insample"):
        start,end = (INSAMPLE_START,INSAMPLE_END) if period=="insample" \
                    else (OUTSAMPLE_START,OUTSAMPLE_END)
        print(f"\nTier 1 — {'In-sample' if period=='insample' else 'Out-of-sample'}")
        print(f"Period: {start.date()} to {end.date()}")
        df_h4 = self.loader.load("H4",  start, end)
        df_h1 = self.loader.load("H1",  start, end)
        df_m15= self.loader.load("M15", start, end)
        if df_h4.empty or df_m15.empty:
            print("ERROR: Could not load data."); return []
        combos  = self._generate_param_combos(TIER1_GRID)
        results = []
        print(f"\nRunning {len(combos)} combinations...")
        print("-"*55)
        for idx, params in enumerate(combos):
            t0 = time.time()
            r  = self._run_single(df_h4, df_h1, df_m15, params)
            results.append(r)
            print(f"[{idx+1:>3}/{len(combos)}] Score:{r.score:>8.3f} | "
                  f"Trades:{r.total_trades:>5} | WR:{r.win_rate*100:>5.1f}% | "
                  f"DD:{r.max_drawdown_pct:>5.1f}% | {params} ({time.time()-t0:.1f}s)")
        results.sort(key=lambda r: r.score, reverse=True)
        print("\n"+"="*55+"\n  TOP 5\n"+"="*55)
        for r in results[:5]: ResultAnalyser.print_summary(r)
        self.logger.save_summary(results)
        run_id = f"tier1_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if results: self.logger.save_trades(results[0], run_id+"_best")
        return results

    def _run_single(self, df_h4, df_h1, df_m15, params):
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

        # Pre-compute indicators
        df_h4["ATR14"]      = Indicators.atr(df_h4["High"],df_h4["Low"],df_h4["Close"],14)
        df_h1["RSI14"]      = Indicators.rsi(df_h1["Close"],14)
        df_h1["EMA20"]      = Indicators.ema(df_h1["Close"],20)
        df_h1["EMA20slope"] = df_h1["EMA20"].diff(3)
        df_m15["RSI14"]     = Indicators.rsi(df_m15["Close"],14)
        df_m15["RSIslope"]  = Indicators.rsi_slope(df_m15["RSI14"],3)
        df_m15["ATR14"]     = Indicators.atr(df_m15["High"],df_m15["Low"],df_m15["Close"],14)
        df_m15["VolZ"]      = Indicators.volume_zscore(df_m15["Volume"],20)
        stk,std = Indicators.stoch_rsi(df_m15["Close"])
        df_m15["StochK"]=stk; df_m15["StochD"]=std

        for i in range(50, len(df_m15)):
            bar_time = df_m15.index[i]
            bar      = df_m15.iloc[i]

            # ── Daily reset ───────────────────────────────────
            bar_date = bar_time.date()
            if bar_date != daily_reset_date:
                self.simulator.daily_loss     = 0.0
                self.simulator.session_losses = 0
                daily_reset_date = bar_date

            # ── Weekly reset ──────────────────────────────────
            week_key = f"{bar_time.year}_{bar_time.isocalendar()[1]}"
            if week_key != weekly_reset_key:
                self.simulator.weekly_loss = 0.0
                weekly_reset_key = week_key

            hour = bar_time.hour

            # ── Session detection — v1.2 expanded ─────────────
            in_asian  = (ASIAN_OPEN_H  <= hour < ASIAN_CLOSE_H)
            in_london = (LONDON_OPEN_H <= hour < LONDON_CLOSE_H)
            in_ny     = (NY_OPEN_H     <= hour < NY_CLOSE_H)
            in_session= in_asian or in_london or in_ny

            # Session trade counter reset
            if in_asian:
                sess_key = f"asian_{bar_date}"
            elif in_london:
                sess_key = f"london_{bar_date}"
            elif in_ny:
                sess_key = f"ny_{bar_date}"
            else:
                sess_key = None

            if sess_key is not None and sess_key != self.simulator.current_session_key:
                self.simulator.current_session_key = sess_key
                self.simulator.session_trade_count = 0

            atr_val = bar["ATR14"]    if not pd.isna(bar["ATR14"])    else 10.0
            rsi_val = bar["RSI14"]    if not pd.isna(bar["RSI14"])    else 50.0
            rsi_slp = bar["RSIslope"] if not pd.isna(bar["RSIslope"]) else 0.0
            vol_z   = bar["VolZ"]     if not pd.isna(bar["VolZ"])     else 0.0

            # ── Update all open trades ─────────────────────────
            self.simulator.update_trades(bar, bar_time, atr_val, rsi_val, rsi_slp, vol_z)

            if not in_session: continue

            # ── Circuit breaker check ─────────────────────────
            allowed, _ = self.simulator.is_trading_allowed()
            if not allowed: continue

            # ── v1.2: allow up to MAX_CONCURRENT_TRADES ────────
            if len(self.simulator.open_trades) >= MAX_CONCURRENT_TRADES:
                continue

            # ── Spread filter — max 20 pts (replaces commission) ──
            spread_pts = bar.get("Spread", 25)
            if pd.isna(spread_pts): spread_pts = 25
            if spread_pts > MAX_SPREAD_PTS: continue

            # ── Contextual data ───────────────────────────────
            h4_bars = df_h4[df_h4.index <= bar_time].tail(20)
            h1_bars = df_h1[df_h1.index <= bar_time].tail(20)
            if len(h4_bars)<10 or len(h1_bars)<5: continue

            h1_last = h1_bars.iloc[-1]

            # ── H4 trend: v1.0 method — 5-bar close slope ─────
            # Reverted from v1.1 EMA20/EMA50 (too restrictive)
            h4_trend_up = h4_bars["Close"].iloc[-1] > h4_bars["Close"].iloc[-5]

            # ── H1 momentum: v1.2 LOOSENED ────────────────────
            # v1.0/v1.1: RSI>50 AND slope>0 (both required)
            # v1.2:       RSI>45 OR slope>0  (either is enough)
            h1_rsi   = h1_last.get("RSI14", 50.0)
            h1_slope = h1_last.get("EMA20slope", 0.0)
            h1_bull  = (h1_rsi > 45) or (h1_slope > 0)
            h1_bear  = (h1_rsi < 55) or (h1_slope < 0)

            # ── M15 StochRSI cross — v1.2 widened thresholds ──
            # v1.0/v1.1: bull<80, bear>20
            # v1.2:       bull<85, bear>15 (more entry opportunities)
            stk_k = bar.get("StochK", 50.0)
            stk_d = bar.get("StochD", 50.0)
            prev  = df_m15.iloc[i-1]
            prev_k = prev.get("StochK", 50.0)
            prev_d = prev.get("StochD", 50.0)
            bull_cross = (stk_k > stk_d and prev_k <= prev_d and stk_k < 85)
            bear_cross = (stk_k < stk_d and prev_k >= prev_d and stk_k > 15)

            long_signal  = h4_trend_up        and h1_bull and bull_cross
            short_signal = (not h4_trend_up)  and h1_bear and bear_cross
            if not (long_signal or short_signal): continue

            direction = "LONG" if long_signal else "SHORT"

            # ── Prevent same-direction stacking ───────────────
            open_dirs = [t.direction for t in self.simulator.open_trades]
            if open_dirs.count(direction) >= 2: continue

            # ── Volume filter ─────────────────────────────────
            if vol_z < FILTER_VOL_MIN_Z: continue

            # ── Scoring ───────────────────────────────────────
            conditions = {
                "elliott_bias":   h4_trend_up if direction=="LONG" else not h4_trend_up,
                "smc_structure":  h4_trend_up if direction=="LONG" else not h4_trend_up,
                "fvg_confluence": False,
                "h1_momentum":    h1_bull if direction=="LONG" else h1_bear,
                "m15_bos":        bull_cross if direction=="LONG" else bear_cross,
                "volume_confirm": vol_z > 0.5,
            }
            pw, raw_score, grade = self.scorer.calculate(conditions)
            if grade == "SKIP": continue

            # ── ATR-based SL/TP ───────────────────────────────
            atr_pts = atr_val / POINT
            sl_pts  = int(atr_pts * SL_ATR_MULTIPLIER)
            sl_pts  = max(int(atr_pts*SL_ATR_MIN_MULT),
                          min(int(atr_pts*SL_ATR_MAX_MULT), sl_pts))
            sl_pts  = max(sl_pts, 100)
            tp_pts  = int(sl_pts * TP_RR_RATIO)
            entry   = bar["Close"]
            sl = entry - sl_pts*POINT if direction=="LONG" else entry + sl_pts*POINT
            tp = entry + tp_pts*POINT if direction=="LONG" else entry - tp_pts*POINT

            lot = self.simulator.calc_lot_size(sl_pts, pw, self.scorer)
            if lot <= 0: continue

            # Monday lot reduction
            if bar_time.weekday() == 0:
                lot = max(LOT_MIN, round(
                    round(lot*FILTER_MONDAY_LOT/LOT_STEP)*LOT_STEP, 2))

            self.simulator.open_trade(
                direction, entry, sl, tp, lot,
                spread_pts, raw_score, grade, pw,
                bar_time, params)

        # Close remaining open trades at backtest end
        if self.simulator.open_trades:
            last_bar = df_m15.iloc[-1]
            for t in list(self.simulator.open_trades):
                self.simulator._close_trade(
                    t, last_bar["Close"], "BACKTEST_END", df_m15.index[-1])

        return ResultAnalyser.analyse(self.simulator, params)

# ---- ENTRY POINT -------------------------------------------
if __name__ == "__main__":
    print("="*55)
    print("  GoldScalperBot v1.2  —  XAUUSD...")
    print("="*55)
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}"); exit(1)
    a = mt5.account_info()
    if a is None:
        print("MT5 not logged in"); mt5.shutdown(); exit(1)
    print(f"MT5 connected — {a.login}  Balance: ${a.balance:,.2f}\n")
    runner  = BacktestRunner()
    results = runner.run_tier1(period="insample")
    if results:
        print("\nBest:"); ResultAnalyser.print_summary(results[0])
    mt5.shutdown(); print("Done.")
