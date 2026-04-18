# ================================================================
#  config_v1.5.py  --  GoldScalperBot  XAUUSD...
#  VERSION 1.5  --  ML-Optimized (Random Forest + Optuna)
#
#  Generated : 2026-04-18
#  Source    : ml_optimizer_lite.py results
#
#  KEY CHANGES FROM v1.3:
#    Signal completely rebuilt from ML feature importance:
#      - H4 5-bar slope REMOVED (most harmful feature: -0.023)
#      - H1 EMA 3-bar slope is NEW primary signal (#1: 17.03%)
#      - Bollinger band WIDTH added (#8: 5.26%)
#      - Volume vs 20-bar mean added (#4: 8.67%)
#      - NY session REMOVED (harmful: -0.0003)
#      - London session ONLY (essential: 4.95%)
#      - RSI raw/5bar removed (harmful)
#      - BE trigger: 3.5x -> 7.4x (Optuna: #1 param importance 0.30)
#      - SL wider: ATR x2.39 (Optuna)
#      - TP better: 2.76 R:R (Optuna)
#      - Single trade only (Optuna: max_concurrent=1)
#
#  Optuna result: 475 trades | 80.6% WR | $1,852 | 2.8% DD | PF 2.39
# ================================================================

# ---- BROKER / SYMBOL ---------------------------------------
SYMBOL          = "XAUUSD..."
BROKER_SERVER   = "MEXGlobalFinancial-Demo"
POINT           = 0.01
DIGITS          = 2
CONTRACT_SIZE   = 100
PIP_VALUE_STD   = 1.0
MAX_SPREAD_PTS  = 20

# ---- ACCOUNT -----------------------------------------------
ACCOUNT_BALANCE = 2000.0
ACCOUNT_LEVERAGE= 500

# ---- RISK PER TRADE ----------------------------------------
# v1.5: single trade, 0.51% per trade (Optuna optimal)
RISK_NORMAL_PCT  = 0.0051
RISK_NORMAL_USD  = ACCOUNT_BALANCE * RISK_NORMAL_PCT
RISK_NEWS_PCT    = 0.005
RISK_NEWS_USD    = ACCOUNT_BALANCE * RISK_NEWS_PCT

# ---- CONCURRENT TRADES ------------------------------------
# v1.5: Optuna found max_concurrent=1 is optimal
# single trade = no correlation risk, cleaner signals
MAX_CONCURRENT_TRADES  = 1
TOTAL_RISK_BUDGET_PCT  = RISK_NORMAL_PCT
COMMISSION_USD  = 0.0

# ---- SL/TP (Optuna-optimised) ------------------------------
# Wider SL to avoid premature stops (ATR x2.39 vs x1.5 in v1.0)
# Better R:R: 2.76 vs 2.0 in v1.0
SL_ATR_MULTIPLIER   = 2.3857
SL_ATR_MIN_MULT     = 0.9861
SL_ATR_MAX_MULT     = 2.7255
TP_RR_RATIO         = 2.7624
TP_TRAIL_ATR_MULT   = 1.5

# ---- LOT SIZING --------------------------------------------
LOT_MIN         = 0.01
LOT_MAX         = 10.0
LOT_STEP        = 0.01
MIN_RR          = 1.5

# ---- BREAKEVEN (Optuna: be_trigger is #1 most important) ---
# Optuna found 7.4x vs our 3.5x -- wait longer before trailing
# This dramatically reduces premature breakeven exits
BE_PROFIT_TRIGGER_MULT = 7.4156
BE_SL_LOCK_MULT        = 1.4086

# ---- CIRCUIT BREAKERS (Optuna-optimised) -------------------
MAX_DAILY_LOSS_PCT   = 0.0479
MAX_DAILY_LOSS_USD   = ACCOUNT_BALANCE * MAX_DAILY_LOSS_PCT
MAX_CONSEC_LOSSES    = 7               # Optuna: #2 most important param
MAX_TRADES_PER_SESS  = 20
WEEKLY_DD_PCT        = 0.1418
WEEKLY_DD_USD        = ACCOUNT_BALANCE * WEEKLY_DD_PCT

# ---- SESSIONS (ML: London ONLY, NY harmful) ----------------
# is_london: rank 9 ESSENTIAL (+4.95%)
# is_ny:     HARMFUL (-0.0003)  <-- completely removed
LONDON_OPEN_H    = 9               # Optuna: lon_open=9
LONDON_OPEN_MIN  = 0
LONDON_CLOSE_H   = 14              # Optuna: lon_close=14
# NY session removed entirely
NY_OPEN_H        = 99              # disabled
NY_CLOSE_H       = 99              # disabled

# ---- ML-DERIVED SIGNAL FILTERS -----------------------------
# Vol vs 20-bar mean (rank 4: 8.67%)
FILTER_VOL_VS_MEAN_MIN  = -0.7916  # Optuna vol_min_z (loose)

# H1 EMA slope thresholds (rank 1: 17.03%)
# For LONG: H1 EMA slope must be positive and accelerating
# For SHORT: H1 EMA slope must be negative and accelerating
H1_EMA_SLOPE3_MIN_LONG  =  0.0    # EMA rising for longs
H1_EMA_SLOPE3_MAX_SHORT =  0.0    # EMA falling for shorts

# Bollinger band width minimum (rank 8: 5.26%)
# Low BB width = squeeze = pre-breakout = good entry
# High BB width = already expanded = late entry
BB_WIDTH_MIN    = 0.01             # ignore extreme squeezes
BB_WIDTH_MAX    = 0.15             # avoid already-expanded bands

# ATR points minimum (rank 6: 7.74%)
# Need minimum volatility for meaningful moves
ATR_PTS_MIN     = 150.0           # minimum 150 pts ATR

# RSI 3-bar slope (rank 15: 2.17%) -- only useful RSI metric
RSI_3BAR_SLOPE_MIN_LONG  = -5.0   # not aggressively overbought
RSI_3BAR_SLOPE_MAX_SHORT =  5.0   # not aggressively oversold

# StochRSI oversold filter (rank 14: 2.48%)
STOCH_BULL_CAP          = 81.8    # Optuna
STOCH_BEAR_FLOOR        = 15.6    # Optuna

# H1 RSI thresholds (loosened per Optuna)
H1_RSI_BULL_MIN         = 42.3
H1_RSI_BEAR_MAX         = 60.4

# Removed features (actively harmful per ML):
# - H4 5-bar slope (most harmful: -0.023)
# - RSI raw level (harmful: -0.0020)
# - RSI 5-bar slope (harmful: -0.0107)
# - StochRSI K level (harmful: -0.0084)
# - H4 vs EMA20 (harmful: -0.0104)
# - H1 price vs EMA (harmful: -0.0131)
# - Body ratio (harmful: -0.0050)
# - ADX (harmful: -0.0030)
# - NY session (harmful: -0.0003)

# ---- DAY OF WEEK (Optuna found near-full lots optimal) -----
FILTER_MONDAY_LOT       = 0.9394  # Optuna: nearly full
FILTER_THURSDAY_LOT     = 0.9667  # Optuna: nearly full
FILTER_LONDON_DELAY     = False

# ---- LOGGING -----------------------------------------------
LOG_VERBOSE     = False
LOG_TRADES      = True
