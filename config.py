# ================================================================
#  config.py  --  GoldScalperBot  XAUUSD...  (MultiBank)
#  VERSION 1.3  --  Restored working baseline
#
#  Results: $13,352 | 73.8% WR | 1,821 trades | 26.1% DD
#  Score: 804.081  (best achieved so far)
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
MAX_CONCURRENT_TRADES  = 3
TOTAL_RISK_BUDGET_PCT  = 0.015       # 1.5% total budget
RISK_NORMAL_PCT        = TOTAL_RISK_BUDGET_PCT / MAX_CONCURRENT_TRADES  # 0.5%
RISK_NORMAL_USD        = ACCOUNT_BALANCE * RISK_NORMAL_PCT
RISK_NEWS_PCT          = 0.005
RISK_NEWS_USD          = ACCOUNT_BALANCE * RISK_NEWS_PCT

# ---- SL/TP -------------------------------------------------
SL_ATR_MULTIPLIER   = 1.5
SL_ATR_MIN_MULT     = 1.0
SL_ATR_MAX_MULT     = 2.5
TP_RR_RATIO         = 2.0
TP_TRAIL_ATR_MULT   = 1.5

# ---- LOT SIZING --------------------------------------------
LOT_MIN         = 0.01
LOT_MAX         = 10.0
LOT_STEP        = 0.01
MIN_RR          = 1.5

# ---- COMMISSION --------------------------------------------
COMMISSION_USD  = 0.0

# ---- CIRCUIT BREAKERS --------------------------------------
MAX_DAILY_LOSS_PCT   = 0.04
MAX_DAILY_LOSS_USD   = ACCOUNT_BALANCE * MAX_DAILY_LOSS_PCT
MAX_CONSEC_LOSSES    = 4
MAX_TRADES_PER_SESS  = 20
WEEKLY_DD_PCT        = 0.10
WEEKLY_DD_USD        = ACCOUNT_BALANCE * WEEKLY_DD_PCT

# ---- SESSIONS (UTC) ----------------------------------------
LONDON_OPEN_H    = 7
LONDON_OPEN_MIN  = 0
LONDON_CLOSE_H   = 12
NY_OPEN_H        = 12
NY_CLOSE_H       = 18

# ---- FILTERS -----------------------------------------------
FILTER_VOL_MIN_Z     = 0.0
FILTER_LONDON_DELAY  = False
FILTER_MONDAY_LOT    = 0.70
FILTER_THURSDAY_LOT  = 1.0

# ---- LOGGING -----------------------------------------------
LOG_VERBOSE     = False
LOG_TRADES      = True
