# ================================================================
#  config_v1.1.py  —  GoldScalperBot  XAUUSD...  (MultiBank)
#  VERSION 1.1  —  Drawdown reduction config
#
#  Changes from v1.0:
#    - COMMISSION_USD   : added 3.50/lot/side (MEX Global realistic)
#    - FILTER_VOL_MIN_Z : 0.0 → 0.5  (stronger volume gate)
#
#  DO NOT MODIFY — locked reference for v1.1
# ================================================================

# ---- BROKER / SYMBOL ---------------------------------------
SYMBOL          = "XAUUSD..."         # THREE DOTS — MultiBank/MEX Global
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
RISK_NORMAL_PCT = 0.01                # 1% per trade = $20
RISK_NEWS_PCT   = 0.005
RISK_NORMAL_USD = ACCOUNT_BALANCE * RISK_NORMAL_PCT
RISK_NEWS_USD   = ACCOUNT_BALANCE * RISK_NEWS_PCT

# ---- ATR-BASED SL/TP ---------------------------------------
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

# ---- COMMISSION (v1.1: realistic model) --------------------
COMMISSION_USD  = 3.50               # Per lot per side (MEX Global typical)

# ---- CIRCUIT BREAKERS --------------------------------------
MAX_DAILY_LOSS_PCT   = 0.03
MAX_DAILY_LOSS_USD   = ACCOUNT_BALANCE * MAX_DAILY_LOSS_PCT
MAX_CONSEC_LOSSES    = 3
MAX_TRADES_PER_SESS  = 5
WEEKLY_DD_PCT        = 0.08
WEEKLY_DD_USD        = ACCOUNT_BALANCE * WEEKLY_DD_PCT

# ---- SESSIONS (UTC) ----------------------------------------
LONDON_OPEN_H    = 8
LONDON_OPEN_MIN  = 30
LONDON_CLOSE_H   = 11
NY_OPEN_H        = 13
NY_CLOSE_H       = 16

# ---- DATA-SUPPORTED FILTERS --------------------------------
FILTER_VOL_MIN_Z     = 0.5            # v1.1: raised from 0.0
FILTER_LONDON_DELAY  = True
FILTER_MONDAY_LOT    = 0.70

# ---- NEWS MODE ---------------------------------------------
NEWS_TP1_PTS    = 100
NEWS_TP2_PTS    = 180
NEWS_SL_PTS     = 50

# ---- LOGGING -----------------------------------------------
LOG_VERBOSE     = False
LOG_TRADES      = True
