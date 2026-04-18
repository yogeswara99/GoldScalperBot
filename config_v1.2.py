# ================================================================
#  config_v1.2.py  —  GoldScalperBot  XAUUSD...  (MultiBank)
#  VERSION 1.2  —  High-frequency spread-only cost model
#
#  Goal: 5-6 trades/day | DD <= 30% | WR >= 68%
#
#  Changes from v1.1:
#    - COMMISSION_USD      : removed entirely (spread-only model)
#    - MAX_SPREAD_PTS      : 20 pts hard entry filter (replaces commission)
#    - FILTER_VOL_MIN_Z    : 0.5 → 0.0  (reverted to v1.0)
#    - MAX_CONCURRENT_TRADES: 1 → 3     (primary frequency lever)
#    - MAX_TRADES_PER_SESS : 5 → 20     (effectively removed)
#    - MAX_DAILY_LOSS_PCT  : 3% → 4%    ($80)
#    - MAX_CONSEC_LOSSES   : 3 → 4
#    - WEEKLY_DD_PCT       : 8% → 12%   ($240)
#    - Sessions            : expanded + Asian session added
#    - H4 trend            : reverted to 5-bar slope (more signals)
#    - H1 condition        : loosened (OR instead of AND)
#
#  DO NOT MODIFY — locked reference for v1.2
# ================================================================

# ---- BROKER / SYMBOL ---------------------------------------
SYMBOL          = "XAUUSD..."         # THREE DOTS — MultiBank/MEX Global
BROKER_SERVER   = "MEXGlobalFinancial-Demo"
POINT           = 0.01
DIGITS          = 2
CONTRACT_SIZE   = 100
PIP_VALUE_STD   = 1.0
MAX_SPREAD_PTS  = 20                  # Hard entry filter — skip if spread > 20pts

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

# ---- COMMISSION — REMOVED (spread-only model) --------------
COMMISSION_USD  = 0.0                 # v1.2: spread captured via MAX_SPREAD_PTS filter

# ---- CONCURRENT TRADES (v1.2: key frequency lever) ---------
MAX_CONCURRENT_TRADES = 3            # Allow up to 3 open trades simultaneously

# ---- CIRCUIT BREAKERS --------------------------------------
MAX_DAILY_LOSS_PCT   = 0.04          # 4% = $80
MAX_DAILY_LOSS_USD   = ACCOUNT_BALANCE * MAX_DAILY_LOSS_PCT
MAX_CONSEC_LOSSES    = 4
MAX_TRADES_PER_SESS  = 20            # Effectively no limit
WEEKLY_DD_PCT        = 0.12          # 12% = $240
WEEKLY_DD_USD        = ACCOUNT_BALANCE * WEEKLY_DD_PCT

# ---- SESSIONS (UTC) — v1.2 expanded ------------------------
ASIAN_OPEN_H     = 1                 # NEW: Asian session
ASIAN_CLOSE_H    = 6
LONDON_OPEN_H    = 7                 # Extended from 8 → 7
LONDON_OPEN_MIN  = 0                 # No delay on London open
LONDON_CLOSE_H   = 12                # Extended from 11 → 12
NY_OPEN_H        = 12                # Overlap with London close
NY_CLOSE_H       = 17                # Extended from 16 → 17

# ---- DATA-SUPPORTED FILTERS --------------------------------
FILTER_VOL_MIN_Z     = 0.0           # v1.2: reverted to v1.0 (0.5 was too aggressive)
FILTER_LONDON_DELAY  = False         # v1.2: removed delay (07:00 open is clean enough)
FILTER_MONDAY_LOT    = 0.70          # Keep Monday protection

# ---- NEWS MODE ---------------------------------------------
NEWS_TP1_PTS    = 100
NEWS_TP2_PTS    = 180
NEWS_SL_PTS     = 50

# ---- LOGGING -----------------------------------------------
LOG_VERBOSE     = False
LOG_TRADES      = True
