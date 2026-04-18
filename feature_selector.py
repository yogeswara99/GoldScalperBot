#!/usr/bin/env python3
# ================================================================
#  feature_selector.py  --  GoldScalperBot  XAUUSD...
#
#  PURPOSE:
#    Post-process ml_optimizer.py results to identify:
#      1. Which of 54 features actually matter (SHAP analysis)
#      2. Which features are redundant (correlation clustering)
#      3. Which of 23 Optuna parameters actually moved the score
#      4. Retrain lean XGBoost on essential features only
#      5. Generate final minimal signal filter set for v1.5
#
#  MUST RUN AFTER: python ml_optimizer.py
#  REQUIRES: analysis_results/ml_feature_importance.csv
#             analysis_results/ml_full_features.csv
#             analysis_results/optuna_trials.csv
#
#  USAGE: python feature_selector.py
# ================================================================

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "analysis_results"

# ---- Check dependencies ----------------------------------------
def check_deps():
    missing = []
    for pkg in ["xgboost","sklearn","shap","optuna","matplotlib"]:
        try:
            __import__("sklearn" if pkg == "sklearn" else pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: pip install {' '.join(missing)}"); exit(1)

check_deps()

import xgboost as xgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ================================================================
#  SECTION 1 — FEATURE IMPORTANCE ANALYSIS
# ================================================================

def analyse_feature_importance():
    print("=" * 70)
    print("  SECTION 1: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    path = os.path.join(OUTPUT_DIR, "ml_feature_importance.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run ml_optimizer.py first.")
        return None, None

    fi = pd.read_csv(path)
    fi = fi.sort_values("shap_importance", ascending=False).reset_index(drop=True)

    total_shap = fi["shap_importance"].sum()
    fi["shap_pct"]      = fi["shap_importance"] / total_shap * 100
    fi["shap_cumsum"]   = fi["shap_pct"].cumsum()
    fi["rank"]          = fi.index + 1

    print(f"\n  Total features: {len(fi)}")
    print(f"\n  Full Feature Ranking (SHAP Mean |Value|):")
    print(f"  {'Rank':<5} {'Feature':<30} {'SHAP%':>8} {'Cumul%':>8} {'Category':<15} {'Signal?'}")
    print("  " + "-" * 80)

    # Categorise each feature
    categories = {
        "rsi":         ["rsi","rsi_vs_50","rsi_3bar","rsi_5bar","rsi_10bar","rsi_over","rsi_under","rsi_oversold","rsi_overbought"],
        "ema":         ["h1_ema","ema"],
        "volume":      ["vol_"],
        "atr":         ["atr_"],
        "price_action":["body_ratio","close_pos","is_bull","gap_from","consec","price_mom"],
        "stoch":       ["stoch_"],
        "macd":        ["macd_"],
        "bollinger":   ["bb_"],
        "adx":         ["adx","di_plus","di_minus","di_spread"],
        "h4":          ["h4_"],
        "session":     ["hour","day_of","is_london","is_ny","is_overlap","session_age"],
        "spread":      ["spread_"],
    }

    def get_cat(name):
        for cat, patterns in categories.items():
            if any(p in name for p in patterns):
                return cat
        return "other"

    # Significance tiers
    # Tier 1: top 80% cumulative SHAP = essential
    # Tier 2: 80-95% cumulative SHAP = useful
    # Tier 3: >95% = marginal / can drop

    for _, row in fi.iterrows():
        cat  = get_cat(row["feature"])
        tier = ("ESSENTIAL" if row["shap_cumsum"] <= 80
                else "USEFUL"    if row["shap_cumsum"] <= 95
                else "MARGINAL")
        signal = "<<< DROP" if tier == "MARGINAL" else ("*** KEEP" if tier == "ESSENTIAL" else "    keep")
        print(f"  {row['rank']:<5} {row['feature']:<30} {row['shap_pct']:>7.2f}% "
              f"{row['shap_cumsum']:>7.1f}% {cat:<15} {signal}")

    # Summary
    essential = fi[fi["shap_cumsum"] <= 80]
    useful    = fi[(fi["shap_cumsum"] > 80) & (fi["shap_cumsum"] <= 95)]
    marginal  = fi[fi["shap_cumsum"] > 95]

    print(f"\n  TIER SUMMARY:")
    print(f"  ESSENTIAL (top 80% SHAP): {len(essential)} features")
    print(f"  USEFUL    (80-95% SHAP) : {len(useful)} features")
    print(f"  MARGINAL  (>95% SHAP)   : {len(marginal)} features  <-- candidates for removal")

    # Category breakdown
    print(f"\n  CATEGORY BREAKDOWN (ESSENTIAL features only):")
    cat_df = essential.copy()
    cat_df["category"] = cat_df["feature"].apply(get_cat)
    cat_summary = cat_df.groupby("category")["shap_pct"].sum().sort_values(ascending=False)
    for cat, pct in cat_summary.items():
        bar = "=" * int(pct / 2)
        print(f"    {cat:<15} {pct:>6.1f}%  {bar}")

    return fi, essential


# ================================================================
#  SECTION 2 — CORRELATION ANALYSIS (find redundant features)
# ================================================================

def analyse_correlations(fi_df):
    print("\n" + "=" * 70)
    print("  SECTION 2: CORRELATION ANALYSIS (Redundant Feature Detection)")
    print("=" * 70)

    feat_path = os.path.join(OUTPUT_DIR, "ml_full_features.csv")
    if not os.path.exists(feat_path):
        print(f"  ERROR: {feat_path} not found."); return None

    df = pd.read_csv(feat_path)
    available = [c for c in fi_df["feature"].tolist() if c in df.columns]
    df_feat   = df[available].fillna(0)

    corr_matrix = df_feat.corr().abs()

    # Find pairs with correlation > threshold
    CORR_THRESHOLD = 0.80
    redundant_pairs = []
    seen = set()

    for i in range(len(available)):
        for j in range(i+1, len(available)):
            c = corr_matrix.iloc[i, j]
            if c >= CORR_THRESHOLD:
                f1, f2 = available[i], available[j]
                pair_key = tuple(sorted([f1, f2]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    # Keep the one with higher SHAP importance
                    shap1 = fi_df[fi_df["feature"]==f1]["shap_importance"].values
                    shap2 = fi_df[fi_df["feature"]==f2]["shap_importance"].values
                    s1 = shap1[0] if len(shap1) > 0 else 0
                    s2 = shap2[0] if len(shap2) > 0 else 0
                    keep   = f1 if s1 >= s2 else f2
                    drop   = f2 if s1 >= s2 else f1
                    redundant_pairs.append({
                        "feature_a": f1, "feature_b": f2,
                        "correlation": round(c, 3),
                        "keep": keep, "drop_candidate": drop,
                        "shap_keep": max(s1, s2),
                        "shap_drop": min(s1, s2),
                    })

    if redundant_pairs:
        rp_df = pd.DataFrame(redundant_pairs).sort_values("correlation", ascending=False)
        print(f"\n  Found {len(rp_df)} highly correlated pairs (threshold: {CORR_THRESHOLD}):")
        print(f"\n  {'Feature A':<28} {'Feature B':<28} {'Corr':>6} {'Keep':<28} {'Drop'}")
        print("  " + "-" * 100)
        for _, r in rp_df.iterrows():
            print(f"  {r['feature_a']:<28} {r['feature_b']:<28} "
                  f"{r['correlation']:>6.3f} {r['keep']:<28} {r['drop_candidate']}")

        drop_candidates = set(rp_df["drop_candidate"].tolist())
        print(f"\n  Drop candidates from correlation (keep the stronger of each pair):")
        for f in sorted(drop_candidates):
            shap_val = fi_df[fi_df["feature"]==f]["shap_importance"].values
            sv = shap_val[0] if len(shap_val) > 0 else 0
            print(f"    - {f:<30} SHAP: {sv:.4f}")
    else:
        print(f"  No highly correlated pairs found (threshold: {CORR_THRESHOLD})")
        drop_candidates = set()

    # Save correlation matrix
    corr_path = os.path.join(OUTPUT_DIR, "feature_correlation.csv")
    corr_matrix.to_csv(corr_path)
    print(f"\n  Correlation matrix saved: {corr_path}")

    return drop_candidates


# ================================================================
#  SECTION 3 — OPTUNA PARAMETER IMPORTANCE
# ================================================================

def analyse_optuna_params():
    print("\n" + "=" * 70)
    print("  SECTION 3: OPTUNA PARAMETER IMPORTANCE")
    print("  (Which trading parameters actually moved the score?)")
    print("=" * 70)

    path = os.path.join(OUTPUT_DIR, "optuna_trials.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found."); return None

    trials_df = pd.read_csv(path)
    param_cols = [c for c in trials_df.columns if c.startswith("params_")]

    if not param_cols:
        print("  No parameter columns found in trials CSV.")
        return None

    # Compute correlation between each param and the trial score
    score_col = "value"
    if score_col not in trials_df.columns:
        print("  'value' column not found in trials CSV.")
        return None

    valid = trials_df[trials_df[score_col] > 0].copy()
    print(f"\n  Valid trials (score > 0): {len(valid)} / {len(trials_df)}")

    param_importance = []
    for col in param_cols:
        param_name = col.replace("params_", "")
        if valid[col].nunique() < 2: continue
        corr = abs(valid[col].corr(valid[score_col]))
        # Variance in top 20% vs bottom 20%
        top20    = valid.nlargest(max(1, len(valid)//5), score_col)[col]
        bottom20 = valid.nsmallest(max(1, len(valid)//5), score_col)[col]
        top_mean    = top20.mean()
        bottom_mean = bottom20.mean()
        top_std     = top20.std()

        param_importance.append({
            "parameter":    param_name,
            "corr_w_score": corr,
            "top20_mean":   top_mean,
            "bot20_mean":   bottom_mean,
            "top20_std":    top_std,
            "delta":        abs(top_mean - bottom_mean),
        })

    pi_df = pd.DataFrame(param_importance).sort_values("corr_w_score", ascending=False)

    print(f"\n  Parameter Importance (correlation with trial score):")
    print(f"\n  {'Parameter':<25} {'|Corr|':>8} {'Top20 Mean':>12} {'Bot20 Mean':>12} {'Delta':>10} {'Verdict'}")
    print("  " + "-" * 85)

    for _, r in pi_df.iterrows():
        verdict = ("HIGH IMPACT" if r["corr_w_score"] > 0.20
                   else "MODERATE"   if r["corr_w_score"] > 0.08
                   else "LOW IMPACT")
        print(f"  {r['parameter']:<25} {r['corr_w_score']:>8.4f} "
              f"{r['top20_mean']:>12.4f} {r['bot20_mean']:>12.4f} "
              f"{r['delta']:>10.4f}  {verdict}")

    # Classify parameters
    high_impact = pi_df[pi_df["corr_w_score"] > 0.20]["parameter"].tolist()
    moderate    = pi_df[(pi_df["corr_w_score"] > 0.08) & (pi_df["corr_w_score"] <= 0.20)]["parameter"].tolist()
    low_impact  = pi_df[pi_df["corr_w_score"] <= 0.08]["parameter"].tolist()

    print(f"\n  HIGH IMPACT parameters (|corr| > 0.20) — MUST OPTIMISE:")
    for p in high_impact:
        row = pi_df[pi_df["parameter"]==p].iloc[0]
        print(f"    {p:<25}  optimal range: "
              f"[{row['top20_mean']-row['top20_std']:.3f} — {row['top20_mean']+row['top20_std']:.3f}]")

    print(f"\n  MODERATE parameters (0.08-0.20) — worth tuning:")
    for p in moderate:
        print(f"    {p}")

    print(f"\n  LOW IMPACT parameters (< 0.08) — can use default/fixed values:")
    for p in low_impact:
        print(f"    {p}")

    # Save
    pi_path = os.path.join(OUTPUT_DIR, "parameter_importance.csv")
    pi_df.to_csv(pi_path, index=False)
    print(f"\n  Parameter importance saved: {pi_path}")

    return pi_df, high_impact, low_impact


# ================================================================
#  SECTION 4 — LEAN MODEL (essential features only)
# ================================================================

def train_lean_model(fi_df, drop_corr: set):
    print("\n" + "=" * 70)
    print("  SECTION 4: LEAN MODEL (Essential Features Only)")
    print("=" * 70)

    feat_path = os.path.join(OUTPUT_DIR, "ml_full_features.csv")
    if not os.path.exists(feat_path):
        print(f"  ERROR: {feat_path} not found."); return None

    df = pd.read_csv(feat_path)
    df_clean = df[df["outcome"].isin(["TP","SL"])].copy()
    df_clean["target"] = (df_clean["win"] == 1).astype(int)

    all_feats = [c for c in fi_df["feature"].tolist() if c in df_clean.columns]

    # FULL model features
    full_feats = all_feats

    # ESSENTIAL: top 80% SHAP, minus correlated drops
    fi_sorted = fi_df.sort_values("shap_importance", ascending=False).reset_index(drop=True)
    fi_sorted["shap_cumsum"] = (fi_sorted["shap_importance"] / fi_sorted["shap_importance"].sum() * 100).cumsum()
    essential_feats = [f for f in fi_sorted[fi_sorted["shap_cumsum"] <= 80]["feature"].tolist()
                       if f in df_clean.columns and f not in drop_corr]

    # LEAN: top 10 features by SHAP (minimum viable)
    top10_feats = [f for f in fi_sorted.head(10)["feature"].tolist()
                   if f in df_clean.columns and f not in drop_corr]

    print(f"\n  Full model    : {len(full_feats)} features")
    print(f"  Essential set : {len(essential_feats)} features")
    print(f"  Top-10 set    : {len(top10_feats)} features")
    print(f"\n  Essential features (used in lean model):")
    for f in essential_feats:
        row = fi_sorted[fi_sorted["feature"]==f]
        pct = row["shap_importance"].values[0] / fi_sorted["shap_importance"].sum() * 100 if len(row)>0 else 0
        print(f"    {f:<30} SHAP: {pct:.2f}%")

    # Time-based train/test split (no leakage)
    split = int(len(df_clean) * 0.8)

    results = {}
    for name, feats in [("Full", full_feats), ("Essential", essential_feats), ("Top-10", top10_feats)]:
        if not feats: continue
        X = df_clean[feats].fillna(0).values
        y = df_clean["target"].values
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale, eval_metric="logloss",
            verbosity=0, random_state=42,
        )
        m.fit(X_tr, y_tr, verbose=False)
        preds = m.predict(X_te)
        proba = m.predict_proba(X_te)[:,1]
        acc   = accuracy_score(y_te, preds)
        auc   = roc_auc_score(y_te, proba)

        results[name] = {"acc": acc, "auc": auc, "n_feats": len(feats), "model": m}
        print(f"\n  {name:>10} model ({len(feats):>2} features): "
              f"ACC={acc*100:.1f}%  AUC={auc:.3f}")

    # Compare
    if "Full" in results and "Essential" in results:
        auc_diff = results["Essential"]["auc"] - results["Full"]["auc"]
        feat_pct = results["Essential"]["n_feats"] / results["Full"]["n_feats"] * 100
        print(f"\n  Essential vs Full:")
        print(f"    Features reduced : {results['Full']['n_feats']} -> {results['Essential']['n_feats']} ({feat_pct:.0f}%)")
        print(f"    AUC difference   : {auc_diff:+.4f} ({'NEGLIGIBLE' if abs(auc_diff) < 0.01 else 'SIGNIFICANT'})")
        if abs(auc_diff) < 0.01:
            print(f"    VERDICT: Use Essential set. {100-feat_pct:.0f}% fewer features with no meaningful AUC loss.")
        else:
            print(f"    VERDICT: Essential set loses {abs(auc_diff):.3f} AUC — consider keeping more features.")

    return essential_feats, results


# ================================================================
#  SECTION 5 — GENERATE FINAL MINIMAL SIGNAL FILTER SET
# ================================================================

def generate_minimal_filters(essential_feats: list, pi_df, high_impact: list, best_params_path: str):
    print("\n" + "=" * 70)
    print("  SECTION 5: FINAL MINIMAL SIGNAL FILTER RECOMMENDATIONS")
    print("=" * 70)

    # Load best params from config_v1.5.py if it exists
    if not os.path.exists("config_v1.5.py"):
        print("  config_v1.5.py not found -- run ml_optimizer.py first")
        return

    print("\n  Based on SHAP + Optuna analysis, here is the minimal")
    print("  signal filter set for v1.5:\n")

    # Map essential features to signal filters
    filter_map = {
        "rsi_3bar_slope":    "RSI 3-bar slope gate (strongest discriminator)",
        "rsi_5bar_slope":    "RSI 5-bar slope gate",
        "rsi_10bar_slope":   "RSI 10-bar slope gate",
        "h1_ema_slope3":     "H1 EMA 3-bar slope filter",
        "h1_ema_slope5":     "H1 EMA 5-bar slope cap",
        "atr_ratio":         "ATR ratio filter (volatility regime)",
        "vol_z":             "Volume Z-score minimum",
        "vol_ratio":         "Volume/mean ratio filter",
        "adx":               "ADX trend strength filter",
        "bb_pct_b":          "Bollinger %B position filter",
        "macd_hist":         "MACD histogram filter",
        "stoch_kd_spread":   "StochRSI K-D spread filter",
        "price_mom_5bar":    "5-bar price momentum filter",
        "atr_regime":        "ATR regime (spike avoidance)",
        "hour":              "Hour-of-day session filter",
        "day_of_week":       "Day-of-week filter",
        "spread_atr_ratio":  "Spread/ATR ratio filter",
    }

    print("  Features with actionable signal filters:")
    actionable = []
    for f in essential_feats:
        if f in filter_map:
            actionable.append(f)
            print(f"    {f:<28} --> {filter_map[f]}")

    print(f"\n  Features present but NOT directly filterable (model inputs only):")
    non_filterable = [f for f in essential_feats if f not in filter_map]
    for f in non_filterable:
        print(f"    {f}")

    print(f"\n  HIGH IMPACT Optuna parameters to focus v1.6 tuning on:")
    if pi_df is not None and high_impact:
        for p in high_impact:
            row = pi_df[pi_df["parameter"]==p]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"    {p:<25}  |corr|={r['corr_w_score']:.4f}  "
                      f"optimal~{r['top20_mean']:.3f}")

    # Save summary
    summary = {
        "essential_features":   essential_feats,
        "actionable_filters":   actionable,
        "high_impact_params":   high_impact if high_impact else [],
        "n_essential":          len(essential_feats),
        "n_actionable_filters": len(actionable),
    }

    import json
    spath = os.path.join(OUTPUT_DIR, "minimal_filter_set.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Minimal filter set saved: {spath}")

    return actionable


# ================================================================
#  SECTION 6 — CHARTS
# ================================================================

def generate_charts(fi_df, pi_df):
    print("\n  Generating charts...")

    fig = plt.figure(figsize=(18, 14), facecolor="#0f1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ---- Chart 1: Top 20 features by SHAP ----------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#1a1a2e")
    top20 = fi_df.head(20)
    colors = ["#16a34a" if fi_df["shap_cumsum"].iloc[i] <= 80
              else "#f59e0b" if fi_df["shap_cumsum"].iloc[i] <= 95
              else "#dc2626"
              for i in range(min(20, len(fi_df)))]
    bars = ax1.barh(range(len(top20)), top20["shap_importance"].values[::-1], color=colors[::-1])
    ax1.set_yticks(range(len(top20)))
    ax1.set_yticklabels(top20["feature"].tolist()[::-1], fontsize=9, color="white")
    ax1.set_xlabel("SHAP Mean |Value|", color="#94a3b8")
    ax1.set_title("Top 20 Features by SHAP Importance\n"
                  "(Green=Essential | Orange=Useful | Red=Marginal)",
                  color="white", fontsize=11)
    ax1.tick_params(colors="#94a3b8")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    for sp in ["bottom","left"]: ax1.spines[sp].set_color("#334155")

    # ---- Chart 2: Cumulative SHAP explained ---------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#1a1a2e")
    n_feat = len(fi_df)
    ax2.plot(range(1, n_feat+1), fi_df["shap_cumsum"].values,
             color="#3b82f6", linewidth=2)
    ax2.axhline(80, color="#16a34a", linestyle="--", linewidth=1.5, label="80% (Essential)")
    ax2.axhline(95, color="#f59e0b", linestyle="--", linewidth=1.5, label="95% (Useful)")
    ax2.fill_between(range(1, n_feat+1), fi_df["shap_cumsum"].values,
                     alpha=0.15, color="#3b82f6")
    ax2.set_xlabel("Number of Features", color="#94a3b8")
    ax2.set_ylabel("Cumulative SHAP %", color="#94a3b8")
    ax2.set_title("Cumulative Feature Importance\n(How many features explain what % of model)",
                  color="white", fontsize=10)
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")
    ax2.tick_params(colors="#94a3b8")
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax2.spines[sp].set_color("#334155")

    # ---- Chart 3: Parameter importance (Optuna) -----------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1a1a2e")
    if pi_df is not None and len(pi_df) > 0:
        pi_top = pi_df.head(15)
        colors3 = ["#16a34a" if v > 0.20 else "#f59e0b" if v > 0.08 else "#64748b"
                   for v in pi_top["corr_w_score"]]
        ax3.barh(range(len(pi_top)),
                 pi_top["corr_w_score"].values[::-1],
                 color=colors3[::-1])
        ax3.set_yticks(range(len(pi_top)))
        ax3.set_yticklabels(pi_top["parameter"].tolist()[::-1], fontsize=8, color="white")
        ax3.axvline(0.20, color="#16a34a", linestyle="--", linewidth=1, label="High impact")
        ax3.axvline(0.08, color="#f59e0b", linestyle="--", linewidth=1, label="Moderate")
        ax3.set_xlabel("|Corr with score|", color="#94a3b8")
        ax3.set_title("Optuna Parameter Importance\n(Green=High | Orange=Moderate | Grey=Low)",
                      color="white", fontsize=10)
        ax3.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e")
        ax3.tick_params(colors="#94a3b8")
        for sp in ["top","right"]: ax3.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax3.spines[sp].set_color("#334155")

    chart_path = os.path.join(OUTPUT_DIR, "feature_selection_charts.png")
    plt.savefig(chart_path, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Charts saved: {chart_path}")


# ================================================================
#  SECTION 7 — PRINT FINAL DECISION TABLE
# ================================================================

def print_decision_table(fi_df, drop_corr, pi_df, high_impact, low_impact):
    print("\n" + "=" * 70)
    print("  FINAL DECISION TABLE")
    print("  What to keep, what to simplify, what to drop entirely")
    print("=" * 70)

    print("""
  SIGNAL FILTERS (entry conditions):
  ---------------------------------------------------
  Keep?  Filter                    Reason
  -----  -------                   ------""")

    decisions = []

    # RSI slope
    rsi3_shap = fi_df[fi_df["feature"]=="rsi_3bar_slope"]["shap_importance"].values
    if len(rsi3_shap) > 0 and rsi3_shap[0] > 0.001:
        decisions.append(("YES", "RSI 3-bar slope gate",
                           f"SHAP={rsi3_shap[0]:.4f} -- strongest signal discriminator"))
    else:
        decisions.append(("NO ", "RSI 3-bar slope gate", "Low SHAP -- remove from filters"))

    rsi5_shap = fi_df[fi_df["feature"]=="rsi_5bar_slope"]["shap_importance"].values
    if len(rsi5_shap) > 0 and rsi5_shap[0] > 0.001:
        decisions.append(("YES", "RSI 5-bar slope gate", f"SHAP={rsi5_shap[0]:.4f}"))
    else:
        decisions.append(("NO ", "RSI 5-bar slope gate", "Low SHAP"))

    ema5_shap = fi_df[fi_df["feature"]=="h1_ema_slope5"]["shap_importance"].values
    if len(ema5_shap) > 0 and ema5_shap[0] > 0.001:
        decisions.append(("YES", "H1 EMA5 slope cap",
                           f"SHAP={ema5_shap[0]:.4f} -- overextension filter"))
    else:
        decisions.append(("NO ", "H1 EMA5 slope cap", "Low SHAP -- remove"))

    atr_shap = fi_df[fi_df["feature"]=="atr_ratio"]["shap_importance"].values
    if len(atr_shap) > 0 and atr_shap[0] > 0.001:
        decisions.append(("YES", "ATR ratio (vol regime)",
                           f"SHAP={atr_shap[0]:.4f} -- volatility context matters"))
    else:
        decisions.append(("NO ", "ATR ratio", "Low SHAP"))

    vol_shap = fi_df[fi_df["feature"]=="vol_z"]["shap_importance"].values
    if len(vol_shap) > 0 and vol_shap[0] > 0.0005:
        decisions.append(("MAY", "Volume Z-score min",
                           f"SHAP={vol_shap[0]:.4f} -- weak but present"))
    else:
        decisions.append(("NO ", "Volume Z-score min", "Very low SHAP -- remove"))

    adx_shap = fi_df[fi_df["feature"]=="adx"]["shap_importance"].values
    if len(adx_shap) > 0 and adx_shap[0] > 0.001:
        decisions.append(("YES", "ADX trend strength",
                           f"SHAP={adx_shap[0]:.4f} -- new in v1.5"))
    else:
        decisions.append(("NO ", "ADX filter", "Low SHAP -- skip for now"))

    bb_shap = fi_df[fi_df["feature"]=="bb_pct_b"]["shap_importance"].values
    if len(bb_shap) > 0 and bb_shap[0] > 0.001:
        decisions.append(("YES", "Bollinger %B position",
                           f"SHAP={bb_shap[0]:.4f} -- new in v1.5"))

    for keep, name, reason in decisions:
        status = ("✓ KEEP   " if keep == "YES"
                  else "✗ DROP   " if keep == "NO " else "? MAYBE  ")
        print(f"  {status} {name:<28} {reason}")

    print(f"""
  OPTUNA PARAMETERS:
  ---------------------------------------------------""")
    if high_impact:
        print(f"  OPTIMISE THESE (high impact):")
        for p in high_impact: print(f"    - {p}")
    if low_impact:
        print(f"\n  FIX THESE TO DEFAULTS (low impact, no need to search):")
        for p in low_impact[:8]: print(f"    - {p}")

    print(f"""
  NEXT STEPS:
  ---------------------------------------------------
  1. Run: python ml_optimizer.py  (if not done already)
  2. Review config_v1.5.py        (auto-generated by Optuna)
  3. Copy:  copy config_v1.5.py config.py
  4. Run:  python quick_diagnostic.py   (validate result)
  5. If AUC loss < 0.01 between Essential and Full model:
     --> Use essential features only in v1.6 (faster, less overfit)
  6. Build v1.6 signal engine using only actionable essential features
""")


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  GoldScalperBot -- Feature Selector & Parameter Analyser")
    print("=" * 70)
    print(f"  Reading from: {OUTPUT_DIR}/")
    print("=" * 70)

    # Check required files exist
    required = [
        "ml_feature_importance.csv",
        "ml_full_features.csv",
        "optuna_trials.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    if missing:
        print(f"\n  MISSING FILES (run ml_optimizer.py first):")
        for f in missing: print(f"    - {OUTPUT_DIR}/{f}")
        exit(1)

    # Section 1: Feature importance
    fi_df, essential_fi = analyse_feature_importance()
    if fi_df is None: exit(1)

    # Section 2: Correlation analysis
    drop_corr = analyse_correlations(fi_df)
    if drop_corr is None: drop_corr = set()

    # Section 3: Optuna parameter importance
    optuna_result = analyse_optuna_params()
    if optuna_result is not None:
        pi_df, high_impact, low_impact = optuna_result
    else:
        pi_df = None; high_impact = []; low_impact = []

    # Section 4: Lean model
    lean_result = train_lean_model(fi_df, drop_corr)
    if lean_result is not None:
        essential_feats, model_results = lean_result
    else:
        essential_feats = []

    # Section 5: Minimal filter recommendations
    actionable = generate_minimal_filters(
        essential_feats, pi_df, high_impact, "config_v1.5.py")

    # Section 6: Charts
    generate_charts(fi_df, pi_df)

    # Section 7: Decision table
    print_decision_table(fi_df, drop_corr, pi_df, high_impact, low_impact)

    print("\n" + "=" * 70)
    print("  DONE.")
    print(f"  All outputs in: {OUTPUT_DIR}/")
    print("  Key files:")
    print("    feature_selection_charts.png  -- visual summary")
    print("    parameter_importance.csv      -- Optuna param ranking")
    print("    feature_correlation.csv       -- redundancy matrix")
    print("    minimal_filter_set.json       -- use this for v1.6 design")
    print("=" * 70)
