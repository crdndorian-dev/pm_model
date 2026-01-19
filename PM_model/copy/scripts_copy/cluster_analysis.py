import pandas as pd
import numpy as np

# ============================================================
# USER INPUT (edit these)
# ============================================================
INPUT_CSV = "../data/clean_data_history.csv"

# Analyze only this week + weekday
WEEK_ID = "2026-01-16"      # typically expiry_date / week_id (YYYY-MM-DD)
WEEKDAY = "Friday"          # "Monday", "Tuesday", ...

# Fit shrinkage lambda using historical data:
# - "same_weekday"  -> fit lambda using ALL weeks but only this WEEKDAY
# - "all_weekdays"  -> fit lambda using all rows (any weekday)
LAMBDA_FIT_SCOPE = "same_weekday"

# Strategy / selection settings
ONE_TRADE_PER_CLUSTER = True          # recommended for dependent strikes
ATM_BAND_MONEYNESS_PCT = 5.0          # define "near ATM" as abs(moneyness_pct) <= 5%
LAMBDA_GRID = np.linspace(0.0, 1.0, 101)  # grid-search lambda in [0,1]

# Optional quality filters (set to None to disable)
PRICE_MIN = 0.05
PRICE_MAX = 0.95
MIN_DAYS_TO_EXPIRY = None             # e.g. 1.0 or 2.0 to avoid very late-week noise

# ============================================================
# Helpers
# ============================================================

def to_dt_utc_safe(series):
    return pd.to_datetime(series, utc=True, errors="coerce")

def compute_trade_pnl(best_side, outcome, yes_price, no_price):
    """
    PnL per $1 staked at midpoint price.
    BUY_YES: pnl = outcome - yes_price
    BUY_NO : pnl = (1-outcome) - no_price
    """
    if pd.isna(outcome):
        return np.nan
    if best_side == "BUY_YES":
        return float(outcome) - float(yes_price)
    if best_side == "BUY_NO":
        return (1.0 - float(outcome)) - float(no_price)
    return np.nan

def pm_median_strike(strikes, pm_yes):
    """
    Approximate strike where PM probability crosses 0.5 (median strike).
    Uses linear interpolation on sorted strikes.
    Returns np.nan if not bracketed.
    """
    if len(strikes) < 2:
        return np.nan
    order = np.argsort(strikes)
    K = np.array(strikes)[order]
    P = np.array(pm_yes)[order]

    # find adjacent points that bracket 0.5
    for i in range(len(K) - 1):
        p1, p2 = P[i], P[i+1]
        if (p1 - 0.5) == 0:
            return float(K[i])
        if (p1 - 0.5) * (p2 - 0.5) < 0:
            # interpolate between (K[i], P[i]) and (K[i+1], P[i+1])
            if p2 == p1:
                return float((K[i] + K[i+1]) / 2.0)
            w = (0.5 - p1) / (p2 - p1)
            return float(K[i] + w * (K[i+1] - K[i]))
    return np.nan

def curve_features_for_cluster(df_cluster, atm_band=5.0):
    """
    df_cluster = all strikes for ONE (ticker, week_id) at ONE snapshot time
    Must have: strike, yes_price, rn_yes, ΔP, time_normalized_ΔP, moneyness_pct (optional)
    Returns a dict of curve-level features.
    """
    out = {}
    out["n_strikes"] = int(len(df_cluster))

    # Use moneyness_pct if available; else compute from spot_price if present
    if "moneyness_pct" in df_cluster.columns and df_cluster["moneyness_pct"].notna().any():
        m = pd.to_numeric(df_cluster["moneyness_pct"], errors="coerce")
    elif "spot_price" in df_cluster.columns and df_cluster["spot_price"].notna().any():
        m = (pd.to_numeric(df_cluster["spot_price"], errors="coerce") / pd.to_numeric(df_cluster["strike"], errors="coerce") - 1.0) * 100.0
    else:
        m = pd.Series([np.nan] * len(df_cluster))

    # Near-ATM mask
    atm_mask = m.abs() <= atm_band

    # Mean ΔP near ATM (if none ATM points exist, use all points)
    dp = pd.to_numeric(df_cluster["ΔP"], errors="coerce")
    tndp = pd.to_numeric(df_cluster["time_normalized_ΔP"], errors="coerce")

    if atm_mask.any():
        out["mean_ΔP_atm"] = float(dp[atm_mask].mean())
        out["mean_time_norm_ΔP_atm"] = float(tndp[atm_mask].mean())
        out["mean_abs_pm_minus_rn_atm"] = float((pd.to_numeric(df_cluster["yes_price"], errors="coerce")[atm_mask]
                                                 - pd.to_numeric(df_cluster["rn_yes"], errors="coerce")[atm_mask]).abs().mean())
    else:
        out["mean_ΔP_atm"] = float(dp.mean())
        out["mean_time_norm_ΔP_atm"] = float(tndp.mean())
        out["mean_abs_pm_minus_rn_atm"] = float((pd.to_numeric(df_cluster["yes_price"], errors="coerce")
                                                 - pd.to_numeric(df_cluster["rn_yes"], errors="coerce")).abs().mean())

    # Where is the max signal located?
    idx_max = tndp.idxmax() if tndp.notna().any() else dp.idxmax()
    if pd.notna(idx_max):
        out["strike_at_max_signal"] = float(df_cluster.loc[idx_max, "strike"])
        out["moneyness_pct_at_max_signal"] = float(m.loc[idx_max]) if pd.notna(m.loc[idx_max]) else np.nan
        out["max_ΔP"] = float(dp.loc[idx_max]) if pd.notna(dp.loc[idx_max]) else np.nan
        out["max_time_norm_ΔP"] = float(tndp.loc[idx_max]) if pd.notna(tndp.loc[idx_max]) else np.nan
        out["best_side_at_max_signal"] = df_cluster.loc[idx_max, "best_side"]
    else:
        out["strike_at_max_signal"] = np.nan
        out["moneyness_pct_at_max_signal"] = np.nan
        out["max_ΔP"] = np.nan
        out["max_time_norm_ΔP"] = np.nan
        out["best_side_at_max_signal"] = None

    # Slope of ΔP vs strike (simple linear fit)
    K = pd.to_numeric(df_cluster["strike"], errors="coerce").to_numpy()
    y = dp.to_numpy()
    good = np.isfinite(K) & np.isfinite(y)
    if good.sum() >= 2:
        # slope per $1 of strike
        slope = np.polyfit(K[good], y[good], 1)[0]
        out["slope_ΔP_vs_strike"] = float(slope)
    else:
        out["slope_ΔP_vs_strike"] = np.nan

    # Median strike for PM and RN (where prob crosses 0.5)
    pm_yes = pd.to_numeric(df_cluster["yes_price"], errors="coerce").to_numpy()
    rn_yes = pd.to_numeric(df_cluster["rn_yes"], errors="coerce").to_numpy()

    out["pm_median_strike"] = pm_median_strike(K, pm_yes)
    out["rn_median_strike"] = pm_median_strike(K, rn_yes)

    if np.isfinite(out["pm_median_strike"]) and np.isfinite(out["rn_median_strike"]):
        out["median_strike_diff_pm_minus_rn"] = float(out["pm_median_strike"] - out["rn_median_strike"])
    else:
        out["median_strike_diff_pm_minus_rn"] = np.nan

    return out

def choose_trade_with_lambda(df_curve, lam):
    """
    Given all strikes for ONE ticker-week curve at one snapshot time,
    build a shrinkage probability and choose the best EV trade (YES or NO) among strikes.

    p_hat = pm_yes + lam * (rn_yes - pm_yes)
    EV_yes = p_hat - yes_price
    EV_no  = (1 - p_hat) - no_price
    Select the single row with max(EV_yes, EV_no)
    """
    pm_yes = pd.to_numeric(df_curve["yes_price"], errors="coerce")
    rn_yes = pd.to_numeric(df_curve["rn_yes"], errors="coerce")
    no_price = pd.to_numeric(df_curve["no_price"], errors="coerce")

    p_hat = pm_yes + lam * (rn_yes - pm_yes)
    ev_yes = p_hat - pm_yes
    ev_no = (1.0 - p_hat) - no_price

    best_ev = np.maximum(ev_yes, ev_no)
    best_side = np.where(ev_yes >= ev_no, "BUY_YES", "BUY_NO")

    tmp = df_curve.copy()
    tmp["p_hat"] = p_hat
    tmp["ev_yes"] = ev_yes
    tmp["ev_no"] = ev_no
    tmp["best_ev_lambda"] = best_ev
    tmp["best_side_lambda"] = best_side

    # choose the strike with max best_ev (ties broken by first occurrence)
    idx = tmp["best_ev_lambda"].idxmax()
    return tmp.loc[[idx]].copy()

# ============================================================
# Load and clean minimal columns
# ============================================================

df = pd.read_csv(INPUT_CSV)

# Required columns for this analysis
need = {"ticker", "week_id", "weekday", "strike", "yes_price", "no_price", "rn_yes", "best_side", "ΔP", "time_normalized_ΔP", "outcome"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {INPUT_CSV}: {missing}")

# Time handling (clean_data may have snapshot_time_utc or snapshot_time string)
if "snapshot_time_utc" in df.columns:
    df["snapshot_time_utc"] = to_dt_utc_safe(df["snapshot_time_utc"])
elif "snapshot_time" in df.columns:
    # snapshot_time is like "YYYY-MM-DD HH:MM"
    df["snapshot_time_utc"] = to_dt_utc_safe(df["snapshot_time"])
else:
    raise ValueError("Need either snapshot_time_utc or snapshot_time column to choose the latest snapshot per day.")

# Numeric coercion
for col in ["strike","yes_price","no_price","rn_yes","rn_no","ΔP","time_normalized_ΔP","implied_volatility","T_days","days_to_expiry","spot_price","moneyness_pct","outcome"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep only resolved rows (we need outcome to compute pnl)
df = df[df["outcome"].notna()].copy()

# Quality filters
if PRICE_MIN is not None:
    df = df[df["yes_price"] >= float(PRICE_MIN)]
if PRICE_MAX is not None:
    df = df[df["yes_price"] <= float(PRICE_MAX)]
if "no_price" in df.columns and PRICE_MIN is not None:
    df = df[df["no_price"] >= float(PRICE_MIN)]
if "no_price" in df.columns and PRICE_MAX is not None:
    df = df[df["no_price"] <= float(PRICE_MAX)]

if MIN_DAYS_TO_EXPIRY is not None:
    # prefer days_to_expiry if present, else T_days
    if "days_to_expiry" in df.columns and df["days_to_expiry"].notna().any():
        df = df[df["days_to_expiry"] >= float(MIN_DAYS_TO_EXPIRY)]
    elif "T_days" in df.columns and df["T_days"].notna().any():
        df = df[df["T_days"] >= float(MIN_DAYS_TO_EXPIRY)]

if df.empty:
    raise ValueError("No rows left after filters. Relax PRICE_MIN/MAX or MIN_DAYS_TO_EXPIRY.")

# ============================================================
# Fit shrinkage lambda on history (cluster-aware)
# ============================================================

fit_df = df.copy()
if LAMBDA_FIT_SCOPE == "same_weekday":
    fit_df = fit_df[fit_df["weekday"].astype(str) == str(WEEKDAY)].copy()
elif LAMBDA_FIT_SCOPE == "all_weekdays":
    pass
else:
    raise ValueError("LAMBDA_FIT_SCOPE must be 'same_weekday' or 'all_weekdays'")

if fit_df.empty:
    raise ValueError("No data available to fit lambda under the chosen LAMBDA_FIT_SCOPE.")

# For each (ticker, week_id, weekday), we will choose ONE snapshot time (latest that day),
# then treat all strikes at that time as the curve.
fit_df = fit_df.sort_values("snapshot_time_utc").copy()
fit_df_latest = (fit_df.groupby(["ticker","week_id","weekday"], as_index=False)
                      .apply(lambda g: g[g["snapshot_time_utc"] == g["snapshot_time_utc"].max()])
                      .reset_index(drop=True))

# Now we evaluate lambda by selecting ONE trade per cluster (ticker, week_id) at that weekday
def evaluate_lambda_on_df(df_in, lam):
    rows = []
    for (ticker, week_id, weekday), g in df_in.groupby(["ticker","week_id","weekday"]):
        # pick best strike using lambda-driven EV
        chosen = choose_trade_with_lambda(g, lam)
        # realized pnl from that choice
        side = chosen["best_side_lambda"].iloc[0]
        pnl = compute_trade_pnl(side, chosen["outcome"].iloc[0], chosen["yes_price"].iloc[0], chosen["no_price"].iloc[0])
        rows.append(pnl)
    pnls = pd.Series(rows, dtype="float64").dropna()
    return pnls.mean() if len(pnls) else np.nan

lambda_scores = []
for lam in LAMBDA_GRID:
    mean_pnl = evaluate_lambda_on_df(fit_df_latest, lam)
    lambda_scores.append((lam, mean_pnl))

lambda_scores_df = pd.DataFrame(lambda_scores, columns=["lambda", "mean_cluster_pnl"])
lambda_scores_df = lambda_scores_df.dropna().sort_values("mean_cluster_pnl", ascending=False)

if lambda_scores_df.empty:
    raise ValueError("Lambda fit failed (no valid PnL computed). Check that outcome/price columns are valid.")

best_lambda = float(lambda_scores_df.iloc[0]["lambda"])
best_lambda_mean_pnl = float(lambda_scores_df.iloc[0]["mean_cluster_pnl"])

print("\n=== Shrinkage fit ===")
print(f"Scope: {LAMBDA_FIT_SCOPE}")
print(f"Best lambda: {best_lambda:.3f} (mean cluster pnl in-sample: {best_lambda_mean_pnl:.4f})")
print("Top 10 lambdas:")
print(lambda_scores_df.head(10).to_string(index=False))

# ============================================================
# Analyze the specific week + weekday the user asked for
# ============================================================

an_df = df[(df["week_id"].astype(str) == str(WEEK_ID)) &
           (df["weekday"].astype(str) == str(WEEKDAY))].copy()

if an_df.empty:
    raise ValueError("No rows for the selected WEEK_ID/WEEKDAY after filtering. Check inputs or relax filters.")

# Use the latest snapshot time per (ticker, week_id, weekday) for curve features
an_df = an_df.sort_values("snapshot_time_utc").copy()
an_latest = (an_df.groupby(["ticker","week_id","weekday"], as_index=False)
                  .apply(lambda g: g[g["snapshot_time_utc"] == g["snapshot_time_utc"].max()])
                  .reset_index(drop=True))

# Build curve-level features per cluster + select trade using shrinkage lambda
cluster_rows = []
trade_rows = []

for (ticker, week_id, weekday), g in an_latest.groupby(["ticker","week_id","weekday"]):
    # curve features
    feats = curve_features_for_cluster(g, atm_band=ATM_BAND_MONEYNESS_PCT)
    feats.update({"ticker": ticker, "week_id": week_id, "weekday": weekday})

    # choose best trade using shrinkage lambda
    chosen = choose_trade_with_lambda(g, best_lambda)
    side = chosen["best_side_lambda"].iloc[0]
    pnl = compute_trade_pnl(side, chosen["outcome"].iloc[0], chosen["yes_price"].iloc[0], chosen["no_price"].iloc[0])
    win = 1 if ((side == "BUY_YES" and chosen["outcome"].iloc[0] == 1) or (side == "BUY_NO" and chosen["outcome"].iloc[0] == 0)) else 0

    # cluster summary
    feats["chosen_strike"] = float(chosen["strike"].iloc[0])
    feats["chosen_side"] = side
    feats["chosen_yes_price"] = float(chosen["yes_price"].iloc[0])
    feats["chosen_no_price"] = float(chosen["no_price"].iloc[0])
    feats["chosen_rn_yes"] = float(chosen["rn_yes"].iloc[0])
    feats["chosen_p_hat"] = float(chosen["p_hat"].iloc[0])
    feats["chosen_best_ev"] = float(chosen["best_ev_lambda"].iloc[0])
    feats["chosen_trade_pnl"] = float(pnl)
    feats["chosen_win"] = int(win)

    # optional: also record what your precomputed signal was at that strike
    feats["chosen_ΔP"] = float(chosen["ΔP"].iloc[0]) if "ΔP" in chosen.columns else np.nan
    feats["chosen_time_norm_ΔP"] = float(chosen["time_normalized_ΔP"].iloc[0]) if "time_normalized_ΔP" in chosen.columns else np.nan

    cluster_rows.append(feats)

    # save chosen trade row too
    chosen_out = chosen.copy()
    chosen_out["lambda_used"] = best_lambda
    chosen_out["chosen_side"] = side
    chosen_out["trade_pnl"] = pnl
    chosen_out["win"] = win
    trade_rows.append(chosen_out)

cluster_df = pd.DataFrame(cluster_rows)
trades_df = pd.concat(trade_rows, ignore_index=True)

# Portfolio summary across clusters for this week+weekday
pnls = cluster_df["chosen_trade_pnl"].astype(float)
summary = {
    "week_id": WEEK_ID,
    "weekday": WEEKDAY,
    "lambda_used": best_lambda,
    "n_clusters": int(len(cluster_df)),
    "total_pnl": float(pnls.sum()),
    "avg_pnl": float(pnls.mean()),
    "median_pnl": float(pnls.median()),
    "win_rate": float(cluster_df["chosen_win"].mean()),
    "pct_profitable_clusters": float((pnls > 0).mean()),
}

print("\n=== Week/weekday cluster summary (one trade per ticker-week) ===")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n=== Cluster table (sorted by chosen_trade_pnl) ===")
show_cols = [
    "ticker","week_id","weekday",
    "chosen_side","chosen_strike","chosen_trade_pnl","chosen_win",
    "chosen_yes_price","chosen_no_price","chosen_rn_yes","chosen_p_hat","chosen_best_ev",
    "max_time_norm_ΔP","mean_time_norm_ΔP_atm","mean_ΔP_atm","slope_ΔP_vs_strike",
    "pm_median_strike","rn_median_strike","median_strike_diff_pm_minus_rn"
]
show_cols = [c for c in show_cols if c in cluster_df.columns]
print(cluster_df.sort_values("chosen_trade_pnl", ascending=False)[show_cols].head(30).to_string(index=False))

# Save outputs
OUT_CLUSTER = f"../data/cluster_curve_metrics_{WEEK_ID}_{WEEKDAY}.csv".replace(" ", "_")
OUT_TRADES = f"../data/chosen_trades_{WEEK_ID}_{WEEKDAY}.csv".replace(" ", "_")
OUT_LAMBDAS = f"../data/lambda_fit_{WEEKDAY if LAMBDA_FIT_SCOPE=='same_weekday' else 'all'}.csv".replace(" ", "_")

cluster_df.to_csv(OUT_CLUSTER, index=False)
trades_df.to_csv(OUT_TRADES, index=False)
lambda_scores_df.to_csv(OUT_LAMBDAS, index=False)

print(f"\nSaved cluster curve metrics to: {OUT_CLUSTER}")
print(f"Saved chosen trades to:        {OUT_TRADES}")
print(f"Saved lambda fit grid to:      {OUT_LAMBDAS}")
