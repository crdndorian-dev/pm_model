import pandas as pd
import numpy as np

# -----------------------
# USER INPUT
# -----------------------
CHOSEN_TRADES_CSV = "../data/chosen_trades_2026-01-16_Friday.csv"  # <-- change to your file
LAMBDA_FIT_CSV = None  # e.g. "../data/lambda_fit_all.csv" or keep None

# Buckets
EV_BUCKETS = [-1, 0, 0.02, 0.05, 0.10, 0.20, 1.0]
PHAT_BUCKETS = [0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
DAYS_BUCKETS = [-1, 0.5, 1, 2, 3, 5, 10, 999]

# -----------------------
# Load
# -----------------------
df = pd.read_csv(CHOSEN_TRADES_CSV)

required = {"p_hat", "best_side_lambda", "trade_pnl", "win"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in chosen trades file: {missing}")

# Numeric conversions
for c in ["p_hat", "ev_yes", "ev_no", "best_ev_lambda", "trade_pnl", "win", "days_to_expiry", "T_days"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------
# Derived columns
# -----------------------
# predicted probability of winning the chosen side
df["pred_win_prob"] = np.where(df["best_side_lambda"] == "BUY_YES", df["p_hat"], 1.0 - df["p_hat"])

# prefer days_to_expiry if present, else T_days
if "days_to_expiry" in df.columns and df["days_to_expiry"].notna().any():
    df["days_left"] = df["days_to_expiry"]
elif "T_days" in df.columns and df["T_days"].notna().any():
    df["days_left"] = df["T_days"]
else:
    df["days_left"] = np.nan

# -----------------------
# Core summary
# -----------------------
def safe_mean(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan

def safe_std(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.std(ddof=1)) if len(x) > 1 else np.nan

print("\n==============================")
print("SUMMARY")
print("==============================")
print("N trades:", len(df))
print("Total PnL:", float(df["trade_pnl"].sum()))
print("Avg PnL:", safe_mean(df["trade_pnl"]))
print("Median PnL:", float(df["trade_pnl"].median()))
print("% profitable:", float((df["trade_pnl"] > 0).mean()))
print("Win rate:", safe_mean(df["win"]))
print("Mean predicted win prob:", safe_mean(df["pred_win_prob"]))

# Sharpe-like
std_pnl = safe_std(df["trade_pnl"])
if std_pnl and std_pnl > 0:
    print("Sharpe-like (mean/std):", safe_mean(df["trade_pnl"]) / std_pnl)
else:
    print("Sharpe-like (mean/std):", np.nan)

# -----------------------
# Calibration check
# -----------------------
print("\n==============================")
print("CALIBRATION (predicted win prob buckets)")
print("==============================")
df_cal = df.dropna(subset=["pred_win_prob", "win"]).copy()
df_cal["p_bucket"] = pd.cut(df_cal["pred_win_prob"], bins=PHAT_BUCKETS, include_lowest=True)

cal_table = (df_cal.groupby("p_bucket")
                  .agg(n=("win", "count"),
                       avg_pred=("pred_win_prob", "mean"),
                       win_rate=("win", "mean"),
                       avg_pnl=("trade_pnl", "mean"))
                  .reset_index())

print(cal_table.to_string(index=False))

# -----------------------
# EV -> realized PnL relationship
# -----------------------
print("\n==============================")
print("EV → PnL (bucket + correlation)")
print("==============================")
if "best_ev_lambda" in df.columns:
    x = df["best_ev_lambda"]
    y = df["trade_pnl"]
    corr = float(pd.Series(x).corr(pd.Series(y)))
    print("Correlation(best_ev_lambda, trade_pnl):", corr)

    df_ev = df.dropna(subset=["best_ev_lambda", "trade_pnl"]).copy()
    df_ev["ev_bucket"] = pd.cut(df_ev["best_ev_lambda"], bins=EV_BUCKETS, include_lowest=True)

    ev_table = (df_ev.groupby("ev_bucket")
                    .agg(n=("trade_pnl", "count"),
                         avg_ev=("best_ev_lambda", "mean"),
                         avg_pnl=("trade_pnl", "mean"),
                         win_rate=("win", "mean"))
                    .reset_index())

    print(ev_table.to_string(index=False))
else:
    print("Column best_ev_lambda not found in this file.")

# -----------------------
# Split by days_to_expiry buckets (if available)
# -----------------------
if df["days_left"].notna().any():
    print("\n==============================")
    print("SPLIT BY days_to_expiry buckets")
    print("==============================")

    df_days = df.dropna(subset=["days_left"]).copy()
    df_days["days_bucket"] = pd.cut(df_days["days_left"], bins=DAYS_BUCKETS, include_lowest=True)

    days_table = (df_days.groupby("days_bucket")
                        .agg(n=("trade_pnl", "count"),
                             avg_days=("days_left", "mean"),
                             avg_pnl=("trade_pnl", "mean"),
                             win_rate=("win", "mean"),
                             avg_pred=("pred_win_prob", "mean"))
                        .reset_index())

    print(days_table.to_string(index=False))

# -----------------------
# Split by weekday/ticker if present
# -----------------------
if "weekday" in df.columns:
    print("\n==============================")
    print("SPLIT BY weekday")
    print("==============================")
    wk = (df.groupby("weekday")
            .agg(n=("trade_pnl", "count"),
                 avg_pnl=("trade_pnl", "mean"),
                 win_rate=("win", "mean"),
                 avg_pred=("pred_win_prob", "mean"))
            .reset_index()
            .sort_values("n", ascending=False))
    print(wk.to_string(index=False))

if "ticker" in df.columns:
    print("\n==============================")
    print("TOP tickers by PnL (if enough)")
    print("==============================")
    tk = (df.groupby("ticker")
            .agg(n=("trade_pnl", "count"),
                 total_pnl=("trade_pnl", "sum"),
                 avg_pnl=("trade_pnl", "mean"),
                 win_rate=("win", "mean"))
            .reset_index()
            .sort_values("total_pnl", ascending=False))
    print(tk.head(20).to_string(index=False))

# -----------------------
# Optional: show lambda fit grid
# -----------------------
if LAMBDA_FIT_CSV:
    lam_df = pd.read_csv(LAMBDA_FIT_CSV)
    if {"lambda", "mean_cluster_pnl"}.issubset(lam_df.columns):
        lam_df = lam_df.sort_values("mean_cluster_pnl", ascending=False)
        print("\n==============================")
        print("LAMBDA GRID (top 15)")
        print("==============================")
        print(lam_df.head(15).to_string(index=False))
    else:
        print("\nLambda fit CSV provided but missing expected columns: lambda, mean_cluster_pnl")

print("\nDone.")
