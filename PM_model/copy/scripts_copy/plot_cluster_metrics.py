import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CLEAN_HISTORY = "../data/clean_data_history.csv"
CLUSTER_METRICS = "../data/cluster_curve_metrics_2026-01-16_Friday.csv"  # change if needed

# Pick one cluster to plot the full curve
TICKER = "AAPL"
WEEK_ID = "2026-01-16"
WEEKDAY = "Friday"

def get_time_col(df):
    """Return the best available timestamp column name, else None."""
    if "snapshot_time_utc" in df.columns:
        return "snapshot_time_utc"
    if "snapshot_time" in df.columns:
        return "snapshot_time"
    if "collected_at_utc" in df.columns:
        return "collected_at_utc"
    return None

# -----------------------
# Load strike-level data
# -----------------------
df = pd.read_csv(CLEAN_HISTORY)

# Normalize weekday formatting
if "weekday" in df.columns:
    df["weekday"] = df["weekday"].astype(str).str.strip()

# Check required columns exist
needed = {"ticker", "week_id", "weekday", "strike", "yes_price", "rn_yes", "ΔP"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {CLEAN_HISTORY}: {missing}")

# Filter cluster
df = df[(df["ticker"] == TICKER) &
        (df["week_id"].astype(str) == str(WEEK_ID)) &
        (df["weekday"] == WEEKDAY)].copy()

if df.empty:
    raise ValueError("No rows for this (ticker, week_id, weekday). Check TICKER/WEEK_ID/WEEKDAY.")

# Determine which time column to use
time_col = get_time_col(df)
if time_col is None:
    print("Available columns:", list(df.columns))
    raise ValueError("No timestamp column found. Expected snapshot_time_utc or snapshot_time.")

# Parse time column
df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

# If multiple snapshots that day, take latest
latest_t = df[time_col].max()
df = df[df[time_col] == latest_t].copy()

# Numeric conversions + sort
for c in ["strike", "yes_price", "rn_yes", "ΔP"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["strike", "yes_price", "rn_yes"]).sort_values("strike")

# -----------------------
# Plot 1: probability curves
# -----------------------
plt.figure()
plt.plot(df["strike"], df["yes_price"], marker="o", label="Polymarket YES (p_pm)")
plt.plot(df["strike"], df["rn_yes"], marker="o", label="Options RN (p_rn)")
plt.title(f"{TICKER} {WEEK_ID} ({WEEKDAY}) - PM vs RN probability curve\nsnapshot={latest_t}")
plt.xlabel("Strike")
plt.ylabel("Probability")
plt.legend()
plt.show()

# -----------------------
# Plot 2: ΔP by strike
# -----------------------
plt.figure()
plt.plot(df["strike"], df["ΔP"], marker="o")
plt.title(f"{TICKER} {WEEK_ID} ({WEEKDAY}) - ΔP vs Strike\nsnapshot={latest_t}")
plt.xlabel("Strike")
plt.ylabel("ΔP (RN - price on chosen side)")
plt.axhline(0, linestyle="--")
plt.show()

# -----------------------
# Plot 3: across clusters scatter (EV vs realized PnL)
# -----------------------
cm = pd.read_csv(CLUSTER_METRICS)
if {"chosen_best_ev", "chosen_trade_pnl"}.issubset(cm.columns):
    cm["chosen_best_ev"] = pd.to_numeric(cm["chosen_best_ev"], errors="coerce")
    cm["chosen_trade_pnl"] = pd.to_numeric(cm["chosen_trade_pnl"], errors="coerce")

    plt.figure()
    plt.scatter(cm["chosen_best_ev"], cm["chosen_trade_pnl"])
    plt.title("Across clusters: EV at entry vs realized PnL")
    plt.xlabel("chosen_best_ev")
    plt.ylabel("chosen_trade_pnl")
    plt.axhline(0, linestyle="--")
    plt.show()
else:
    print("Skipping cluster scatter plot: need chosen_best_ev and chosen_trade_pnl in cluster metrics file.")
