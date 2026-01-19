import os
import hashlib
import pandas as pd
import numpy as np

# -----------------------
# CONFIG
# -----------------------
ALPHA_IN = "../data/delta_p_snapshots.csv"              # NEW alpha output file
OUTCOMES_IN = "../data/market_outcomes_latest.csv"      # outcome collector output

OUT_HISTORY = "../data/clean_data_history.csv"
OUT_LATEST  = "../data/clean_data_latest.csv"

# Filter extreme *POLYMARKET* probs only (do NOT filter on RN)
P_MIN = 0.03
P_MAX = 0.97

# Optional: remove last hours (helps avoid RN degeneracy on Friday)
MIN_DAYS_TO_EXPIRY_FOR_TRADEABLE = 0.0  # set to 0.5 for "tradeable" view; keep 0.0 to keep all

ROUND_N = 7

# -----------------------
# Helpers
# -----------------------
def to_dt_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def fmt_human(dt_series):
    return dt_series.dt.strftime("%Y-%m-%d %H:%M")

def sha1_row_id(row, keys):
    raw = "||".join("" if pd.isna(row.get(k)) else str(row.get(k)) for k in keys)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# -----------------------
# Load
# -----------------------
df = pd.read_csv(ALPHA_IN)

# Required columns from new alpha file
required = {"snapshot_time_utc", "ticker", "slug", "expiry", "strike", "yes_price", "no_price", "rn_yes", "rn_no", "ΔP", "time_normalized_ΔP"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {ALPHA_IN}: {missing}")

# Parse times
df["snapshot_time_utc"] = to_dt_utc(df["snapshot_time_utc"])
df["expiry"] = to_dt_utc(df["expiry"])

# Ensure numeric
num_cols = [
    "strike", "yes_price", "no_price", "rn_yes", "rn_no",
    "spot_price", "implied_volatility", "moneyness_pct",
    "T_days", "ΔP", "time_normalized_ΔP", "rank_ΔP",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Bet name for readability
if "market_question" in df.columns:
    df["bet_name"] = df["market_question"]
elif "event_title" in df.columns:
    df["bet_name"] = df["event_title"]
else:
    df["bet_name"] = df["slug"]

# -----------------------
# Add time features (weekday + better alternatives)
# -----------------------
df["weekday"] = df["snapshot_time_utc"].dt.day_name()  # Monday, Tuesday...
df["snapshot_time"] = fmt_human(df["snapshot_time_utc"])
df["expiry_date"] = df["expiry"].dt.strftime("%Y-%m-%d")

# "week_id": weekly cycle identifier (expiry date works well for weekly Friday markets)
df["week_id"] = df["expiry_date"]

# days_to_expiry (prefer T_days if present; else compute)
if "T_days" in df.columns and df["T_days"].notna().any():
    df["days_to_expiry"] = df["T_days"]
else:
    df["days_to_expiry"] = (df["expiry"] - df["snapshot_time_utc"]).dt.total_seconds() / (60*60*24)

df["days_to_expiry"] = pd.to_numeric(df["days_to_expiry"], errors="coerce")

# -----------------------
# Filter extremes (Polymarket prices only)
# -----------------------
mask = df["yes_price"].between(P_MIN, P_MAX)

# Optional: sanity check yes+no around 1 (midpoint noise)
if "no_price" in df.columns:
    mask &= df["no_price"].between(P_MIN, P_MAX)
    mask &= (df["yes_price"] + df["no_price"]).between(0.98, 1.02)

# Optional: remove near-expiry (if you want a "tradeable" subset)
if MIN_DAYS_TO_EXPIRY_FOR_TRADEABLE > 0:
    mask &= df["days_to_expiry"] >= MIN_DAYS_TO_EXPIRY_FOR_TRADEABLE

df = df.loc[mask].copy()

# -----------------------
# Join outcomes (market_outcomes_latest)
# -----------------------
outcomes = pd.read_csv(OUTCOMES_IN)

# Parse outcome times
if "expiry" in outcomes.columns:
    outcomes["expiry"] = to_dt_utc(outcomes["expiry"])
elif "event_endDate" in outcomes.columns:
    outcomes["expiry"] = to_dt_utc(outcomes["event_endDate"])

if "collected_at_utc" in outcomes.columns:
    outcomes["collected_at_utc"] = to_dt_utc(outcomes["collected_at_utc"])

if "strike" in outcomes.columns:
    outcomes["strike"] = pd.to_numeric(outcomes["strike"], errors="coerce")

# Normalize resolution columns
if "inferred_resolved" in outcomes.columns:
    outcomes["resolved"] = outcomes["inferred_resolved"].fillna(False).astype(bool)
elif "outcome" in outcomes.columns:
    outcomes["resolved"] = outcomes["outcome"].notna()
else:
    outcomes["resolved"] = False

if "market_id" in df.columns and "market_id" in outcomes.columns:
    join_keys = ["slug", "market_id"]
    keep = [c for c in ["slug","market_id","resolved","outcome","inferred_winner","collected_at_utc"] if c in outcomes.columns]
    df = df.merge(outcomes[keep], on=join_keys, how="left")
else:
    # fallback join
    join_keys = ["slug","expiry","strike"]
    keep = [c for c in ["slug","expiry","strike","resolved","outcome","inferred_winner","collected_at_utc"] if c in outcomes.columns]
    df = df.merge(outcomes[keep], on=join_keys, how="left")

df["resolved"] = df["resolved"].fillna(False)
if "outcome" not in df.columns:
    df["outcome"] = np.nan

# -----------------------
# Build row_id for dedupe
# -----------------------
# Identity = (ticker, slug, expiry_date, strike, snapshot_time to minute, yes/no, rn_yes)
df["snapshot_time_min"] = df["snapshot_time_utc"].dt.floor("min")

row_id_keys = ["ticker", "slug", "expiry_date", "strike", "snapshot_time_min", "yes_price", "no_price", "rn_yes"]
df["row_id"] = df.apply(lambda r: sha1_row_id(r, row_id_keys), axis=1)

# -----------------------
# Update history (dedupe + refresh resolution)
# -----------------------
if os.path.exists(OUT_HISTORY):
    old = pd.read_csv(OUT_HISTORY)
    combined = pd.concat([old, df], ignore_index=True)

    # Dedupe by row_id when available
    if "row_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["row_id"], keep="last")

    df_hist = combined.copy()
else:
    df_hist = df.copy()

# Re-parse times after reading history
df_hist["snapshot_time_utc"] = to_dt_utc(df_hist["snapshot_time_utc"])
df_hist["expiry"] = to_dt_utc(df_hist["expiry"])
df_hist["snapshot_time"] = fmt_human(df_hist["snapshot_time_utc"])
df_hist["expiry_date"] = df_hist["expiry"].dt.strftime("%Y-%m-%d")
df_hist["weekday"] = df_hist["snapshot_time_utc"].dt.day_name()
df_hist["week_id"] = df_hist["expiry_date"]

# Refresh outcome info in the full history (updates previously unresolved rows)
for c in ["resolved", "outcome", "inferred_winner", "collected_at_utc"]:
    if c in df_hist.columns:
        df_hist = df_hist.drop(columns=[c])

if "market_id" in df_hist.columns and "market_id" in outcomes.columns:
    keep = [c for c in ["slug","market_id","resolved","outcome","inferred_winner","collected_at_utc"] if c in outcomes.columns]
    df_hist = df_hist.merge(outcomes[keep], on=["slug","market_id"], how="left")
else:
    keep = [c for c in ["slug","expiry","strike","resolved","outcome","inferred_winner","collected_at_utc"] if c in outcomes.columns]
    df_hist = df_hist.merge(outcomes[keep], on=["slug","expiry","strike"], how="left")

df_hist["resolved"] = df_hist["resolved"].fillna(False)

# -----------------------
# Latest view: one row per contract (ticker+slug+expiry+strike)
# -----------------------
latest_keys = [k for k in ["ticker", "slug", "expiry_date", "strike"] if k in df_hist.columns]
df_latest = (
    df_hist.sort_values("snapshot_time_utc")
           .drop_duplicates(subset=latest_keys, keep="last")
           .sort_values(["ticker", "expiry_date", "strike"], kind="stable")
)

# -----------------------
# Round numeric columns for readability
# -----------------------
round_cols = ["strike","yes_price","no_price","rn_yes","rn_no","spot_price","moneyness_pct",
              "implied_volatility","T_days","days_to_expiry","ΔP","time_normalized_ΔP","rank_ΔP"]
for c in round_cols:
    if c in df_hist.columns:
        df_hist[c] = pd.to_numeric(df_hist[c], errors="coerce").round(ROUND_N)
    if c in df_latest.columns:
        df_latest[c] = pd.to_numeric(df_latest[c], errors="coerce").round(ROUND_N)

# -----------------------
# Select & order columns (most logical + minimal)
# -----------------------
ordered_cols = [
    # observation time
    "snapshot_time", "weekday",

    # contract identity
    "bet_name", "ticker", "slug", "market_id", "event_id",

    # contract terms
    "expiry_date", "week_id", "strike", "days_to_expiry",

    # options context
    "spot_price", "moneyness_pct", "implied_volatility",

    # prices + RN baseline
    "yes_price", "no_price", "rn_yes", "rn_no",

    # signals
    "best_side", "ΔP", "time_normalized_ΔP", "rank_ΔP",

    # outcome info
    "resolved", "outcome", "inferred_winner", "collected_at_utc",

    # execution ids
    "yes_token_id", "no_token_id",
]

def existing_cols(df_in, cols):
    return [c for c in cols if c in df_in.columns]

hist_out = df_hist[existing_cols(df_hist, ordered_cols + ["row_id"])].copy()
latest_out = df_latest[existing_cols(df_latest, ordered_cols + ["row_id"])].copy()

# Save
hist_out.to_csv(OUT_HISTORY, index=False)
latest_out.to_csv(OUT_LATEST, index=False)

print(f"✅ Wrote clean history: {OUT_HISTORY} ({len(hist_out)} rows)")
print(f"✅ Wrote clean latest : {OUT_LATEST} ({len(latest_out)} rows)")

latest_out.head(30)
