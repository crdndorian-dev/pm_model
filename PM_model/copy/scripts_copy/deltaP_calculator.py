import os
import pandas as pd
import numpy as np

# -----------------------
# Config
# -----------------------
POLY_SNAPSHOTS = "../data/polymarket_price_snapshots.csv"
OPTIONS_RN = "../data/options_risk_neutral_probabilities.csv"
OUTPUT_ALPHA = "../data/delta_p_snapshots.csv"

ROUND_N = 7
TOLERANCE_MINUTES = 15

# Floors for time-normalized ΔP
IV_FLOOR = 0.02
T_DAYS_FLOOR = 1.0

# -----------------------
# Load
# -----------------------
poly = pd.read_csv(POLY_SNAPSHOTS)
rn = pd.read_csv(OPTIONS_RN)

# -----------------------
# Validate columns
# -----------------------
req_poly = {"snapshot_time_utc", "slug", "strike", "yes_price", "no_price", "event_endDate"}
req_rn = {"snapshot_time_utc", "slug", "ticker", "expiry", "strike", "spot_price", "implied_volatility", "risk_neutral_prob"}

missing_poly = req_poly - set(poly.columns)
missing_rn = req_rn - set(rn.columns)

if missing_poly:
    raise ValueError(f"Missing columns in Polymarket snapshots: {missing_poly}")
if missing_rn:
    raise ValueError(f"Missing columns in options RN file: {missing_rn}")

# -----------------------
# Parse datetimes & standardize keys
# -----------------------
poly["snapshot_time_utc"] = pd.to_datetime(poly["snapshot_time_utc"], utc=True, errors="coerce")
poly["event_endDate"] = pd.to_datetime(poly["event_endDate"], utc=True, errors="coerce")
poly = poly.rename(columns={"event_endDate": "expiry"})

rn["snapshot_time_utc"] = pd.to_datetime(rn["snapshot_time_utc"], utc=True, errors="coerce")
rn["expiry"] = pd.to_datetime(rn["expiry"], utc=True, errors="coerce")

# -----------------------
# Numeric conversions
# -----------------------
poly["strike"] = pd.to_numeric(poly["strike"], errors="coerce")
rn["strike"] = pd.to_numeric(rn["strike"], errors="coerce")

poly["yes_price"] = pd.to_numeric(poly["yes_price"], errors="coerce")
poly["no_price"] = pd.to_numeric(poly["no_price"], errors="coerce")

rn["spot_price"] = pd.to_numeric(rn["spot_price"], errors="coerce")
rn["implied_volatility"] = pd.to_numeric(rn["implied_volatility"], errors="coerce")
rn["risk_neutral_prob"] = pd.to_numeric(rn["risk_neutral_prob"], errors="coerce")

# Ensure slugs are strings
poly["slug"] = poly["slug"].astype(str)
rn["slug"] = rn["slug"].astype(str)

# -----------------------
# Drop rows missing join keys
# merge_asof cannot handle nulls in keys
# -----------------------
poly = poly.dropna(subset=["snapshot_time_utc", "slug", "expiry", "strike", "yes_price", "no_price"]).copy()
rn = rn.dropna(subset=["snapshot_time_utc", "slug", "expiry", "strike", "ticker", "spot_price", "implied_volatility", "risk_neutral_prob"]).copy()

# -----------------------
# ASOF MERGE (NEAREST TIME)
# CRITICAL: sort primarily by the 'on' column globally
# -----------------------
sort_cols = ["snapshot_time_utc", "slug", "expiry", "strike"]

poly = poly.sort_values(sort_cols).reset_index(drop=True)
rn   = rn.sort_values(sort_cols).reset_index(drop=True)

# sanity check (optional)
# print("poly time monotonic:", poly["snapshot_time_utc"].is_monotonic_increasing)
# print("rn time monotonic:", rn["snapshot_time_utc"].is_monotonic_increasing)

merged = pd.merge_asof(
    left=poly,
    right=rn,
    on="snapshot_time_utc",
    by=["slug", "expiry", "strike"],
    direction="nearest",
    tolerance=pd.Timedelta(minutes=TOLERANCE_MINUTES),
    suffixes=("_poly", "_rn"),
)

# Keep only matches
merged = merged[merged["ticker"].notna()].copy()

if merged.empty:
    raise ValueError(
        "No matches found with merge_asof. "
        "Try increasing TOLERANCE_MINUTES or confirm expiry/strike/slug align."
    )

# -----------------------
# Time to expiry
# -----------------------
merged["T_days"] = (merged["expiry"] - merged["snapshot_time_utc"]).dt.total_seconds() / (60 * 60 * 24)
merged["T_years"] = merged["T_days"] / 365.25
merged = merged[merged["T_years"] > 0].copy()

# -----------------------
# RN probabilities and edges
# -----------------------
merged["rn_yes"] = merged["risk_neutral_prob"]
merged["rn_no"] = 1.0 - merged["rn_yes"]

merged["delta_p_yes"] = merged["yes_price"] - merged["rn_yes"]
merged["delta_p_no"]  = merged["no_price"]  - merged["rn_no"]

merged["edge_yes"] = merged["rn_yes"] - merged["yes_price"]
merged["edge_no"]  = merged["rn_no"]  - merged["no_price"]

merged["best_side"] = np.where(merged["edge_yes"] >= merged["edge_no"], "BUY_YES", "BUY_NO")
merged["ΔP"] = np.where(merged["best_side"] == "BUY_YES", merged["edge_yes"], merged["edge_no"])

# -----------------------
# Time-normalized ΔP using T_days with floors
# -----------------------
iv_used = np.maximum(merged["implied_volatility"], IV_FLOOR)
t_days_used = np.maximum(merged["T_days"], T_DAYS_FLOOR)

merged["time_normalized_ΔP"] = merged["ΔP"] / (iv_used * np.sqrt(t_days_used))

# -----------------------
# Extra features
# -----------------------
merged["moneyness"] = merged["spot_price"] / merged["strike"]
merged["moneyness_pct"] = (merged["moneyness"] - 1.0) * 100.0

# Rank signals
merged["rank_ΔP"] = merged.groupby(["snapshot_time_utc", "ticker", "expiry"])["ΔP"] \
                          .rank(ascending=False, method="dense")

# -----------------------
# Round numeric columns
# -----------------------
round_cols = [
    "yes_price", "no_price",
    "spot_price", "implied_volatility",
    "rn_yes", "rn_no",
    "delta_p_yes", "delta_p_no",
    "edge_yes", "edge_no",
    "ΔP", "time_normalized_ΔP",
    "T_days", "T_years",
    "moneyness", "moneyness_pct",
    "strike",
]
for c in round_cols:
    if c in merged.columns:
        merged[c] = merged[c].round(ROUND_N)

# -----------------------
# Output columns
# -----------------------
alpha_cols = [
    "snapshot_time_utc",
    "ticker",
    "slug",
    "expiry",
    "strike",
    "spot_price",
    "moneyness_pct",
    "T_days",
    "implied_volatility",
    "yes_price",
    "no_price",
    "rn_yes",
    "rn_no",
    "delta_p_yes",
    "delta_p_no",
    "edge_yes",
    "edge_no",
    "best_side",
    "ΔP",
    "time_normalized_ΔP",
    "rank_ΔP",
    "market_id", "condition_id", "yes_token_id", "no_token_id",
    "market_question",
    "yes_price_source", "no_price_source",
]
alpha_cols = [c for c in alpha_cols if c in merged.columns]
alpha = merged[alpha_cols].copy()

alpha = alpha.sort_values(["snapshot_time_utc", "ticker", "ΔP"], ascending=[True, True, False])

# -----------------------
# Append to output CSV (dedupe)
# -----------------------
dedupe_keys = [c for c in ["snapshot_time_utc", "slug", "expiry", "strike"] if c in alpha.columns]

if os.path.exists(OUTPUT_ALPHA):
    existing = pd.read_csv(OUTPUT_ALPHA)
    if "snapshot_time_utc" in existing.columns:
        existing["snapshot_time_utc"] = pd.to_datetime(existing["snapshot_time_utc"], utc=True, errors="coerce")
    if "expiry" in existing.columns:
        existing["expiry"] = pd.to_datetime(existing["expiry"], utc=True, errors="coerce")

    combined = pd.concat([existing, alpha], ignore_index=True)
    if dedupe_keys:
        combined = combined.drop_duplicates(subset=dedupe_keys, keep="last")
    combined.to_csv(OUTPUT_ALPHA, index=False)
else:
    alpha.to_csv(OUTPUT_ALPHA, index=False)

print(f"✅ Saved delta P table to: {OUTPUT_ALPHA}")
print(f"Rows written this run: {len(alpha)}")
alpha.head(20)
