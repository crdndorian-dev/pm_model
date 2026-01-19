import math
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timezone

# ----------------------------------------
# CONFIG
# ----------------------------------------
POLYMARKET_SNAPSHOTS = "../data/polymarket_price_snapshots.csv"
TARGET_MARKETS = "../data/polymarket_target_markets.csv"
OUTPUT_RN = "../data/options_risk_neutral_probabilities.csv"

RISK_FREE_RATE = 0.03  # flat rate for now

# ----------------------------------------
# Black–Scholes helpers
# ----------------------------------------

def round7(x):
    return round(float(x), 7) if x is not None else None
    
def compute_d2(S, K, T, r, iv):
    if S <= 0 or K <= 0 or T <= 0 or iv <= 0:
        return None
    return (math.log(S / K) + (r - 0.5 * iv**2) * T) / (iv * math.sqrt(T))

def risk_neutral_prob(S, K, T, r, iv):
    d2 = compute_d2(S, K, T, r, iv)
    return norm.cdf(d2) if d2 is not None else None

# ----------------------------------------
# Load datasets
# ----------------------------------------
snapshots = pd.read_csv(POLYMARKET_SNAPSHOTS)
targets = pd.read_csv(TARGET_MARKETS)

required_targets_cols = {"slug", "ticker"}
if not required_targets_cols.issubset(targets.columns):
    raise ValueError("polymarket_target_markets.csv must contain columns: slug, ticker")

# Join ticker into snapshots
df = snapshots.merge(
    targets[["slug", "ticker"]],
    on="slug",
    how="left"
)

df = df[df["ticker"].notna()].copy()

# ----------------------------------------
# Time handling
# ----------------------------------------
df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True)
df["event_endDate"] = pd.to_datetime(df["event_endDate"], utc=True)

df["T_years"] = (
    df["event_endDate"] - df["snapshot_time_utc"]
).dt.total_seconds() / (365.25 * 24 * 3600)

df = df[df["T_years"] > 0].copy()

# ----------------------------------------
# Fetch options & compute RN probabilities
# ----------------------------------------
rows = []

grouped = df.groupby(["ticker", "event_endDate"])

print(f"Processing {len(grouped)} (ticker, expiry) groups")

for (ticker, expiry), g in grouped:
    try:
        stock = yf.Ticker(ticker)
        expiry_str = expiry.strftime("%Y-%m-%d")

        if expiry_str not in stock.options:
            print(f"⚠️ No options for {ticker} expiry {expiry_str}")
            continue

        chain = stock.option_chain(expiry_str)
        calls = chain.calls.copy()

        # Spot price (close)
        hist = stock.history(period="1d")
        if hist.empty:
            continue
        S = hist["Close"].iloc[-1]

        for _, row in g.iterrows():
            K = row["strike"]
            T = row["T_years"]

            # Find nearest strike
            calls["strike_diff"] = (calls["strike"] - K).abs()
            opt = calls.sort_values("strike_diff").iloc[0]

            iv_raw = opt["impliedVolatility"]

            rn_prob_raw = risk_neutral_prob(
                S=S,
                K=K,
                T=T,
                r=RISK_FREE_RATE,
                iv=iv_raw
)

            rows.append({
                "snapshot_time_utc": row["snapshot_time_utc"],
                "slug": row["slug"],
                "ticker": ticker,
                "expiry": expiry,
                "strike": K,
                "spot_price": S,
                "implied_volatility": round7(iv_raw),
                "T_years": T,
                "risk_neutral_prob": round7(rn_prob_raw)
})


    except Exception as e:
        print(f"❌ Error processing {ticker} {expiry}: {e}")

# ----------------------------------------
# Save output
# ----------------------------------------
rn_df = pd.DataFrame(rows)

if rn_df.empty:
    print("⚠️ No risk-neutral probabilities computed.")
else:
    rn_df.to_csv(OUTPUT_RN, index=False)
    print(f"✅ Saved RN probabilities to {OUTPUT_RN}")

rn_df.head()
