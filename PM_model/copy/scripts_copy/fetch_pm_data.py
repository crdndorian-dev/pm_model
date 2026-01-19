import os
import time
import json
import requests
import re
import pandas as pd
from datetime import datetime, timezone

# -----------------------
# Config
# -----------------------
INPUT_SLUGS_CSV = "../data/polymarket_target_markets.csv"
OUTPUT_SNAPSHOTS_CSV = "../data/polymarket_price_snapshots.csv"

GAMMA_EVENT_BY_SLUG = "https://gamma-api.polymarket.com/events/slug/{}"
CLOB_MIDPOINT = "https://clob.polymarket.com/midpoint"  # query param: token_id

SLEEP_BETWEEN_SLUGS_SECONDS = 0.25
SLEEP_BETWEEN_PRICE_CALLS_SECONDS = 0.05

# -----------------------
# Helpers
# -----------------------

def extract_strike_from_question(question: str):
    """
    Extract strike price from market question.
    Example: 'above $205?' -> 205.0
    """
    if not isinstance(question, str):
        return None
    match = re.search(r"\$(\d+)", question)
    return float(match.group(1)) if match else None
    
def normalize_list_field(x):
    """
    Gamma sometimes returns list-like fields as:
      - a Python list
      - a stringified JSON list (e.g. '["a","b"]')
      - None
    Convert to a Python list (or None).
    """
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def get_midpoint(token_id: str):
    """
    Fetch midpoint price for a token from the CLOB API.
    Returns float or None.
    """
    if not token_id:
        return None
    try:
        r = requests.get(CLOB_MIDPOINT, params={"token_id": token_id}, timeout=30)
        r.raise_for_status()
        data = r.json()
        # docs show {"mid":"0.53"} (field name is "mid")
        return safe_float(data.get("mid"))
    except Exception:
        return None

# -----------------------
# Load slugs
# -----------------------
slugs_df = pd.read_csv(INPUT_SLUGS_CSV)
if "slug" not in slugs_df.columns:
    raise ValueError(f"Expected a column named 'slug'. Found: {list(slugs_df.columns)}")

slugs = [s for s in slugs_df["slug"].dropna().astype(str).str.strip().tolist() if s]
if not slugs:
    raise ValueError("No slugs found in polymarket_target_markets.csv")

snapshot_time_utc = datetime.now(timezone.utc).isoformat()
rows = []
errors = []

print(f"Loaded {len(slugs)} slugs. Taking snapshot at {snapshot_time_utc}")

# -----------------------
# Fetch each slug -> markets -> token IDs -> midpoint prices
# -----------------------
for slug in slugs:
    try:
        event = requests.get(GAMMA_EVENT_BY_SLUG.format(slug), timeout=30).json()

        event_id = event.get("id")
        event_title = event.get("title") or event.get("question")
        updated_at = event.get("updatedAt")
        end_date = event.get("endDate") or event.get("endDateIso") or event.get("end_time")

        markets = event.get("markets", [])
        if not isinstance(markets, list) or not markets:
            errors.append({"slug": slug, "error": "No markets found in event response"})
            continue

        for m in markets:
            market_id = m.get("id")
            condition_id = m.get("conditionId") or m.get("condition_id")
            question = m.get("question") or m.get("title")
            strike = extract_strike_from_question(question)

            # Token IDs for YES/NO usually live here
            token_ids = normalize_list_field(m.get("clobTokenIds"))
            yes_token_id = token_ids[0] if isinstance(token_ids, list) and len(token_ids) >= 2 else None
            no_token_id  = token_ids[1] if isinstance(token_ids, list) and len(token_ids) >= 2 else None

            # Prefer CLOB midpoint for both tokens
            yes_mid = get_midpoint(yes_token_id)
            time.sleep(SLEEP_BETWEEN_PRICE_CALLS_SECONDS)
            no_mid  = get_midpoint(no_token_id)
            time.sleep(SLEEP_BETWEEN_PRICE_CALLS_SECONDS)

            # Fallback: if Gamma outcomePrices exists, use it
            outcome_prices = normalize_list_field(m.get("outcomePrices"))
            gamma_yes = gamma_no = None
            if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                gamma_yes = safe_float(outcome_prices[0])
                gamma_no  = safe_float(outcome_prices[1])

            # Choose best available
            yes_price = yes_mid if yes_mid is not None else gamma_yes
            no_price  = no_mid  if no_mid  is not None else gamma_no

            rows.append({
                "snapshot_time_utc": snapshot_time_utc,
                "slug": slug,
                "event_id": event_id,
                "event_title": event_title,
                "event_updatedAt": updated_at,
                "event_endDate": end_date,
                "market_id": market_id,
                "condition_id": condition_id,
                "market_question": question,
                "strike": strike, 
                "yes_token_id": yes_token_id,
                "no_token_id": no_token_id,
                "yes_price": yes_price,
                "no_price": no_price,
                "yes_price_source": "clob_midpoint" if yes_mid is not None else ("gamma_outcomePrices" if gamma_yes is not None else None),
                "no_price_source": "clob_midpoint" if no_mid is not None else ("gamma_outcomePrices" if gamma_no is not None else None),
            })

    except Exception as e:
        errors.append({"slug": slug, "error": str(e)})

    time.sleep(SLEEP_BETWEEN_SLUGS_SECONDS)

snapshots_df = pd.DataFrame(rows)

# ---------------------------------------
# Compute time to expiry
# ---------------------------------------

snapshots_df["snapshot_time_utc"] = pd.to_datetime(
    snapshots_df["snapshot_time_utc"], utc=True, errors="coerce"
)

snapshots_df["event_endDate"] = pd.to_datetime(
    snapshots_df["event_endDate"], utc=True, errors="coerce"
)

snapshots_df["T_days"] = (
    snapshots_df["event_endDate"] - snapshots_df["snapshot_time_utc"]
).dt.total_seconds() / (60 * 60 * 24)

snapshots_df["T_years"] = snapshots_df["T_days"] / 365.25


print(f"Fetched {len(snapshots_df)} market rows.")
if errors:
    print(f"Errors: {len(errors)} (first 5)")
    print(pd.DataFrame(errors).head())



# -----------------------
# Append to CSV
# -----------------------

if len(snapshots_df) > 0:
    if os.path.exists(OUTPUT_SNAPSHOTS_CSV):
        existing = pd.read_csv(OUTPUT_SNAPSHOTS_CSV)
        combined = pd.concat([existing, snapshots_df], ignore_index=True)
        combined.to_csv(OUTPUT_SNAPSHOTS_CSV, index=False)
    else:
        snapshots_df.to_csv(OUTPUT_SNAPSHOTS_CSV, index=False)

    print(f"Saved to: {OUTPUT_SNAPSHOTS_CSV}")

# Sanity check
missing = snapshots_df[snapshots_df["yes_price"].isna() | snapshots_df["no_price"].isna()]
print(f"Rows with missing yes/no price: {len(missing)}")
snapshots_df.head(10)

