import os
import re
import json
import time
from datetime import datetime, timezone

import requests
import pandas as pd


# -----------------------
# CONFIG
# -----------------------
TARGET_MARKETS_CSV = "../data/polymarket_target_markets.csv"  # must contain columns: slug, ticker
OUTCOMES_HISTORY_CSV = "../data/market_outcomes_history.csv"
OUTCOMES_LATEST_CSV = "../data/market_outcomes_latest.csv"

GAMMA_BASE = "https://gamma-api.polymarket.com"
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_CALLS_SEC = 0.2  # be polite


# -----------------------
# Helpers
# -----------------------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_json_load(x):
    """Gamma sometimes returns arrays as JSON strings. Handle both list and string."""
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        x = x.strip()
        # Try JSON decode if it looks like JSON
        if (x.startswith("[") and x.endswith("]")) or (x.startswith("{") and x.endswith("}")):
            try:
                return json.loads(x)
            except Exception:
                return x
        return x
    return x


def extract_strike_from_question(question: str):
    """
    Extract strike from text like:
      "Will NVDA close above $205?" -> 205.0
    """
    if not isinstance(question, str):
        return None
    m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", question)
    return float(m.group(1)) if m else None


def infer_resolution_from_prices(outcomes, outcome_prices, eps=1e-3):
    """
    If resolved, outcomePrices often become ~[1,0] or ~[0,1].
    Returns (resolved_bool, winner_label, outcome_1_for_yes_else_0, yes_price, no_price)
    """
    if not isinstance(outcomes, list) or not isinstance(outcome_prices, list):
        return (False, None, None, None, None)
    if len(outcomes) < 2 or len(outcome_prices) < 2:
        return (False, None, None, None, None)

    # Identify YES/NO indices
    outcomes_lower = [str(o).strip().lower() for o in outcomes]
    yes_idx = outcomes_lower.index("yes") if "yes" in outcomes_lower else 0
    no_idx = outcomes_lower.index("no") if "no" in outcomes_lower else (1 if yes_idx == 0 else 0)

    try:
        yes_p = float(outcome_prices[yes_idx])
        no_p = float(outcome_prices[no_idx])
    except Exception:
        return (False, None, None, None, None)

    # Resolution heuristic: one side ~1, the other ~0
    if yes_p >= 1.0 - eps and no_p <= eps:
        return (True, "Yes", 1, yes_p, no_p)
    if no_p >= 1.0 - eps and yes_p <= eps:
        return (True, "No", 0, yes_p, no_p)

    # Not clearly resolved yet
    return (False, None, None, yes_p, no_p)


def fetch_event_by_slug(slug: str):
    """
    Gamma: GET /events/slug/{slug}
    """
    url = f"{GAMMA_BASE}/events/slug/{slug}"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# -----------------------
# Main
# -----------------------
def main():
    targets = pd.read_csv(TARGET_MARKETS_CSV)

    if "slug" not in targets.columns or "ticker" not in targets.columns:
        raise ValueError("polymarket_target_markets.csv must contain columns: slug, ticker")

    slugs = targets["slug"].dropna().astype(str).unique().tolist()
    slug_to_ticker = dict(zip(targets["slug"].astype(str), targets["ticker"].astype(str)))

    collected_at = utc_now_iso()
    rows = []

    print(f"Collecting outcomes for {len(slugs)} event slugs...")

    for i, slug in enumerate(slugs, start=1):
        ticker = slug_to_ticker.get(slug, None)
        try:
            event = fetch_event_by_slug(slug)
        except Exception as e:
            print(f"❌ [{i}/{len(slugs)}] Failed slug={slug}: {e}")
            continue

        # Event-level fields
        event_id = event.get("id")
        event_title = event.get("title")
        event_end = event.get("endDate") or event.get("endDateIso")  # depends on payload
        event_closed = event.get("closed")
        event_active = event.get("active")

        markets = event.get("markets", []) or []
        if not markets:
            print(f"⚠️ [{i}/{len(slugs)}] No markets found in event payload for slug={slug}")
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)
            continue

        for m in markets:
            market_id = m.get("id")
            condition_id = m.get("conditionId")
            question = m.get("question")

            # Prefer market endDate if present, else event end
            expiry = m.get("endDate") or m.get("endDateIso") or event_end

            # Arrays may come as JSON strings
            outcomes = safe_json_load(m.get("outcomes"))
            outcome_prices = safe_json_load(m.get("outcomePrices"))
            clob_token_ids = safe_json_load(m.get("clobTokenIds"))

            strike = extract_strike_from_question(question)

            # Status fields
            closed = m.get("closed")
            active = m.get("active")
            resolved_by = m.get("resolvedBy")
            uma_status = m.get("umaResolutionStatus")

            # Infer resolution from outcomePrices (most reliable for your backtests)
            inferred_resolved, winner_label, outcome_binary, yes_price_final, no_price_final = infer_resolution_from_prices(
                outcomes, outcome_prices
            )

            # You can choose stricter "resolved" logic if you want:
            # resolved_flag = inferred_resolved and bool(closed)
            # For now, store both.
            resolved_flag = inferred_resolved

            rows.append({
                "collected_at_utc": collected_at,

                "slug": slug,
                "ticker": ticker,
                "event_id": event_id,
                "event_title": event_title,
                "event_endDate": event_end,
                "event_closed": event_closed,
                "event_active": event_active,

                "market_id": market_id,
                "condition_id": condition_id,
                "question": question,
                "expiry": expiry,
                "strike": strike,

                "outcomes": json.dumps(outcomes) if isinstance(outcomes, list) else outcomes,
                "outcomePrices": json.dumps(outcome_prices) if isinstance(outcome_prices, list) else outcome_prices,
                "clobTokenIds": json.dumps(clob_token_ids) if isinstance(clob_token_ids, list) else clob_token_ids,

                "market_closed": closed,
                "market_active": active,
                "resolved_by": resolved_by,
                "uma_resolution_status": uma_status,

                "inferred_resolved": resolved_flag,
                "inferred_winner": winner_label,      # "Yes" / "No" / None
                "outcome": outcome_binary,            # 1 if YES wins, 0 if NO wins, None if unresolved
                "yes_price_final": yes_price_final,
                "no_price_final": no_price_final,
            })

        print(f"✅ [{i}/{len(slugs)}] slug={slug} markets={len(markets)}")
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    outcomes_df = pd.DataFrame(rows)

    if outcomes_df.empty:
        print("⚠️ No outcome rows collected. Nothing to write.")
        return

    # -----------------------
    # Write HISTORY (append)
    # -----------------------
    if os.path.exists(OUTCOMES_HISTORY_CSV):
        old = pd.read_csv(OUTCOMES_HISTORY_CSV)
        hist = pd.concat([old, outcomes_df], ignore_index=True)
        hist.to_csv(OUTCOMES_HISTORY_CSV, index=False)
    else:
        outcomes_df.to_csv(OUTCOMES_HISTORY_CSV, index=False)

    # -----------------------
    # Write LATEST (dedup)
    # Keep latest row per (slug, market_id) based on collected_at_utc
    # -----------------------
    latest = outcomes_df.copy()
    latest["collected_at_utc"] = pd.to_datetime(latest["collected_at_utc"], utc=True, errors="coerce")
    latest = latest.sort_values("collected_at_utc").drop_duplicates(subset=["slug", "market_id"], keep="last")

    if os.path.exists(OUTCOMES_LATEST_CSV):
        old_latest = pd.read_csv(OUTCOMES_LATEST_CSV)
        old_latest["collected_at_utc"] = pd.to_datetime(old_latest["collected_at_utc"], utc=True, errors="coerce")

        combined = pd.concat([old_latest, latest], ignore_index=True)
        combined = combined.sort_values("collected_at_utc").drop_duplicates(subset=["slug", "market_id"], keep="last")
        combined.to_csv(OUTCOMES_LATEST_CSV, index=False)
    else:
        latest.to_csv(OUTCOMES_LATEST_CSV, index=False)

    resolved_count = int(outcomes_df["inferred_resolved"].fillna(False).sum())
    print(f"\n✅ Wrote {len(outcomes_df)} rows to history.")
    print(f"✅ Latest file updated. Resolved markets found this run: {resolved_count}")
    print(f"📄 HISTORY: {OUTCOMES_HISTORY_CSV}")
    print(f"📄 LATEST : {OUTCOMES_LATEST_CSV}")


if __name__ == "__main__":
    main()
