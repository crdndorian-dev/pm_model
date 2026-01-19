#!/usr/bin/env python3
"""
weekly_label_and_merge.py

Goal:
- Maintain ONE single CSV that contains:
  (A) your full snapshot history (all rows)
  (B) the best-known outcome for each market (slug, market_id), updated over time

Behavior:
- Reads your snapshot history CSV (append-only).
- Adds a weekday label for each snapshot (local TZ).
- Loads the existing combined CSV (if it exists).
- Appends any new snapshots not already present.
- Fetches outcomes from Polymarket Gamma only for markets that are:
    - newly seen, OR
    - previously unresolved (y missing / is_resolved false)
- Updates outcome columns IN-PLACE for all matching rows.
- Writes back the same single combined CSV (overwrite).

Usage:
python weekly_label_and_merge.py \
  --snapshot-history "/Users/you/.../data/history/final_dataset_history.csv" \
  --combined-out "/Users/you/.../data/history/snapshots_with_outcomes.csv" \
  --tz "Europe/Paris" \
  --only-expired
"""

import os
import re
import json
import time
import math
import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # python<3.9 not supported


# -----------------------
# Defaults
# -----------------------
DEFAULT_GAMMA_BASE = "https://gamma-api.polymarket.com"
DEFAULT_TIMEOUT = 25
DEFAULT_SLEEP_SEC = 0.25

# outcome inference tolerance
EPS = 1e-3


# -----------------------
# Gamma client
# -----------------------
@dataclass
class GammaClient:
    base: str = DEFAULT_GAMMA_BASE
    timeout: int = DEFAULT_TIMEOUT
    sleep_sec: float = DEFAULT_SLEEP_SEC

    def fetch_event_by_slug(self, slug: str) -> Dict[str, Any]:
        url = f"{self.base}/events/slug/{slug}"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# -----------------------
# Helpers
# -----------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_utc_dt(x) -> pd.Series:
    return pd.to_datetime(x, utc=True, errors="coerce")


def safe_json_load(x):
    """Gamma sometimes returns lists as JSON strings; normalize to python objects when possible."""
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return x
        return x
    return x


def extract_strike_K_from_question(question: str) -> Optional[float]:
    """Basic strike extractor: looks for '$123.45' patterns."""
    if not isinstance(question, str):
        return None
    m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", question)
    return float(m.group(1)) if m else None


def infer_resolution_from_prices(
    outcomes: Any, outcome_prices: Any, eps: float = EPS
) -> Tuple[bool, Optional[str], Optional[int], Optional[float], Optional[float]]:
    """
    Infer final resolution from outcomePrices.
    Returns: (is_resolved, winner_label, y, yes_final, no_final)
      y=1 if YES wins else 0 if NO wins.
    """
    outcomes = safe_json_load(outcomes)
    outcome_prices = safe_json_load(outcome_prices)

    if not isinstance(outcomes, list) or not isinstance(outcome_prices, list):
        return (False, None, None, None, None)
    if len(outcomes) < 2 or len(outcome_prices) < 2:
        return (False, None, None, None, None)

    outs = [str(o).strip().lower() for o in outcomes]
    yes_idx = outs.index("yes") if "yes" in outs else 0
    no_idx = outs.index("no") if "no" in outs else (1 if yes_idx == 0 else 0)

    try:
        yes_p = float(outcome_prices[yes_idx])
        no_p = float(outcome_prices[no_idx])
    except Exception:
        return (False, None, None, None, None)

    if yes_p >= 1.0 - eps and no_p <= eps:
        return (True, "YES", 1, yes_p, no_p)
    if no_p >= 1.0 - eps and yes_p <= eps:
        return (True, "NO", 0, yes_p, no_p)

    return (False, None, None, yes_p, no_p)


def stable_snapshot_id(row: pd.Series) -> str:
    """
    Stable row id so we can upsert snapshots without duplicates.
    Uses slug|market_id|snapshot_time_utc (stringified).
    """
    slug = str(row.get("slug", "") or "")
    market_id = str(row.get("market_id", "") or "")
    t = str(row.get("snapshot_time_utc", "") or "")
    raw = f"{slug}|{market_id}|{t}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def add_weekday_labels(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    """
    Add local-weekday labels for easier retrospective analysis.
    Creates:
      - snapshot_time_local
      - snapshot_date_local
      - snapshot_weekday_local (Mon/Tue/...)
      - snapshot_weekday_num (Mon=0 ... Sun=6)
    """
    out = df.copy()
    out["snapshot_time_utc"] = to_utc_dt(out["snapshot_time_utc"])

    if ZoneInfo is None:
        # fallback: keep UTC as "local"
        out["snapshot_time_local"] = out["snapshot_time_utc"]
    else:
        tz = ZoneInfo(tz_name)
        out["snapshot_time_local"] = out["snapshot_time_utc"].dt.tz_convert(tz)

    out["snapshot_date_local"] = out["snapshot_time_local"].dt.date.astype(str)
    out["snapshot_weekday_local"] = out["snapshot_time_local"].dt.day_name().str.slice(0, 3)
    out["snapshot_weekday_num"] = out["snapshot_time_local"].dt.weekday
    return out


def human_readable_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the combined file very readable and stable.
    We keep important identifiers and fields first, then everything else.
    """
    preferred = [
        # snapshot timing
        "snapshot_time_utc",
        "snapshot_time_local",
        "snapshot_date_local",
        "snapshot_weekday_local",
        "snapshot_weekday_num",

        # week cluster (if present)
        "week_monday",
        "week_friday",
        "week_sunday",

        # identity / traceability
        "ticker",
        "slug",
        "event_id",
        "market_id",
        "condition_id",
        "yes_token_id",
        "no_token_id",
        "event_endDate",
        "event_title",
        "market_question",

        # key contract params
        "K",
        "S",
        "σ",
        "T_years",
        "r",

        # prices/probs
        "pPM",
        "qPM",
        "pRN",
        "qRN",

        # minimal indicators (per your request)
        "edgeYES",
        "edgeNO",
        "ΔP",

        # outcomes (what we are adding/updating)
        "is_resolved",
        "winner",
        "y",
        "yes_price_final",
        "no_price_final",
        "outcome_collected_at_utc",
        "resolved_by",
        "uma_resolution_status",
    ]

    cols = [c for c in preferred if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras].copy()


# -----------------------
# Outcome collection for a set of markets
# -----------------------
def determine_needed_outcome_pairs(combined: pd.DataFrame, snapshots: pd.DataFrame):
    """
    Return list of (slug, market_id) pairs that need outcome fetching.
    Rules:
      - If combined lacks outcome columns entirely -> fetch for ALL pairs in snapshots
      - Else fetch for pairs where:
          * outcome_collected_at_utc is missing OR
          * is_resolved is not True OR
          * y is missing
    """
    def norm_pairs(df: pd.DataFrame) -> set:
        if df is None or df.empty:
            return set()
        x = df[["slug", "market_id"]].dropna().copy()
        x["slug"] = x["slug"].astype(str)
        x["market_id"] = x["market_id"].astype(str)
        return set(zip(x["slug"], x["market_id"]))

    snap_pairs = norm_pairs(snapshots)
    if not snap_pairs:
        return []

    # If combined is empty -> all snapshot pairs need fetch
    if combined is None or combined.empty:
        return sorted(snap_pairs)

    # If outcome columns don't exist yet -> fetch all
    outcome_cols = {"outcome_collected_at_utc", "is_resolved", "y"}
    if not outcome_cols.intersection(set(combined.columns)):
        return sorted(snap_pairs)

    tmp = combined.copy()
    tmp["slug"] = tmp["slug"].astype(str)
    tmp["market_id"] = tmp["market_id"].astype(str)

    # Missing outcome timestamp => need fetch
    need = pd.Series(False, index=tmp.index)
    if "outcome_collected_at_utc" in tmp.columns:
        need |= pd.to_datetime(tmp["outcome_collected_at_utc"], utc=True, errors="coerce").isna()
    else:
        need |= True

    # Not resolved => need fetch
    if "is_resolved" in tmp.columns:
        need |= ~(tmp["is_resolved"].fillna(False).astype(bool))
    else:
        need |= True

    # Missing y => need fetch
    if "y" in tmp.columns:
        need |= pd.to_numeric(tmp["y"], errors="coerce").isna()
    else:
        need |= True

    unresolved = tmp.loc[need, ["slug", "market_id"]].dropna()
    unresolved_pairs = set(zip(unresolved["slug"], unresolved["market_id"]))

    # Only fetch for pairs that actually exist in snapshots
    return sorted(unresolved_pairs.intersection(snap_pairs))



def filter_pairs_by_expiry(
    pairs: List[Tuple[str, str]],
    snapshots: pd.DataFrame,
    *,
    only_expired: bool,
    expiry_grace_hours: int
) -> List[Tuple[str, str]]:
    """
    Optional: only fetch outcomes for markets whose event_endDate is safely in the past.
    Uses max event_endDate per (slug, market_id) if available, else per slug.
    """
    if not only_expired:
        return pairs

    now_utc = datetime.now(timezone.utc)
    grace = timedelta(hours=int(expiry_grace_hours))

    df = snapshots.copy()
    if df.empty:
        return pairs

    df["event_endDate"] = to_utc_dt(df.get("event_endDate"))
    df["slug"] = df.get("slug").astype(str)
    df["market_id"] = df.get("market_id").astype(str)

    # max endDate per pair
    pair_end = (
        df[["slug", "market_id", "event_endDate"]]
        .dropna()
        .groupby(["slug", "market_id"])["event_endDate"]
        .max()
        .to_dict()
    )

    out = []
    for slug, market_id in pairs:
        end_dt = pair_end.get((slug, market_id))
        if end_dt is None or pd.isna(end_dt):
            # if no end date, we can't safely skip; include
            out.append((slug, market_id))
            continue
        if end_dt.to_pydatetime() <= (now_utc - grace):
            out.append((slug, market_id))
    return out


def fetch_outcomes_for_pairs(
    gamma: GammaClient,
    pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Fetch outcomes by slug (Gamma event endpoint), then pick the needed market_ids.
    Returns one row per (slug, market_id) fetched with outcome fields.
    """
    if not pairs:
        return pd.DataFrame()

    # group requested market_ids by slug to minimize API calls
    by_slug: Dict[str, set] = {}
    for slug, mid in pairs:
        by_slug.setdefault(slug, set()).add(mid)

    rows = []
    collected_at = utc_now_iso()
    slugs = sorted(by_slug.keys())

    for i, slug in enumerate(slugs, start=1):
        want_mids = by_slug[slug]

        try:
            event = gamma.fetch_event_by_slug(slug)
        except Exception as e:
            print(f"❌ [{i}/{len(slugs)}] fetch failed slug={slug}: {e}")
            time.sleep(gamma.sleep_sec)
            continue

        event_id = event.get("id")
        event_title = event.get("title") or event.get("question")
        event_end = event.get("endDate") or event.get("endDateIso")
        event_closed = event.get("closed")
        event_active = event.get("active")

        markets = event.get("markets", []) or []
        if not isinstance(markets, list):
            markets = []

        found_any = False

        for m in markets:
            market_id = str(m.get("id")) if m.get("id") is not None else None
            if not market_id or market_id not in want_mids:
                continue

            found_any = True

            question = m.get("question") or m.get("title")
            condition_id = m.get("conditionId")

            # prefer market endDate
            expiry = m.get("endDate") or m.get("endDateIso") or event_end
            K = extract_strike_K_from_question(question)

            outcomes = safe_json_load(m.get("outcomes"))
            outcome_prices = safe_json_load(m.get("outcomePrices"))

            is_resolved, winner, y, yes_final, no_final = infer_resolution_from_prices(outcomes, outcome_prices)

            rows.append({
                "outcome_collected_at_utc": collected_at,

                "slug": slug,
                "event_id": event_id,
                "event_title": event_title,
                "event_endDate": event_end,

                "market_id": market_id,
                "condition_id": condition_id,
                "market_question": question,
                "K": K,
                "expiry": expiry,

                # status fields useful for debugging
                "event_closed": event_closed,
                "event_active": event_active,
                "market_closed": m.get("closed"),
                "market_active": m.get("active"),
                "resolved_by": m.get("resolvedBy"),
                "uma_resolution_status": m.get("umaResolutionStatus"),

                # labels
                "is_resolved": bool(is_resolved),
                "winner": winner,  # YES/NO
                "y": y,            # 1/0
                "yes_price_final": yes_final,
                "no_price_final": no_final,
            })

        if not found_any:
            print(f"⚠️ [{i}/{len(slugs)}] slug={slug} had none of the requested market_ids")
        else:
            print(f"✅ [{i}/{len(slugs)}] slug={slug} fetched")

        time.sleep(gamma.sleep_sec)

    return pd.DataFrame(rows)


# -----------------------
# Upsert combined file
# -----------------------
def upsert_snapshots_into_combined(combined: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Upsert snapshots by snapshot_id. Keep existing rows; append new ones.
    """
    snap = snapshots.copy()
    if snap.empty:
        return combined

    # enforce required columns
    for col in ["slug", "market_id", "snapshot_time_utc"]:
        if col not in snap.columns:
            snap[col] = pd.NA

    # Create snapshot_id
    snap["snapshot_id"] = snap.apply(stable_snapshot_id, axis=1)

    if combined is None or combined.empty:
        out = snap
    else:
        out = combined.copy()
        if "snapshot_id" not in out.columns:
            # backfill ids for old combined
            out["snapshot_id"] = out.apply(stable_snapshot_id, axis=1)

        existing_ids = set(out["snapshot_id"].astype(str))
        new_rows = snap[~snap["snapshot_id"].astype(str).isin(existing_ids)].copy()
        out = pd.concat([out, new_rows], ignore_index=True)

    return out


def apply_outcomes_update(combined: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """
    Update outcome fields in combined for all rows matching (slug, market_id).
    This overwrites older/unknown outcome values with the newly fetched ones.
    """
    if combined.empty or outcomes.empty:
        return combined

    out = combined.copy()
    out["slug"] = out["slug"].astype(str)
    out["market_id"] = out["market_id"].astype(str)

    upd = outcomes.copy()
    upd["slug"] = upd["slug"].astype(str)
    upd["market_id"] = upd["market_id"].astype(str)

    # Build a mapping per pair to latest outcome row
    key_cols = ["slug", "market_id"]
    # If multiple entries for a pair, keep the latest collected time
    if "outcome_collected_at_utc" in upd.columns:
        upd["outcome_collected_at_utc"] = to_utc_dt(upd["outcome_collected_at_utc"])
        upd = upd.sort_values("outcome_collected_at_utc").drop_duplicates(key_cols, keep="last")

    upd = upd.set_index(key_cols)

    # columns to update in combined
    update_cols = [
        "outcome_collected_at_utc",
        "is_resolved",
        "winner",
        "y",
        "yes_price_final",
        "no_price_final",
        "resolved_by",
        "uma_resolution_status",
    ]
    update_cols = [c for c in update_cols if c in upd.columns]

    # Ensure combined has these columns
    for c in update_cols:
        if c not in out.columns:
            out[c] = pd.NA

    # Vectorized update via merge
    merged = out.merge(
        upd[update_cols].reset_index(),
        on=["slug", "market_id"],
        how="left",
        suffixes=("", "_new"),
    )

    for c in update_cols:
        newc = f"{c}_new"
        if newc in merged.columns:
            # overwrite only if new value is not null
            merged[c] = merged[newc].combine_first(merged[c])
            merged.drop(columns=[newc], inplace=True)

    return merged


# -----------------------
# CLI / Main
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Build ONE combined snapshots+outcomes CSV; updates existing rows when outcomes resolve."
    )
    ap.add_argument("--snapshot-history", required=True, help="Path to your snapshot history CSV (append-only).")
    ap.add_argument("--combined-out", required=True, help="Path to the ONE combined output CSV (will be overwritten).")

    ap.add_argument("--gamma-base", default=DEFAULT_GAMMA_BASE)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)

    ap.add_argument("--tz", default="Europe/Paris", help="Timezone for weekday labeling.")
    ap.add_argument("--only-expired", action="store_true", help="Only fetch outcomes for markets past expiry.")
    ap.add_argument("--expiry-grace-hours", type=int, default=6, help="Wait this many hours after expiry before fetching.")

    ap.add_argument("--max-slugs", type=int, default=0, help="Debug: cap number of slugs fetched (0 = no cap).")
    return ap.parse_args()


def main():
    args = parse_args()

    if ZoneInfo is None:
        raise RuntimeError("Python 3.9+ required (zoneinfo).")

    # Load snapshots history
    snap = pd.read_csv(args.snapshot_history)
    if snap.empty:
        print("⚠️ Snapshot history is empty. Nothing to do.")
        return

    # Create weekday labels (human-friendly)
    if "snapshot_time_utc" not in snap.columns:
        raise ValueError("snapshot history must contain 'snapshot_time_utc'")
    snap = add_weekday_labels(snap, args.tz)

    # Load existing combined file (if exists)
    if os.path.exists(args.combined_out) and os.path.getsize(args.combined_out) > 0:
        combined = pd.read_csv(args.combined_out)
        # parse datetime columns if present
        if "snapshot_time_utc" in combined.columns:
            combined["snapshot_time_utc"] = to_utc_dt(combined["snapshot_time_utc"])
        if "outcome_collected_at_utc" in combined.columns:
            combined["outcome_collected_at_utc"] = to_utc_dt(combined["outcome_collected_at_utc"])
    else:
        combined = pd.DataFrame()

    # Upsert snapshots into combined (no duplicates)
    combined = upsert_snapshots_into_combined(combined, snap)

    # Decide what outcomes we need to fetch
    pairs_needed = determine_needed_outcome_pairs(combined, snap)
    pairs_needed = filter_pairs_by_expiry(
        pairs_needed,
        snap,
        only_expired=bool(args.only_expired),
        expiry_grace_hours=int(args.expiry_grace_hours),
    )

    # Optional debug cap by slug
    if args.max_slugs and args.max_slugs > 0 and pairs_needed:
        # cap unique slugs
        seen = set()
        capped = []
        for slug, mid in pairs_needed:
            if slug not in seen and len(seen) >= args.max_slugs:
                continue
            seen.add(slug)
            capped.append((slug, mid))
        pairs_needed = capped

    print(f"[Need] outcomes pairs: {len(pairs_needed)}")

    # Fetch outcomes
    gamma = GammaClient(base=args.gamma_base, timeout=args.timeout, sleep_sec=args.sleep_sec)
    outcomes = fetch_outcomes_for_pairs(gamma, pairs_needed)

    if outcomes.empty:
        print("[Outcomes] No new outcomes fetched (maybe nothing resolved yet).")
    else:
        resolved_ct = int(outcomes["is_resolved"].fillna(False).astype(bool).sum()) if "is_resolved" in outcomes.columns else 0
        print(f"[Outcomes] fetched rows={len(outcomes)}  resolved={resolved_ct}")

    # Apply updates into combined
    combined = apply_outcomes_update(combined, outcomes)

    # Make it readable
    combined = human_readable_order(combined)

    # Write ONE combined CSV (overwrite)
    ensure_dir_for_file(args.combined_out)
    combined.to_csv(args.combined_out, index=False)
    print(f"[Write] {args.combined_out} (rows={len(combined)})")


if __name__ == "__main__":
    main()
