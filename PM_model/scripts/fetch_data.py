from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm
from zoneinfo import ZoneInfo


# -----------------------
# Defaults / Config
# -----------------------

DEFAULT_TICKERS = [
    "NVDA", "AAPL", "GOOGL", "MSFT", "META", "AMZN", "TSLA", "PLTR", "OPEN", "NFLX"
]

GAMMA_EVENT_BY_SLUG = "https://gamma-api.polymarket.com/events/slug/{}"
CLOB_MIDPOINT = "https://clob.polymarket.com/midpoint"  # query param: token_id


@dataclass(frozen=True)
class Config:
    tz_name: str = "Europe/Paris"
    risk_free_rate: float = 0.03

    sleep_between_slugs_s: float = 0.25
    sleep_between_price_calls_s: float = 0.05

    request_timeout_s: int = 30


# -----------------------
# Trading-week logic (ISO week Mon–Sun; closes on Friday EOD)
# -----------------------

def trading_week_bounds(today_local: date) -> Tuple[date, date, date]:
    """
    Returns (monday, friday, sunday) for the ISO week containing today_local.

    This matches your example:
      If today is within Mon 12 Jan 2026 ... Sun 18 Jan 2026,
      then friday is 16 Jan 2026 (week closes on that Friday).
    """
    weekday = today_local.weekday()  # Mon=0 ... Sun=6
    monday = today_local - timedelta(days=weekday)
    friday = monday + timedelta(days=4)
    sunday = monday + timedelta(days=6)
    return monday, friday, sunday


# -----------------------
# Ticker selection + slug mapping
# -----------------------

def parse_tickers_arg(tickers_csv: Optional[str], tickers_list: Optional[str]) -> List[str]:
    """
    Priority:
      1) --tickers "NVDA,AAPL"
      2) --tickers-csv file with a column named 'ticker'
      3) DEFAULT_TICKERS
    """
    if tickers_list:
        tickers = [t.strip().upper() for t in tickers_list.split(",") if t.strip()]
        if not tickers:
            raise ValueError("Provided --tickers is empty after parsing.")
        return tickers

    if tickers_csv:
        df = pd.read_csv(tickers_csv)
        if "ticker" not in df.columns:
            raise ValueError(f"--tickers-csv must contain a 'ticker' column. Found: {list(df.columns)}")
        tickers = (
            df["ticker"].dropna().astype(str).str.strip().str.upper().tolist()
        )
        tickers = [t for t in tickers if t]
        if not tickers:
            raise ValueError("No tickers found in --tickers-csv.")
        return tickers

    return DEFAULT_TICKERS.copy()


def load_slug_overrides(path: Optional[str]) -> Dict[str, str]:
    """
    Optional override mapping from ticker->slug (or ticker->slug_prefix).
    Accepts:
      - JSON: {"NVDA": "nvda-above-on-january-16-2026", ...}
      - CSV with columns: ticker, slug
    """
    if not path:
        return {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"Slug overrides file not found: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Slug overrides JSON must be an object/dict of ticker->slug.")
        return {str(k).upper(): str(v) for k, v in data.items()}

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if not {"ticker", "slug"}.issubset(df.columns):
            raise ValueError(f"Slug overrides CSV must contain columns ticker, slug. Found: {list(df.columns)}")
        out = {}
        for _, r in df.iterrows():
            t = str(r["ticker"]).strip().upper()
            s = str(r["slug"]).strip()
            if t and s:
                out[t] = s
        return out

    raise ValueError("Slug overrides file must be .json or .csv")


def ticker_to_slug_prefix(ticker: str) -> str:
    """
    Best-effort conversion. You can override per-ticker via --slug-overrides.
    """
    t = ticker.strip().lower()
    # Common cleanups for US equity tickers
    t = t.replace(".", "").replace("/", "-").replace(" ", "")
    return t


def build_weekly_slug(ticker: str, week_friday: date) -> str:
    """
    Matches the pattern you used, e.g.:
      nvda-above-on-january-16-2026
    """
    prefix = ticker_to_slug_prefix(ticker)
    month = week_friday.strftime("%B").lower()
    return f"{prefix}-above-on-{month}-{week_friday.day}-{week_friday.year}"


def build_targets_df(
    tickers: List[str],
    week_monday: date,
    week_friday: date,
    week_sunday: date,
    slug_overrides: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        if t in slug_overrides:
            slug = slug_overrides[t]
        else:
            slug = build_weekly_slug(t, week_friday)

        rows.append(
            {
                "ticker": t,
                "slug": slug,
                "week_monday": week_monday.isoformat(),
                "week_friday": week_friday.isoformat(),
                "week_sunday": week_sunday.isoformat(),
            }
        )
    return pd.DataFrame(rows)

def ensure_columns(df: pd.DataFrame, ordered_cols: list[str]) -> pd.DataFrame:
    """
    Ensure df has all columns in ordered_cols (create missing as NA),
    then return df with columns in that order + any extras at the end.
    """
    out = df.copy()
    for c in ordered_cols:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in ordered_cols]
    return out[ordered_cols + extras]


def append_df_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Append df to CSV at path. Creates file if missing. Writes header only once.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    df.to_csv(path, mode="a", header=not file_exists, index=False)

# -----------------------
# Polymarket fetch (Gamma + CLOB midpoint)
# -----------------------

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


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


def extract_strike_K_from_question(question: str) -> Optional[float]:
    """
    Extract strike/threshold K from market question.
    Examples:
      '... above $205?' -> 205.0
      '... above $205.5?' -> 205.5
    """
    if not isinstance(question, str):
        return None
    m = re.search(r"\$(\d+(?:\.\d+)?)", question)
    return float(m.group(1)) if m else None


def get_midpoint_price(token_id: str, cfg: Config) -> Optional[float]:
    if not token_id:
        return None
    try:
        r = requests.get(
            CLOB_MIDPOINT,
            params={"token_id": token_id},
            timeout=cfg.request_timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        return safe_float(data.get("mid"))
    except Exception:
        return None


def fetch_polymarket_snapshots(targets_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    For each slug:
      - fetch Gamma event by slug
      - iterate markets inside event
      - fetch YES/NO token midpoint prices (CLOB), fallback to Gamma outcomePrices
    Outputs a per-market-row snapshot dataset.
    """
    if "slug" not in targets_df.columns or "ticker" not in targets_df.columns:
        raise ValueError("targets_df must contain columns: ticker, slug")

    slugs = (
        targets_df["slug"].dropna().astype(str).str.strip().tolist()
    )
    slugs = [s for s in slugs if s]
    if not slugs:
        raise ValueError("No slugs to fetch.")

    snapshot_time_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    errors = []

    for slug in slugs:
        try:
            resp = requests.get(
                GAMMA_EVENT_BY_SLUG.format(slug),
                timeout=cfg.request_timeout_s,
            )
            resp.raise_for_status()
            event = resp.json()

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
                K = extract_strike_K_from_question(question)  # K in cheatsheet notation

                token_ids = normalize_list_field(m.get("clobTokenIds"))
                yes_token_id = token_ids[0] if isinstance(token_ids, list) and len(token_ids) >= 2 else None
                no_token_id  = token_ids[1] if isinstance(token_ids, list) and len(token_ids) >= 2 else None

                # CLOB midpoint
                p_yes_mid = get_midpoint_price(yes_token_id, cfg)
                time.sleep(cfg.sleep_between_price_calls_s)
                p_no_mid = get_midpoint_price(no_token_id, cfg)
                time.sleep(cfg.sleep_between_price_calls_s)

                # Gamma fallback
                outcome_prices = normalize_list_field(m.get("outcomePrices"))
                gamma_yes = gamma_no = None
                if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                    gamma_yes = safe_float(outcome_prices[0])
                    gamma_no = safe_float(outcome_prices[1])

                pPM = p_yes_mid if p_yes_mid is not None else gamma_yes  # cheatsheet: pPM
                qPM = p_no_mid if p_no_mid is not None else gamma_no    # cheatsheet: qPM

                rows.append(
                    {
                        "snapshot_time_utc": snapshot_time_utc,
                        "slug": slug,
                        "event_id": event_id,
                        "event_title": event_title,
                        "event_updatedAt": updated_at,
                        "event_endDate": end_date,
                        "market_id": market_id,
                        "condition_id": condition_id,
                        "market_question": question,
                        "K": K,
                        "yes_token_id": yes_token_id,
                        "no_token_id": no_token_id,
                        "pPM": pPM,
                        "qPM": qPM,
                        "pPM_source": "clob_midpoint" if p_yes_mid is not None else ("gamma_outcomePrices" if gamma_yes is not None else None),
                        "qPM_source": "clob_midpoint" if p_no_mid is not None else ("gamma_outcomePrices" if gamma_no is not None else None),
                    }
                )

        except Exception as e:
            errors.append({"slug": slug, "error": str(e)})

        time.sleep(cfg.sleep_between_slugs_s)

    df = pd.DataFrame(rows)

    # Parse datetimes and compute T
    if not df.empty:
        df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
        df["event_endDate"] = pd.to_datetime(df["event_endDate"], utc=True, errors="coerce")

        df["T_days"] = (df["event_endDate"] - df["snapshot_time_utc"]).dt.total_seconds() / (60 * 60 * 24)
        df["T_years"] = df["T_days"] / 365.25

    # Optional: print a small error summary
    if errors:
        print(f"[Polymarket] Errors: {len(errors)} (showing up to 5)")
        print(pd.DataFrame(errors).head())

    return df


# -----------------------
# Options / risk-neutral probability pRN (Black–Scholes Φ(d2))
# -----------------------

def compute_d2(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    if S is None or K is None or T is None or sigma is None:
        return None
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    return (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def risk_neutral_prob_pRN(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    d2 = compute_d2(S, K, T, r, sigma)
    return float(norm.cdf(d2)) if d2 is not None else None


def round7(x) -> Optional[float]:
    return round(float(x), 7) if x is not None else None


def compute_rn_probabilities(
    snapshots_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    For each (ticker, expiry) group:
      - fetch yfinance option chain for expiry date
      - fetch spot close S
      - for each market row, choose nearest call strike to K and use its IV as sigma
      - compute pRN = Φ(d2)
    """
    if snapshots_df.empty:
        return pd.DataFrame()

    # Join ticker onto snapshots (traceable join key: slug)
    df = snapshots_df.merge(
        targets_df[["slug", "ticker"]],
        on="slug",
        how="left",
        validate="m:1",
    )
    df = df[df["ticker"].notna()].copy()

    # Ensure time fields
    df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    df["event_endDate"] = pd.to_datetime(df["event_endDate"], utc=True, errors="coerce")

    # Keep only valid T
    df = df[df["T_years"].notna() & (df["T_years"] > 0)].copy()
    df = df[df["K"].notna() & (df["K"] > 0)].copy()

    rows = []

    grouped = df.groupby(["ticker", "event_endDate"], dropna=True)
    print(f"[yfinance] Processing {len(grouped)} (ticker, expiry) groups")

    for (ticker, expiry_dt), g in grouped:
        try:
            stock = yf.Ticker(ticker)
            expiry_str = pd.Timestamp(expiry_dt).strftime("%Y-%m-%d")

            # Ensure expiry exists in yfinance
            if expiry_str not in getattr(stock, "options", []):
                print(f"[yfinance] ⚠️ No options for {ticker} expiry {expiry_str}")
                continue

            chain = stock.option_chain(expiry_str)
            calls = chain.calls.copy()
            if calls.empty:
                print(f"[yfinance] ⚠️ Empty calls for {ticker} expiry {expiry_str}")
                continue

            # Spot close S (cheatsheet: S)
            hist = stock.history(period="1d")
            if hist.empty:
                print(f"[yfinance] ⚠️ No history for {ticker}")
                continue
            S = float(hist["Close"].iloc[-1])

            # For speed: only keep needed columns
            calls = calls[["strike", "impliedVolatility"]].dropna().copy()

            for _, r in g.iterrows():
                K = float(r["K"])
                T = float(r["T_years"])

                calls["strike_diff"] = (calls["strike"] - K).abs()
                opt = calls.sort_values("strike_diff").iloc[0]

                K_opt = float(opt["strike"])
                sigma = safe_float(opt["impliedVolatility"])  # cheatsheet: σ (IV)

                pRN = risk_neutral_prob_pRN(
                    S=S,
                    K=K,
                    T=T,
                    r=cfg.risk_free_rate,
                    sigma=sigma,
                )

                rows.append(
                    {
                        "snapshot_time_utc": r["snapshot_time_utc"],
                        "slug": r["slug"],
                        "ticker": ticker,
                        "expiry": expiry_dt,   # for traceability
                        "S": round7(S),
                        "K": round7(K),
                        "K_opt": round7(K_opt),
                        "σ": round7(sigma),    # Unicode sigma to match cheatsheet notation
                        "T_years": round7(T),
                        "pRN": round7(pRN),    # cheatsheet: pRN
                        "r": cfg.risk_free_rate,
                        "iv_source": "nearest_call_iv",
                        "spot_source": "yfinance_history_close_1d",
                    }
                )

        except Exception as e:
            print(f"[yfinance] ❌ Error processing {ticker} {expiry_dt}: {e}")

    return pd.DataFrame(rows)


# -----------------------
# Main (PART 1)
# -----------------------

def main_part1() -> None:
    parser = argparse.ArgumentParser(description="Unified Polymarket + yfinance pipeline (PART 1/2)")
    parser.add_argument("--out-dir", type=str, default="./data", help="Output directory for CSVs")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers, e.g. NVDA,AAPL")
    parser.add_argument("--tickers-csv", type=str, default=None, help="CSV with a 'ticker' column")
    parser.add_argument("--slug-overrides", type=str, default=None, help="Optional .json or .csv mapping ticker->slug")
    parser.add_argument("--risk-free-rate", type=float, default=Config().risk_free_rate, help="Flat risk-free rate r")
    parser.add_argument("--tz", type=str, default=Config().tz_name, help="Timezone for trading-week logic, default Europe/Paris")
    args = parser.parse_args()

    cfg = Config(
        tz_name=args.tz,
        risk_free_rate=float(args.risk_free_rate),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Determine trading week ----
    tz = ZoneInfo(cfg.tz_name)
    today_local = datetime.now(tz).date()
    week_monday, week_friday, week_sunday = trading_week_bounds(today_local)

    print(f"[Week] Today (local): {today_local.isoformat()}  |  "
          f"Week: {week_monday.isoformat()} → {week_sunday.isoformat()}  |  "
          f"Closes Friday: {week_friday.isoformat()}")

    # ---- Select tickers ----
    tickers = parse_tickers_arg(args.tickers_csv, args.tickers)
    print(f"[Tickers] Selected {len(tickers)} tickers: {', '.join(tickers)}")

    # ---- Slug mapping ----
    slug_overrides = load_slug_overrides(args.slug_overrides)

    targets_df = build_targets_df(
        tickers=tickers,
        week_monday=week_monday,
        week_friday=week_friday,
        week_sunday=week_sunday,
        slug_overrides=slug_overrides,
    )

    targets_path = os.path.join(args.out_dir, "polymarket_target_markets.csv")
    targets_df.to_csv(targets_path, index=False)
    print(f"[Write] {targets_path}")

    # ---- Fetch Polymarket snapshots ----
    pm_df = fetch_polymarket_snapshots(targets_df, cfg)
    pm_path = os.path.join(args.out_dir, "polymarket_price_snapshots.csv")
    pm_df.to_csv(pm_path, index=False)
    print(f"[Write] {pm_path}  (rows={len(pm_df)})")

    # ---- Compute RN probabilities from options ----
    rn_df = compute_rn_probabilities(pm_df, targets_df, cfg)
    rn_path = os.path.join(args.out_dir, "options_risk_neutral_probabilities.csv")
    rn_df.to_csv(rn_path, index=False)
    print(f"[Write] {rn_path}  (rows={len(rn_df)})")

    # PART 2 will start here: merge + clean + final single CSV output.
    print("[Next] Append PART 2 to merge/clean and output final dataset CSV.")


# =========================
# PART 2/2 — Merge, clean, signal engineering, final CSV
# =========================

def _coalesce_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """
    For each canonical column name, take the first existing column from its aliases.
    """
    out = df.copy()
    for canon, aliases in mapping.items():
        if canon in out.columns:
            continue
        for a in aliases:
            if a in out.columns:
                out[canon] = out[a]
                break
    return out


def standardize_polymarket_df(pm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Polymarket snapshot column names to cheatsheet notation:
      - pPM (YES mid) and qPM (NO mid)
      - K (threshold/strike)
    """
    df = pm_df.copy()

    df = _coalesce_columns(
        df,
        mapping={
            "pPM": ["yes_price", "yes_mid", "yes_midpoint", "p_yes_mid", "p_yes"],
            "qPM": ["no_price", "no_mid", "no_midpoint", "p_no_mid", "p_no"],
            "K": ["strike", "threshold", "K"],
            "pPM_source": ["yes_price_source", "pPM_source"],
            "qPM_source": ["no_price_source", "qPM_source"],
        },
    )

    # Ensure timestamps
    df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    df["event_endDate"] = pd.to_datetime(df["event_endDate"], utc=True, errors="coerce")

    # Numeric
    for c in ["pPM", "qPM", "K", "T_days", "T_years"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute T if missing
    if "T_days" not in df.columns or df["T_days"].isna().all():
        df["T_days"] = (df["event_endDate"] - df["snapshot_time_utc"]).dt.total_seconds() / (60 * 60 * 24)
    if "T_years" not in df.columns or df["T_years"].isna().all():
        df["T_years"] = df["T_days"] / 365.25

    return df


def standardize_rn_df(rn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize options/RN dataset to cheatsheet notation:
      - pRN, σ, S, K, T_years
    Also robust to:
      - snapshot_time_utc stored as index
      - CSV read without headers (RangeIndex columns)
      - alternate timestamp column names
    """
    df = rn_df.copy()

    # If timestamp was saved as index, restore it
    if df.index.name in ("snapshot_time_utc", "snapshot_time", "timestamp", "datetime"):
        df = df.reset_index()

    # If user loaded CSV with header=None, columns become RangeIndex(0..n-1)
    # Try to assign known schemas by column count.
    if isinstance(df.columns, pd.RangeIndex):
        if df.shape[1] == 9:
            # Matches your original fetch_yf_data.py output order
            df.columns = [
                "snapshot_time_utc", "slug", "ticker", "expiry",
                "strike", "spot_price", "implied_volatility",
                "T_years", "risk_neutral_prob"
            ]
        elif df.shape[1] == 13:
            # Matches the schema produced by the integrated Part 1 compute_rn_probabilities
            df.columns = [
                "snapshot_time_utc", "slug", "ticker", "expiry",
                "S", "K", "K_opt", "σ", "T_years", "pRN",
                "r", "iv_source", "spot_source"
            ]
        # else: leave as-is; we'll still try alias detection below.

    # Coalesce aliases -> canonical names
    df = _coalesce_columns(
        df,
        mapping={
            # Timestamp aliases
            "snapshot_time_utc": ["snapshot_time_utc", "snapshot_time", "timestamp", "datetime", "as_of"],
            # Probability / model fields
            "pRN": ["pRN", "risk_neutral_prob", "risk_neutral_probability", "rn_prob", "risk_neutral_prob_raw"],
            "σ": ["σ", "implied_volatility", "impliedVolatility", "iv", "sigma"],
            "S": ["S", "spot_price", "spot", "underlying_price"],
            "K": ["K", "strike", "threshold"],
            "K_opt": ["K_opt", "strike_opt", "nearest_strike"],
        },
    )

    # If timestamp still missing, create it (so we don't crash).
    # Merge will fall back to key-only join (see patched merge_pm_with_rn below).
    if "snapshot_time_utc" not in df.columns:
        df["snapshot_time_utc"] = pd.NaT

    # Parse datetimes
    df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], utc=True, errors="coerce")

    # Numeric
    for c in ["pRN", "σ", "S", "K", "K_opt", "T_years", "r"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df



def merge_pm_with_rn(
    pm_df: pd.DataFrame,
    rn_df: pd.DataFrame,
    *,
    tolerance_minutes: int = 15,
) -> pd.DataFrame:
    """
    Merge Polymarket snapshots with RN rows.
    If rn_df is empty or missing expected columns, return pm_df with RN columns as NaN,
    so the pipeline still writes a final CSV.
    """
    left = pm_df.copy()

    # If RN is empty OR has no columns, don't crash
    if rn_df is None or rn_df.empty or len(rn_df.columns) == 0:
        # Add placeholders expected downstream
        for c in ["pRN", "σ", "S", "K_opt", "r", "iv_source", "spot_source", "snapshot_time_utc_rn"]:
            if c not in left.columns:
                left[c] = pd.NA
        left["merge_time_diff_s"] = pd.NA
        left["rn_missing_flag"] = True
        return left

    right = rn_df.copy()

    # Ensure the required columns exist even if partially missing
    for col in ["slug", "K", "snapshot_time_utc"]:
        if col not in right.columns:
            right[col] = pd.NA

    left["K_round"] = pd.to_numeric(left.get("K"), errors="coerce").round(4)
    right["K_round"] = pd.to_numeric(right.get("K"), errors="coerce").round(4)

    left["contract_key"] = left["slug"].astype(str) + "|" + left["K_round"].astype(str)
    right["contract_key"] = right["slug"].astype(str) + "|" + right["K_round"].astype(str)

    # Preserve RN timestamp for diagnostics
    right = right.rename(columns={"snapshot_time_utc": "snapshot_time_utc_rn"})

    left["snapshot_time_utc"] = pd.to_datetime(left["snapshot_time_utc"], utc=True, errors="coerce")
    right["snapshot_time_utc_rn"] = pd.to_datetime(right["snapshot_time_utc_rn"], utc=True, errors="coerce")

    has_rn_time = right["snapshot_time_utc_rn"].notna().any()

    if has_rn_time:
        left = left.sort_values(["contract_key", "snapshot_time_utc"])
        right = right.sort_values(["contract_key", "snapshot_time_utc_rn"])

        merged = pd.merge_asof(
            left,
            right,
            left_on="snapshot_time_utc",
            right_on="snapshot_time_utc_rn",
            by="contract_key",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
            direction="nearest",
            suffixes=("", "_rn"),
        )
        merged["merge_time_diff_s"] = (
            (merged["snapshot_time_utc"] - merged["snapshot_time_utc_rn"])
            .abs()
            .dt.total_seconds()
        )
        merged["rn_missing_flag"] = merged["pRN"].isna() if "pRN" in merged.columns else True
        return merged

    # fallback key-only merge
    right_one = right.sort_values("contract_key").drop_duplicates("contract_key", keep="last")
    merged = left.merge(right_one, on="contract_key", how="left", suffixes=("", "_rn"))
    merged["merge_time_diff_s"] = pd.NA
    merged["rn_missing_flag"] = merged["pRN"].isna() if "pRN" in merged.columns else True
    return merged



def add_signals_and_clean(
    df: pd.DataFrame,
    *,
    pm_sum_min: float = 0.98,
    pm_sum_max: float = 1.02,
    filter_invalid: bool = False,
) -> pd.DataFrame:
    """
    Minimal indicators ONLY:
      - edgeYES = pRN - pPM
      - edgeNO  = (1 - pRN) - qPM
      - ΔP      = max(edgeYES, edgeNO)

    No ΔPnorm, no p-hat, no EV, no clustering, no action labels.
    """
    out = df.copy()
    if out is None:
        return pd.DataFrame()
    if out.empty:
        return out

    # Ensure numeric where needed (safe with <NA>)
    for c in ["pPM", "qPM", "pRN"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Basic PM sanity (optional)
    if "pPM" in out.columns and "qPM" in out.columns:
        out["pm_sum"] = out["pPM"] + out["qPM"]
        out["pm_sum_ok"] = out["pm_sum"].between(pm_sum_min, pm_sum_max, inclusive="both")
        if filter_invalid:
            out = out[out["pm_sum_ok"]].copy()

    # RN complement if available
    if "pRN" in out.columns:
        out["qRN"] = 1.0 - out["pRN"]
    else:
        out["qRN"] = pd.NA

    # Edges only if RN exists; otherwise keep as NA
    if "pRN" in out.columns and "pPM" in out.columns:
        out["edgeYES"] = out["pRN"] - out["pPM"]
    else:
        out["edgeYES"] = pd.NA

    if "pRN" in out.columns and "qPM" in out.columns:
        out["edgeNO"] = (1.0 - out["pRN"]) - out["qPM"]
    else:
        out["edgeNO"] = pd.NA

    # ΔP
    out["ΔP"] = out[["edgeYES", "edgeNO"]].max(axis=1, skipna=True)

    # Round the few numeric outputs
    for c in ["pPM", "qPM", "pRN", "qRN", "edgeYES", "edgeNO", "ΔP", "pm_sum"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(7)

    return out

def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        # identity / traceability
        "snapshot_time_utc",
        "ticker",
        "week_monday", "week_friday", "week_sunday",
        "slug",
        "event_id", "market_id", "condition_id",
        "yes_token_id", "no_token_id",
        "event_endDate",
        "event_title",
        "market_question",

        # model inputs (keep if you want them)
        "S", "K", "σ", "T_years", "r",

        # prices + probs
        "pPM", "qPM", "pRN", "qRN",

        # ONLY indicators
        "edgeYES", "edgeNO", "ΔP",
    ]
    cols_present = [c for c in cols if c in df.columns]
    return df[cols_present].copy()



def finalize_and_write(
    targets_df: pd.DataFrame,
    pm_df: pd.DataFrame,
    rn_df: pd.DataFrame,
    *,
    out_dir: str,
    run_id: str,
    run_dir: str,
    final_filename: str = "final_dataset.csv",
    merge_tolerance_min: int = 15,
    pm_sum_min: float = 0.98,
    pm_sum_max: float = 1.02,
    pm_extreme_cutoff: float = 0.03,
    sigma_floor: float = 0.02,
    lambda_blend: float = 1.0,
    filter_invalid: bool = True,
) -> str:
 
    pm_std = standardize_polymarket_df(pm_df)
    rn_std = standardize_rn_df(rn_df)

    # Bring week fields onto pm rows (traceable join: slug)
    if targets_df is not None and not targets_df.empty:
        keep = [c for c in ["slug", "ticker", "week_monday", "week_friday", "week_sunday"] if c in targets_df.columns]
        if "slug" in keep:
            pm_std = pm_std.merge(
                targets_df[keep].drop_duplicates("slug"),
                on="slug",
                how="left",
                validate="m:1",
            )

    merged = merge_pm_with_rn(pm_std, rn_std, tolerance_minutes=merge_tolerance_min)
    
    effective_filter_invalid = filter_invalid and (rn_std is not None) and (not rn_std.empty) and (len(rn_std.columns) > 0)
    
    enriched = add_signals_and_clean(
    merged,
    pm_sum_min=pm_sum_min,
    pm_sum_max=pm_sum_max,
    filter_invalid=effective_filter_invalid,
)

    
    if enriched is None:
        print("[Warn] add_signals_and_clean returned None; falling back to merged dataframe.")
        enriched = merged.copy()

    final_df = select_final_columns(enriched)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, final_filename)
    final_df.to_csv(out_path, index=False)
    final_hist = final_df.copy()
    final_hist["run_id"] = run_id
    final_hist["run_time_utc"] = datetime.now(timezone.utc).isoformat()
    append_df_to_csv(
    final_hist,
    os.path.join(out_dir, "history", "final_dataset_history.csv")
)
    print("[History] appended final rows")
    print(f"[Write] {out_path}  (rows={len(final_df)})")
    return out_path


# -----------------------
# NEW main() that runs PART 1 + PART 2 end-to-end
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Polymarket + yfinance pipeline (FULL)")
    parser.add_argument("--out-dir", type=str, default="./data", help="Output directory for CSVs")
    parser.add_argument("--final-filename", type=str, default="final_dataset.csv", help="Final merged CSV filename")

    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers, e.g. NVDA,AAPL")
    parser.add_argument("--tickers-csv", type=str, default=None, help="CSV with a 'ticker' column")
    parser.add_argument("--slug-overrides", type=str, default=None, help="Optional .json or .csv mapping ticker->slug")

    parser.add_argument("--risk-free-rate", type=float, default=Config().risk_free_rate, help="Flat risk-free rate r")
    parser.add_argument("--tz", type=str, default=Config().tz_name, help="Timezone for trading-week logic")

    # Merge + cleaning controls
    parser.add_argument("--merge-tol-min", type=int, default=15, help="Nearest-merge tolerance (minutes)")
    parser.add_argument("--pm-sum-min", type=float, default=0.98, help="Min allowed pPM+qPM")
    parser.add_argument("--pm-sum-max", type=float, default=1.02, help="Max allowed pPM+qPM")
    parser.add_argument("--pm-extreme-cutoff", type=float, default=0.03, help="Filter pPM < c or > 1-c")
    parser.add_argument("--sigma-floor", type=float, default=0.02, help="σ floor for ΔPnorm")
    parser.add_argument("--lambda-blend", type=float, default=1.0, help="λ in p_hat = pPM + λ(pRN - pPM)")
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep invalid rows (don’t filter); invalidity is still flagged in quality columns",
    )

    args = parser.parse_args()

    cfg = Config(
        tz_name=args.tz,
        risk_free_rate=float(args.risk_free_rate),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Determine trading week ----
    tz = ZoneInfo(cfg.tz_name)
    today_local = datetime.now(tz).date()
    week_monday, week_friday, week_sunday = trading_week_bounds(today_local)

    print(f"[Week] Today (local): {today_local.isoformat()}  |  "
          f"Week: {week_monday.isoformat()} → {week_sunday.isoformat()}  |  "
          f"Closes Friday: {week_friday.isoformat()}")

    # ---- Select tickers ----
    tickers = parse_tickers_arg(args.tickers_csv, args.tickers)
    print(f"[Tickers] Selected {len(tickers)} tickers: {', '.join(tickers)}")

    # ---- Slug mapping ----
    slug_overrides = load_slug_overrides(args.slug_overrides)

    targets_df = build_targets_df(
        tickers=tickers,
        week_monday=week_monday,
        week_friday=week_friday,
        week_sunday=week_sunday,
        slug_overrides=slug_overrides,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    print("[Run] run_id:", run_id)
    print("[Run] run_dir:", run_dir)
    
# ---- Write targets (per-run) ----
    targets_path = os.path.join(run_dir, "polymarket_target_markets.csv")
    targets_df.to_csv(targets_path, index=False)
    print(f"[Write] {targets_path}")

# ---- Fetch + write Polymarket snapshots (per-run) ----
    pm_df = fetch_polymarket_snapshots(targets_df, cfg)
    pm_path = os.path.join(run_dir, "polymarket_price_snapshots.csv")
    pm_df.to_csv(pm_path, index=False)
    print(f"[Write] {pm_path} (rows={len(pm_df)})")

# ---- Compute + write RN (per-run) ----
    rn_df = compute_rn_probabilities(pm_df, targets_df, cfg)
    rn_path = os.path.join(run_dir, "options_risk_neutral_probabilities.csv")
    rn_df.to_csv(rn_path, index=False)
    print(f"[Write] {rn_path} (rows={len(rn_df)})")


    # ---- Fetch Polymarket snapshots ----
    pm_df = fetch_polymarket_snapshots(targets_df, cfg)
    pm_path = os.path.join(args.out_dir, "polymarket_price_snapshots.csv")
    pm_df.to_csv(pm_path, index=False)
    
    pm_hist = pm_df.copy()
    pm_hist["run_id"] = run_id
    pm_hist["run_time_utc"] = datetime.now(timezone.utc).isoformat()
    append_df_to_csv(
    pm_hist,
    os.path.join(args.out_dir, "history", "polymarket_price_snapshots_history.csv")
)
    
    print(f"[Write] {pm_path}  (rows={len(pm_df)})")

    # ---- Compute RN probabilities from options ----
    rn_df = compute_rn_probabilities(pm_df, targets_df, cfg)
    rn_path = os.path.join(args.out_dir, "options_risk_neutral_probabilities.csv")
    rn_df.to_csv(rn_path, index=False)
    rn_hist = rn_df.copy()
    rn_hist["run_id"] = run_id
    rn_hist["run_time_utc"] = datetime.now(timezone.utc).isoformat()

    append_df_to_csv(
        rn_hist,
        os.path.join(args.out_dir, "history", "options_risk_neutral_probabilities_history.csv")
)
    print(f"[Write] {rn_path}  (rows={len(rn_df)})")

    # ---- PART 2: Final merge + clean + signals ----
    finalize_and_write(
        targets_df=targets_df,
        pm_df=pm_df,
        rn_df=rn_df,
        out_dir=args.out_dir,
        run_id=run_id,
        run_dir=run_dir,
        final_filename=args.final_filename,
        merge_tolerance_min=int(args.merge_tol_min),
        pm_sum_min=float(args.pm_sum_min),
        pm_sum_max=float(args.pm_sum_max),
        filter_invalid=not args.keep_invalid,
)


# -----------------------
# NEW __main__ guard (must be at very bottom of the file)
# -----------------------
    
if __name__ == "__main__":
    main()

