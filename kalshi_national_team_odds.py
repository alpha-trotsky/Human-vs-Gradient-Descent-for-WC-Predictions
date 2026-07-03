"""
Kalshi national-team soccer odds pipeline.

For every national-team soccer match on Kalshi since Jan 1 2025, grabs the
market prices ~2 hours before kickoff (last hourly candle closing at or
before kickoff - 2h) and writes one row per outcome (Home / Away / Tie)
to a CSV.

Usage:
    pip install requests
    python kalshi_national_team_odds.py
    python kalshi_national_team_odds.py --series KXWCGAME --out wc_odds.csv

No API key needed: all endpoints used here are public market-data endpoints.
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone

import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
START_TS = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
HOURS_BEFORE = 2               # snapshot time: kickoff minus 2 hours
CANDLE_INTERVAL = 60           # hourly candles (valid: 1, 60, 1440 minutes)
LOOKBACK_HOURS = 48            # search window before the snapshot time
SLEEP = 0.15                   # polite delay between requests

session = requests.Session()
session.headers["User-Agent"] = "kalshi-odds-pipeline/1.0"


def get(path, params=None, ok404=False):
    """GET with basic 429 backoff."""
    for attempt in range(6):
        r = session.get(f"{BASE}{path}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 404 and ok404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Rate-limited too long on {path}")


def paginate(path, params, key):
    """Cursor-based pagination helper."""
    cursor = None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        data = get(path, p)
        items = data.get(key) or []
        yield from items
        cursor = data.get("cursor")
        if not cursor or not items:
            return


def discover_national_team_series():
    """
    Find soccer series that involve national teams.

    Heuristic: series in the Sports category whose ticker/title/tags mention
    soccer AND a national-team competition (world cup, nations league,
    qualifiers, friendlies, euro, copa, gold cup). KXWCGAME (FIFA World Cup
    games) is always included.
    """
    natl_words = ("world cup", "nations", "qualif", "friendl", "euro",
                  "copa", "gold cup", "international")
    found = {}
    for s in paginate("/series", {"category": "Sports", "limit": 200}, "series"):
        blob = " ".join([s.get("ticker", ""), s.get("title", ""),
                         " ".join(s.get("tags") or [])]).lower()
        if "soccer" not in blob and "football" not in blob and "fifa" not in blob:
            continue
        if any(w in blob for w in natl_words):
            found[s["ticker"]] = s.get("title", "")
    found.setdefault("KXWCGAME", "FIFA World Cup games")
    return found


def fetch_markets_for_event(event_ticker):
    """Fetch markets separately when with_nested_markets doesn't nest them."""
    data = get("/markets", {"event_ticker": event_ticker, "limit": 100}, ok404=True)
    return (data or {}).get("markets") or []


def fetch_match_events(series_ticker):
    """All events (matches) in a series with kickoff after START_TS."""
    events = []
    for ev in paginate("/events", {
        "series_ticker": series_ticker,
        "with_nested_markets": "true",
        "limit": 200,
    }, "events"):
        # Events have no top-level date; kickoff lives in each market's occurrence_datetime.
        # Some series don't nest markets even with with_nested_markets=true — fetch separately.
        markets = ev.get("markets") or []
        if not markets:
            markets = fetch_markets_for_event(ev["event_ticker"])
            ev["markets"] = markets
            time.sleep(SLEEP)
        strike = markets[0].get("occurrence_datetime") if markets else None
        if not strike:
            continue
        kickoff = datetime.fromisoformat(strike.replace("Z", "+00:00"))
        if int(kickoff.timestamp()) >= START_TS:
            ev["_kickoff"] = kickoff
            events.append(ev)
    return events


def candles_for_market(series_ticker, market_ticker, snapshot_ts):
    """Hourly candles in [snapshot - LOOKBACK, snapshot]; fall back to the
    historical archive endpoint if the market has been archived."""
    params = {
        "start_ts": snapshot_ts - LOOKBACK_HOURS * 3600,
        "end_ts": snapshot_ts,
        "period_interval": CANDLE_INTERVAL,
    }
    data = get(f"/series/{series_ticker}/markets/{market_ticker}/candlesticks",
               params, ok404=True)
    if data is None or not data.get("candlesticks"):
        data = get(f"/historical/markets/{market_ticker}/candlesticks",
                   params, ok404=True)
    return (data or {}).get("candlesticks") or []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_snapshot(candles):
    """Last candle at/before the snapshot time -> price fields.
    Handles both *_dollars (live endpoint) and plain (historical) keys."""
    if not candles:
        return None
    c = candles[-1]

    def field(block, name):
        b = c.get(block) or {}
        return num(b.get(f"{name}_dollars", b.get(name)))

    return {
        "price_close": field("price", "close") or field("price", "mean"),
        "yes_bid": field("yes_bid", "close"),
        "yes_ask": field("yes_ask", "close"),
        "candle_end_ts": c.get("end_period_ts"),
        "volume": c.get("volume_fp", c.get("volume")),
    }


def debug_dump(label, obj, max_items=3):
    """Pretty-print the first few items of a list or the keys of a dict."""
    import json
    print(f"\n{'='*60}")
    print(f"DEBUG: {label}")
    print('='*60)
    if isinstance(obj, list):
        for i, item in enumerate(obj[:max_items]):
            print(f"  [{i}]: {json.dumps(item, indent=4, default=str)}")
        if len(obj) > max_items:
            print(f"  ... ({len(obj) - max_items} more items)")
    else:
        print(json.dumps(obj, indent=4, default=str))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=None,
                    help="Series tickers to pull (default: auto-discover)")
    ap.add_argument("--out", default="kalshi_national_team_odds_2h.csv")
    ap.add_argument("--start", default="2025-01-01",
                    help="Earliest kickoff date, YYYY-MM-DD")
    ap.add_argument("--debug", action="store_true",
                    help="Dump raw API responses to find correct field names")
    args = ap.parse_args()

    global START_TS
    START_TS = int(datetime.strptime(args.start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp())

    if args.debug:
        print("\n--- DEBUG: raw /series response (first 2 Sports series) ---")
        raw_series = get("/series", {"category": "Sports", "limit": 5})
        debug_dump("/series response keys", list(raw_series.keys()))
        debug_dump("first 2 series objects", (raw_series.get("series") or [])[:2])

        probe_ticker = (args.series or ["KXWCGAME"])[0]
        print(f"\n--- DEBUG: raw /events for series={probe_ticker} (first 2 events) ---")
        raw_events = get("/events", {
            "series_ticker": probe_ticker,
            "with_nested_markets": "true",
            "limit": 3,
        })
        debug_dump("/events response keys", list(raw_events.keys()))
        events_list = raw_events.get("events") or []
        debug_dump("first 2 event objects", events_list[:2])

        if events_list:
            first_ev = events_list[0]
            markets = first_ev.get("markets") or []
            if markets:
                first_mkt = markets[0]
                print(f"\n--- DEBUG: raw candlesticks for market {first_mkt['ticker']} ---")
                now_ts = int(time.time())
                raw_candles = get(
                    f"/series/{probe_ticker}/markets/{first_mkt['ticker']}/candlesticks",
                    {"start_ts": now_ts - 7 * 24 * 3600, "end_ts": now_ts,
                     "period_interval": 60},
                    ok404=True,
                )
                debug_dump("candlestick response", raw_candles or {"note": "404 / no data"})
            else:
                print("\nDEBUG: no markets nested in first event (check with_nested_markets param)")
        return

    if args.series:
        series_map = {t: t for t in args.series}
    else:
        print("Discovering national-team soccer series...")
        series_map = discover_national_team_series()
    print(f"Series to scan: {', '.join(series_map)}")

    rows = []
    for series_ticker in series_map:
        print(f"\n=== {series_ticker} ===")
        try:
            events = fetch_match_events(series_ticker)
        except requests.HTTPError as e:
            print(f"  skipped ({e})")
            continue
        print(f"  {len(events)} matches since {args.start}")

        for i, ev in enumerate(events, 1):
            kickoff = ev["_kickoff"]
            snapshot_ts = int(kickoff.timestamp()) - HOURS_BEFORE * 3600
            # skip games whose snapshot time is still in the future
            if snapshot_ts > time.time():
                continue
            for m in ev.get("markets") or []:
                snap = None
                try:
                    candles = candles_for_market(series_ticker, m["ticker"],
                                                 snapshot_ts)
                    snap = extract_snapshot(candles)
                except requests.HTTPError:
                    pass
                time.sleep(SLEEP)
                price = snap["price_close"] if snap else None
                rows.append({
                    "series": series_ticker,
                    "event_ticker": ev["event_ticker"],
                    "match": ev.get("title", ""),
                    "kickoff_utc": kickoff.isoformat(),
                    "snapshot_utc": datetime.fromtimestamp(
                        snapshot_ts, tz=timezone.utc).isoformat(),
                    "market_ticker": m["ticker"],
                    "outcome": m.get("yes_sub_title", ""),
                    "price_2h_before": price,
                    "implied_prob": price,
                    "decimal_odds": round(1 / price, 3) if price else None,
                    "yes_bid": snap["yes_bid"] if snap else None,
                    "yes_ask": snap["yes_ask"] if snap else None,
                    "result": m.get("result", ""),
                    "market_status": m.get("status", ""),
                    "volume_at_snapshot": snap["volume"] if snap else None,
                })
            if i % 25 == 0:
                print(f"  ...{i}/{len(events)} matches done")

    fieldnames = ["series", "event_ticker", "match", "kickoff_utc",
                  "snapshot_utc", "market_ticker", "outcome",
                  "price_2h_before", "implied_prob", "decimal_odds",
                  "yes_bid", "yes_ask", "result", "market_status",
                  "volume_at_snapshot"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with_price = sum(1 for r in rows if r["price_2h_before"] is not None)
    print(f"\nWrote {len(rows)} rows ({with_price} with a price) -> {args.out}")
    if not rows:
        print("No matches found — check the series tickers with "
              "GET /series?category=Sports or the kalshi.com URL slugs.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
