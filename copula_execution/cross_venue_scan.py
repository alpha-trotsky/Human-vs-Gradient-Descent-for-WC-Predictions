"""
Multi-Market Cross-Venue Scanner:
Runs the Copula Coherence and Pricing Engine across ALL available multi-contract
events on Polymarket and Kalshi. Computes basket sums, monotonicity inversions,
fee-adjusted edges, and model-derived fair values.
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from copula_execution.types import (
    MarketVenue, MarketContract, OrderBook, OrderBookLevel, DisagreementType
)
from copula_execution.ingestion import MarketIngestion
from copula_execution.polymarket import PolymarketIngestion
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.pricer import EventPricer
from copula_execution.risk_manager import FeeCalculator


def scan_all_venues():
    print("=" * 80)
    print(" COPULA ENGINE — MULTI-VENUE COMPREHENSIVE SCAN (POLYMARKET + KALSHI)")
    print(f" Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    poly = PolymarketIngestion()
    kalshi = MarketIngestion()
    scanner = CoherenceScanner(min_edge=0.01) # 1 cent min edge
    engine_pricer = EventPricer()

    all_results = []

    # ---------------------------------------------------------
    # PART 1: POLYMARKET SCAN
    # ---------------------------------------------------------
    print("\n[PHASE 1] Scanning Polymarket Events...")
    poly_events = poly.fetch_open_events(limit=40)
    print(f"Found {len(poly_events)} active Polymarket events.")

    poly_scanned_count = 0
    poly_flags_count = 0

    for ev in poly_events:
        ev_id = str(ev.get("id"))
        title = ev.get("title", "")
        # Open, tradable markets only, with YES token / fee schedule / event kind resolved by ingestion
        contracts = poly.contracts_from_event(ev)

        # Only scan multi-market / bracket events
        if len(contracts) < 2:
            continue

        poly_scanned_count += 1
        kind = contracts[0].event_kind.value
        exh = " exhaustive" if all(c.exhaustive for c in contracts) else ""
        print(f"\n--- [Polymarket] Event #{ev_id}: '{title}' ({len(contracts)} open markets, {kind}{exh}) ---")

        books = {}
        for c in contracts:
            ob = poly.fetch_order_book(c.ticker)
            if ob:
                books[c.ticker] = ob

        print(f"  > Successfully retrieved {len(books)}/{len(contracts)} CLOB order books.")

        # Basket sums only mean something for a mutually exclusive set
        sum_asks = sum(b.best_yes_ask for b in books.values() if b.best_yes_ask is not None)
        sum_bids = sum(b.best_yes_bid for b in books.values() if b.best_yes_bid is not None)
        total_poly_fees = sum(FeeCalculator.fee_per_contract(c, books[c.ticker].best_yes_ask, scanner.trade_size)
                              for c in contracts if c.ticker in books and books[c.ticker].best_yes_ask is not None)
        applies = "" if kind == "partition" else "  (not a partition: sums are informational only)"
        print(f"  > Basket Ask Sum: ${sum_asks:.4f} | Basket Bid Sum: ${sum_bids:.4f} | Est. Fees/basket: ${total_poly_fees:.4f}{applies}")

        pricing = engine_pricer.price_event(contracts, books)
        priced = sum(1 for f in pricing.fair_values.values() if f.fv is not None)
        print(f"  > Pricing: {pricing.method} ({priced}/{len(contracts)} contracts priced)"
              + (f" | {pricing.warnings[0]}" if pricing.warnings else ""))

        # Run mathematical coherence scanner (guards decide which checks are valid for this event kind)
        flags = scanner.check_mutually_exclusive_sum(ev_id, contracts, books) + \
            scanner.check_monotonicity(ev_id, contracts, books)
        for f in flags:
            poly_flags_count += 1
            print(f"  >>> [FLAG DETECTED] {f.disagreement_type.value.upper()}: Gap = ${f.gross_discrepancy:.4f} | Net Edge After Fees = ${f.gap_size:.4f}")
            for t in f.suggested_trades[:3]:
                print(f"      - {t['action'].value.upper()} {t['ticker'][:14]}... @ ${t['price']:.4f}")
            all_results.append({
                "venue": "Polymarket",
                "event": title,
                "event_id": ev_id,
                "flag": f,
                "ask_sum": sum_asks,
                "bid_sum": sum_bids
            })

    # ---------------------------------------------------------
    # PART 2: KALSHI SCAN
    # ---------------------------------------------------------
    print("\n\n[PHASE 2] Scanning Kalshi Economics & Sports Bracket Events...")
    kalshi_events = kalshi.fetch_open_events(limit=40)
    kalshi_scanned_count = 0
    kalshi_flags_count = 0

    for ev in kalshi_events:
        series = ev.get("series_ticker", "")
        ev_ticker = ev.get("event_ticker", "")
        title = ev.get("title", "")

        # Look for multi-bracket series
        if not any(k in series.lower() for k in ["gdp", "cpi", "fed", "rate", "year", "inf"]):
            continue

        contracts = kalshi.fetch_event_markets(ev_ticker)
        if len(contracts) < 3:
            continue

        kalshi_scanned_count += 1
        kind = contracts[0].event_kind.value
        exh = " exhaustive" if all(c.exhaustive for c in contracts) else ""
        print(f"\n--- [Kalshi] Event {ev_ticker}: '{title}' ({len(contracts)} brackets, {kind}{exh}) ---")

        books = {}
        for c in contracts[:40]:  # a partial book set can never prove a buy-basket (the scanner enforces it)
            ob = kalshi.fetch_order_book(c.ticker)
            if ob:
                books[c.ticker] = ob

        print(f"  > Successfully retrieved {len(books)} order books.")

        sum_asks = sum(b.best_yes_ask for b in books.values() if b.best_yes_ask is not None)
        sum_bids = sum(b.best_yes_bid for b in books.values() if b.best_yes_bid is not None)
        total_kalshi_fees = sum(FeeCalculator.fee_per_contract(c, books[c.ticker].best_yes_ask, scanner.trade_size)
                                for c in contracts if c.ticker in books and books[c.ticker].best_yes_ask is not None)
        applies = "" if kind == "partition" else "  (not a partition: sums are informational only)"
        print(f"  > Basket Ask Sum: ${sum_asks:.4f} | Basket Bid Sum: ${sum_bids:.4f} | Est. Fees/basket: ${total_kalshi_fees:.4f}{applies}")

        pricing = engine_pricer.price_event(contracts, books)
        priced = sum(1 for f in pricing.fair_values.values() if f.fv is not None)
        print(f"  > Pricing: {pricing.method} ({priced}/{len(contracts)} contracts priced)"
              + (f" | {pricing.warnings[0]}" if pricing.warnings else ""))

        # Check Sum to 1 & Monotonicity
        sum_flags = scanner.check_mutually_exclusive_sum(ev_ticker, contracts, books)
        mono_flags = scanner.check_monotonicity(ev_ticker, contracts, books)
        k_flags = sum_flags + mono_flags

        for f in k_flags:
            kalshi_flags_count += 1
            print(f"  >>> [FLAG DETECTED] {f.disagreement_type.value.upper()}: Gap = ${f.gross_discrepancy:.4f} | Net Edge After Fees = ${f.gap_size:.4f}")
            for t in f.suggested_trades[:3]:
                print(f"      - {t['action'].value.upper()} {t['ticker']} @ ${t['price']:.4f}")
            all_results.append({
                "venue": "Kalshi",
                "event": title,
                "event_id": ev_ticker,
                "flag": f,
                "ask_sum": sum_asks,
                "bid_sum": sum_bids
            })

    # ---------------------------------------------------------
    # PART 3: COMPARATIVE SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(" SCAN RESULTS & CROSS-VENUE DISCOVERY SUMMARY")
    print("=" * 80)
    print(f"Polymarket Multi-Contract Events Scanned: {poly_scanned_count} | Arbitrage Flags: {poly_flags_count}")
    print(f"Kalshi Multi-Bracket Events Scanned:     {kalshi_scanned_count} | Arbitrage Flags: {kalshi_flags_count}")
    print(f"Total Actionable Opportunities Found:     {len(all_results)}")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    scan_all_venues()
