import os
import sys
from pathlib import Path

# Allow multiple OpenMP initializations in Windows Python environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure repo root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timezone

from copula_execution.ingestion import MarketIngestion
from copula_execution.gateway import PaperExecutionGateway
from copula_execution.risk_manager import RiskManager, RiskLimits
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.engine import ExecutionEngine
from copula_execution.classify import classify_kalshi_event
from copula_execution.types import OrderBook, OrderBookLevel, MarketVenue, MarketContract


def run_live_or_simulated_demo():
    print("=" * 70)
    print(" COPULA EXECUTION LAYER — LIVE SCANNER & EXECUTION DEMO")
    print("=" * 70)

    # 1. Initialize components
    ingestion = MarketIngestion()
    gateway = PaperExecutionGateway(latency_ms=10.0)
    limits = RiskLimits(
        max_total_exposure_dollars=2000.0,
        max_event_exposure_dollars=500.0,
        max_contract_order_dollars=100.0,
        min_net_edge_dollars=0.015,
        max_daily_loss_dollars=200.0
    )
    risk_manager = RiskManager(limits)
    scanner = CoherenceScanner(min_edge=0.015)
    engine = ExecutionEngine(gateway=gateway, risk_manager=risk_manager, scanner=scanner)

    # 2. Ingest live Kalshi events
    print("\n[1] Querying Kalshi API for active economic / sports bracket events...")
    open_events = ingestion.fetch_open_events(limit=40)
    target_event = None
    target_contracts = []
    books = {}

    for ev in open_events:
        series = ev.get("series_ticker", "")
        if any(k in series.lower() for k in ["gdp", "cpi", "fed", "rate"]):
            ev_ticker = ev.get("event_ticker")
            contracts = ingestion.fetch_event_markets(ev_ticker)
            if len(contracts) >= 3:
                target_event = ev
                target_contracts = contracts
                break

    if target_event:
        print(f" -> Found active event: {target_event.get('title')} ({target_event.get('event_ticker')})")
        print(f" -> Ingesting order books for {len(target_contracts)} contracts...")
        # every leg's book: a buy-basket arbitrage is only valid with a live ask on all of them
        for c in target_contracts[:40]:
            ob = ingestion.fetch_order_book(c.ticker)
            if ob:
                books[c.ticker] = ob
                print(f"    * {c.ticker} [{c.subtitle}]: Best Bid={ob.best_yes_bid} | Best Ask={ob.best_yes_ask} | Spread={ob.spread}")
    else:
        print(" -> No active open bracket event with liquid book. Generating realistic bracket scenario...")
        ev_ticker = "KXUSCPI-DEMO"
        target_contracts = [
            MarketContract("CPI-B1.5", ev_ticker, MarketVenue.KALSHI, "CPI <= 1.5%", cap_strike=1.5, strike_type="less"),
            MarketContract("CPI-B2.5", ev_ticker, MarketVenue.KALSHI, "1.5% < CPI <= 2.5%", floor_strike=1.5, cap_strike=2.5, strike_type="between"),
            MarketContract("CPI-B3.5", ev_ticker, MarketVenue.KALSHI, "2.5% < CPI <= 3.5%", floor_strike=2.5, cap_strike=3.5, strike_type="between"),
            MarketContract("CPI-B4.5", ev_ticker, MarketVenue.KALSHI, "CPI > 3.5%", floor_strike=3.5, strike_type="greater"),
        ]
        # The brackets tile the real line, so the classifier proves them mutually exclusive AND exhaustive.
        classify_kalshi_event(target_contracts, event_mutually_exclusive=True)
        # Introduce a real-world mispricing where the basket sums to $0.84 (16c gross arb)
        books = {
            "CPI-B1.5": OrderBook("CPI-B1.5", MarketVenue.KALSHI, datetime.now(timezone.utc),
                                  yes_bids=[OrderBookLevel(0.10, 100)], yes_asks=[OrderBookLevel(0.14, 100)]),
            "CPI-B2.5": OrderBook("CPI-B2.5", MarketVenue.KALSHI, datetime.now(timezone.utc),
                                  yes_bids=[OrderBookLevel(0.30, 100)], yes_asks=[OrderBookLevel(0.35, 100)]),
            "CPI-B3.5": OrderBook("CPI-B3.5", MarketVenue.KALSHI, datetime.now(timezone.utc),
                                  yes_bids=[OrderBookLevel(0.20, 100)], yes_asks=[OrderBookLevel(0.24, 100)]),
            "CPI-B4.5": OrderBook("CPI-B4.5", MarketVenue.KALSHI, datetime.now(timezone.utc),
                                  yes_bids=[OrderBookLevel(0.08, 100)], yes_asks=[OrderBookLevel(0.11, 100)]),
        }
        for c in target_contracts:
            ob = books[c.ticker]
            print(f"    * {c.ticker} [{c.title}]: Best Bid={ob.best_yes_bid} | Best Ask={ob.best_yes_ask}")

    # 3. Model calibration & quoting
    print("\n[2] Calibrating Structural Model & Generating Fair Values...")
    # Fair values come from THIS event's own live books (no hardcoded strikes).
    pricing = engine.price_event(target_contracts, books)
    print(f" -> Event kind: {target_contracts[0].event_kind.value} | method: {pricing.method}")
    if pricing.params.get("mu") is not None:
        print(f" -> Fitted {pricing.params['dist']}: mean={pricing.params['mu']:.3f}, scale={pricing.params['sigma']:.3f}, "
              f"fit rmse={pricing.params['rmse']:.4f} over {pricing.params['n_anchors']} liquid brackets")
    for w in pricing.warnings:
        print(f" !! {w}")

    for c in target_contracts[:6]:
        fv = pricing.fair_values[c.ticker]
        if fv.fv is None:
            print(f"    * {c.ticker}: not priced ({fv.note})")
            continue
        bid_quote, ask_quote = engine.quoter.quote_fair_value(fv, c)
        fmt = lambda x: "none" if x is None else f"{x:.2f}"
        print(f"    * Model FV for {c.ticker}: {fv.fv:.4f} +/- {fv.uncertainty:.4f} "
              f"| Quoted Two-Sided Market: Bid={fmt(bid_quote)} / Ask={fmt(ask_quote)}")

    # 4. Scan & Execute Arbitrage
    print("\n[3] Scanning for Coherence / Arbitrage Violations & Routing Orders...")
    order_results = engine.scan_and_execute_bracket_arbitrage(
        event_ticker=target_contracts[0].event_ticker,
        contracts=target_contracts,
        books=books,
        auto_execute=True
    )

    print(f" -> Flags detected: {len(engine.flags_history)}")
    for f in engine.flags_history:
        print(f"    [FLAG] ID: {f.flag_id} | Type: {f.disagreement_type.value} | Net Edge: ${f.gap_size:.4f} | Gross Gap: ${f.gross_discrepancy:.4f}")

    print(f" -> Orders executed: {len(order_results)}")
    for r in order_results:
        print(f"    [ORDER] ID: {r.order_id} | Status: {r.status.value} | Filled: {r.filled_count} contracts @ ${r.average_fill_price:.4f} | Fee: ${r.fee_paid:.4f}")

    # 5. Convergence tracking
    print("\n[4] Convergence Rate Tracking Simulation...")
    if engine.flags_history:
        flag = engine.flags_history[0]
        # Simulate price moving 40% toward fair value
        converged_prices = {}
        for t in flag.contracts_involved:
            orig = flag.market_prices[t]["mid"]
            fv = flag.model_fair_values[t]
            converged_prices[t] = round(orig + 0.4 * (fv - orig), 4)

        engine.record_convergence_observation(flag, converged_prices)
        rec = engine.convergence_records[0]
        print(f" -> Convergence verified: Ticker={rec['ticker']} | Initial Gap={rec['initial_gap']:.4f} -> New Gap={rec['current_gap']:.4f} (Converged={rec['converged']})")

    # 6. Summary Report
    print("\n[5] Portfolio & Risk Gate State:")
    summary = engine.get_portfolio_summary()
    for k, v in summary.items():
        print(f"    - {k}: {v}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_live_or_simulated_demo()
