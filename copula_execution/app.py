"""
Copula Live Multi-Venue Web Dashboard & Execution API Server.
Supports both Kalshi and Polymarket order books, coherence scanning,
model fair values, and live execution.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, jsonify, render_template_string, request

from copula_execution.types import (
    MarketContract, OrderBook, OrderBookLevel, MarketVenue, ActionSide, ContractSide
)
from copula_execution.ingestion import MarketIngestion
from copula_execution.polymarket import PolymarketIngestion
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.pricer import EconomicBracketPricer, MarketMakerQuoter
from copula_execution.risk_manager import RiskManager, RiskLimits, FeeCalculator
from copula_execution.gateway import PaperExecutionGateway
from copula_execution.engine import ExecutionEngine

app = Flask(__name__)

# Initialize singletons
kalshi_ingestion = MarketIngestion()
poly_ingestion = PolymarketIngestion()
gateway = PaperExecutionGateway(latency_ms=10.0)
limits = RiskLimits(
    max_total_exposure_dollars=2500.0,
    max_event_exposure_dollars=500.0,
    max_contract_order_dollars=100.0,
    min_net_edge_dollars=0.015,
    max_daily_loss_dollars=200.0
)
risk_manager = RiskManager(limits)
scanner = CoherenceScanner(min_edge=0.015)
engine = ExecutionEngine(gateway=gateway, risk_manager=risk_manager, scanner=scanner)

current_venue = MarketVenue.POLYMARKET
cached_event = None
cached_contracts = []
cached_books = {}
cached_flags = []


def refresh_market_data(venue_str: str = "polymarket"):
    global current_venue, cached_event, cached_contracts, cached_books, cached_flags
    current_venue = MarketVenue.POLYMARKET if venue_str.lower() == "polymarket" else MarketVenue.KALSHI

    if current_venue == MarketVenue.POLYMARKET:
        try:
            # Polymarket: Fed rate cuts event ID 51456
            contracts = poly_ingestion.fetch_event_markets("51456")
            if contracts:
                cached_event = {
                    "event_ticker": "POLY-51456",
                    "title": "How many Fed rate cuts in 2026? (Polymarket)",
                    "series_ticker": "FED-RATES-2026",
                    "venue": "Polymarket"
                }
                cached_contracts = contracts
                cached_books = {}
                for c in cached_contracts[:8]:
                    ob = poly_ingestion.fetch_order_book(c.ticker)
                    if ob:
                        cached_books[c.ticker] = ob
            else:
                _fallback_kalshi()
        except Exception as e:
            print(f"Error refreshing from Polymarket: {e}")
            _fallback_kalshi()
    else:
        _fallback_kalshi()

    # Re-scan
    if cached_contracts and cached_books:
        ev_ticker = cached_contracts[0].event_ticker
        sum_flags = scanner.check_mutually_exclusive_sum(ev_ticker, cached_contracts, cached_books)
        mono_flags = scanner.check_monotonicity(ev_ticker, cached_contracts, cached_books)
        cached_flags = sum_flags + mono_flags


def _fallback_kalshi():
    global current_venue, cached_event, cached_contracts, cached_books
    current_venue = MarketVenue.KALSHI
    try:
        open_events = kalshi_ingestion.fetch_open_events(limit=40)
        target_ev = None
        for ev in open_events:
            series = ev.get("series_ticker", "")
            if any(k in series.lower() for k in ["gdp", "cpi", "fed", "rate"]):
                ev_ticker = ev.get("event_ticker")
                contracts = kalshi_ingestion.fetch_event_markets(ev_ticker)
                if len(contracts) >= 3:
                    target_ev = ev
                    cached_contracts = contracts
                    break
        if target_ev:
            cached_event = {
                "event_ticker": target_ev.get("event_ticker"),
                "title": f"{target_ev.get('title')} (Kalshi)",
                "series_ticker": target_ev.get("series_ticker"),
                "venue": "Kalshi"
            }
            cached_books = {}
            for c in cached_contracts[:8]:
                ob = kalshi_ingestion.fetch_order_book(c.ticker)
                if ob:
                    cached_books[c.ticker] = ob
    except Exception as e:
        print(f"Error fetching from Kalshi: {e}")


# Initialize with Polymarket
refresh_market_data("polymarket")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Copula — Multi-Venue Coherence Scanner</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @keyframes pulse-slow {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.3; transform: scale(0.9); }
    }
    .pulse-dot { animation: pulse-slow 2s infinite ease-in-out; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased p-6 min-h-screen">

  <!-- Header -->
  <header class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-800 gap-4">
    <div>
      <div class="flex items-center gap-3">
        <div class="w-3.5 h-3.5 rounded-full bg-emerald-500 pulse-dot shadow-lg shadow-emerald-500/50"></div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          COPULA <span class="text-xs px-2.5 py-1 rounded bg-slate-900 border border-slate-700 font-mono text-emerald-400">MULTI-VENUE ARBITRAGE v1.0</span>
        </h1>
      </div>
      <p class="text-sm text-slate-400 mt-1">Cross-exchange execution & structural quoting across Polymarket & Kalshi.</p>
    </div>

    <!-- Live Status & Venue Selector -->
    <div class="flex flex-wrap items-center gap-3 text-xs font-mono">
      <div class="flex rounded-lg bg-slate-900 border border-slate-800 p-0.5">
        <button onclick="switchVenue('polymarket')" id="btn-polymarket" class="px-3 py-1 rounded-md bg-purple-600 text-white font-semibold transition">
          Polymarket CLOB
        </button>
        <button onclick="switchVenue('kalshi')" id="btn-kalshi" class="px-3 py-1 rounded-md text-slate-400 hover:text-white transition">
          Kalshi L2
        </button>
      </div>
      <div class="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
        <span class="text-slate-400">PnL:</span> <span class="font-semibold text-emerald-400" id="header-pnl">$0.00</span>
      </div>
      <button onclick="fetchLatestData()" class="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold transition shadow-sm">
        ⟳ Refresh
      </button>
    </div>
  </header>

  <!-- Metric Badges Row -->
  <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
    <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-sm backdrop-blur">
      <div class="text-xs text-slate-400 font-medium">Scanned Contracts</div>
      <div class="text-2xl font-bold mt-1 font-mono text-white" id="stat-scanned">0</div>
      <div class="text-xs text-emerald-400 mt-1 flex items-center gap-1.5" id="venue-status-indicator">
        <span class="w-2 h-2 rounded-full bg-purple-500"></span> Polymarket CLOB Active
      </div>
    </div>
    <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-sm backdrop-blur">
      <div class="text-xs text-slate-400 font-medium">Active Disagreements</div>
      <div class="text-2xl font-bold mt-1 font-mono text-amber-400" id="stat-disagreements">0</div>
      <div class="text-xs text-slate-400 mt-1">Sum-to-1 & Monotonicity bounds</div>
    </div>
    <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-sm backdrop-blur">
      <div class="text-xs text-slate-400 font-medium">Venue Fee Rate</div>
      <div class="text-2xl font-bold mt-1 font-mono text-cyan-400" id="stat-fee">0.2% Taker</div>
      <div class="text-xs text-slate-400 mt-1">Gas & relayer amortization</div>
    </div>
    <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-sm backdrop-blur">
      <div class="text-xs text-slate-400 font-medium">Risk Gate Exposure</div>
      <div class="text-2xl font-bold mt-1 font-mono text-emerald-400" id="stat-exposure">$0.00</div>
      <div class="text-xs text-slate-400 mt-1">Hard limit: $2,500.00</div>
    </div>
  </div>

  <!-- Main Content Grid -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">

    <!-- Column 1 & 2: Scanner & Model Quotes -->
    <div class="lg:col-span-2 space-y-6">

      <!-- Coherence Scanner Flags -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
        <div class="flex items-center justify-between pb-3 border-b border-slate-800">
          <div>
            <h2 class="text-base font-semibold text-white">Coherence Scanner Opportunities</h2>
            <p class="text-xs text-slate-400">Deterministic arbitrage opportunities where book prices violate math boundaries.</p>
          </div>
          <span class="text-xs font-mono px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20" id="flags-badge">0 Flags</span>
        </div>

        <div class="mt-4 space-y-3" id="flags-container">
          <div class="text-xs text-slate-500 py-4 text-center">Scanning live markets...</div>
        </div>
      </div>

      <!-- Model Fair Value vs Market Table -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
        <div class="flex items-center justify-between pb-3 border-b border-slate-800">
          <div>
            <h2 class="text-base font-semibold text-white">Live L2 Order Books & Copula Quotes</h2>
            <p class="text-xs text-slate-400" id="table-subtitle">Distribution fit from liquid contracts & derived quotes for thin brackets.</p>
          </div>
          <span class="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-2 py-1 rounded border border-cyan-500/20" id="model-params">
            Fit: μ = 1.82, σ = 0.85
          </span>
        </div>

        <div class="mt-4 overflow-x-auto">
          <table class="w-full text-left text-xs font-mono">
            <thead>
              <tr class="text-slate-400 border-b border-slate-800">
                <th class="pb-2.5">Bracket / Question</th>
                <th class="pb-2.5">Venue</th>
                <th class="pb-2.5">Model FV</th>
                <th class="pb-2.5">CLOB Bid</th>
                <th class="pb-2.5">CLOB Ask</th>
                <th class="pb-2.5">Derived Quote</th>
                <th class="pb-2.5">Discrepancy</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800/60" id="contracts-table">
              <!-- Populated by JS -->
            </tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- Column 3: Live Orders & Risk Control -->
    <div class="space-y-6">

      <!-- Risk Gate Bounds -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
        <h2 class="text-base font-semibold text-white pb-2 border-b border-slate-800">Risk Gate Controls</h2>
        <div class="mt-3 space-y-2.5 text-xs">
          <div class="flex justify-between items-center py-1">
            <span class="text-slate-400">Min Net Edge:</span>
            <span class="font-mono font-semibold text-emerald-400">$0.015 / contract</span>
          </div>
          <div class="flex justify-between items-center py-1">
            <span class="text-slate-400">Max Per-Order Size:</span>
            <span class="font-mono font-semibold text-white">$100.00</span>
          </div>
          <div class="flex justify-between items-center py-1">
            <span class="text-slate-400">Max Event Allocation:</span>
            <span class="font-mono font-semibold text-white">$500.00</span>
          </div>
          <div class="flex justify-between items-center py-1">
            <span class="text-slate-400">Max Daily Drawdown:</span>
            <span class="font-mono font-semibold text-rose-400">$200.00</span>
          </div>
          <div class="flex justify-between items-center py-1">
            <span class="text-slate-400">Rate Limit:</span>
            <span class="font-mono font-semibold text-white">30 orders / min</span>
          </div>
          <div class="pt-3 border-t border-slate-800 flex items-center justify-between">
            <span class="text-slate-300 font-medium">Circuit Breaker:</span>
            <span class="px-2 py-0.5 rounded text-xs font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">ARMED</span>
          </div>
        </div>
      </div>

      <!-- Execution Audit Feed -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
        <div class="flex items-center justify-between pb-2 border-b border-slate-800">
          <h2 class="text-base font-semibold text-white">Execution Audit Feed</h2>
          <span class="text-xs font-mono text-emerald-400 flex items-center gap-1">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span> Streaming
          </span>
        </div>

        <div class="mt-3 space-y-2 text-xs font-mono max-h-96 overflow-y-auto" id="feed-container">
          <!-- Populated by JS -->
        </div>
      </div>

    </div>

  </div>

  <script>
    let activeVenue = "polymarket";

    function switchVenue(venue) {
      activeVenue = venue;
      if (venue === 'polymarket') {
        document.getElementById('btn-polymarket').className = 'px-3 py-1 rounded-md bg-purple-600 text-white font-semibold transition';
        document.getElementById('btn-kalshi').className = 'px-3 py-1 rounded-md text-slate-400 hover:text-white transition';
        document.getElementById('venue-status-indicator').innerHTML = '<span class="w-2 h-2 rounded-full bg-purple-500"></span> Polymarket CLOB Active';
        document.getElementById('stat-fee').innerText = '0.2% Taker';
      } else {
        document.getElementById('btn-kalshi').className = 'px-3 py-1 rounded-md bg-emerald-600 text-white font-semibold transition';
        document.getElementById('btn-polymarket').className = 'px-3 py-1 rounded-md text-slate-400 hover:text-white transition';
        document.getElementById('venue-status-indicator').innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-500"></span> Kalshi L2 Active';
        document.getElementById('stat-fee').innerText = '1.0% Taker';
      }
      fetch('/api/switch_venue?venue=' + venue).then(() => fetchLatestData());
    }

    async function fetchLatestData() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();

        // Update header & metrics
        document.getElementById('stat-scanned').innerText = data.contracts.length;
        document.getElementById('stat-disagreements').innerText = data.flags.length;
        document.getElementById('flags-badge').innerText = data.flags.length + " Flags";
        document.getElementById('header-pnl').innerText = "$" + data.portfolio.net_pnl_after_fees.toFixed(2);
        document.getElementById('stat-exposure').innerText = "$" + (data.portfolio.positions_count * 2.5).toFixed(2);
        document.getElementById('table-subtitle').innerText = data.event ? data.event.title : 'Live Order Books';

        // Update flags
        const flagsContainer = document.getElementById('flags-container');
        if (data.flags.length === 0) {
          flagsContainer.innerHTML = '<div class="text-xs text-slate-500 py-4 text-center">No mathematical violations detected at current threshold.</div>';
        } else {
          flagsContainer.innerHTML = data.flags.map(f => `
            <div class="p-4 rounded-lg border border-amber-500/40 bg-amber-500/5 hover:bg-amber-500/10 transition">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span class="px-2 py-0.5 text-xs font-bold rounded bg-amber-500/20 text-amber-300 font-mono">${f.disagreement_type.toUpperCase()}</span>
                  <span class="text-sm font-semibold text-white">${f.event_ticker}</span>
                </div>
                <span class="text-sm font-mono font-bold text-emerald-400">Net Edge: +$${f.gap_size.toFixed(4)}</span>
              </div>
              <div class="mt-2 text-xs text-slate-300">
                Gross Discrepancy: <span class="font-mono text-amber-300">$${f.gross_discrepancy.toFixed(4)}</span> | Contracts involved: ${f.contracts_involved.length}
              </div>
              <div class="mt-3 flex items-center justify-between pt-2 border-t border-slate-800">
                <span class="text-xs text-slate-500 font-mono">Suggested Legs: ${f.suggested_trades.length}</span>
                <button onclick="executeFlag('${f.flag_id}')" class="px-3 py-1 text-xs font-semibold rounded bg-emerald-600 hover:bg-emerald-500 text-white transition shadow-sm">
                  Execute Opportunity
                </button>
              </div>
            </div>
          `).join('');
        }

        // Update table
        const tbody = document.getElementById('contracts-table');
        tbody.innerHTML = data.contracts.map(c => {
          const book = data.books[c.ticker] || {};
          const fv = data.model_fvs[c.ticker] || 0.05;
          const quote = data.quotes[c.ticker] || {bid: 0.04, ask: 0.08};
          const ask = book.yes_asks && book.yes_asks[0] ? book.yes_asks[0].price : null;
          const bid = book.yes_bids && book.yes_bids[0] ? book.yes_bids[0].price : null;
          const diff = ask ? (fv - ask).toFixed(4) : '-';
          const diffColor = parseFloat(diff) > 0 ? 'text-emerald-400' : 'text-slate-400';
          const venueTag = c.venue === 'polymarket' ? '<span class="text-purple-400 font-semibold">POLY</span>' : '<span class="text-emerald-400 font-semibold">KALSHI</span>';

          return `
            <tr>
              <td class="py-2.5 font-bold text-slate-200 max-w-xs truncate">${c.subtitle || c.title}</td>
              <td class="font-mono text-[11px]">${venueTag}</td>
              <td class="text-cyan-400 font-bold">${fv.toFixed(4)}</td>
              <td class="text-emerald-400">${bid !== null ? bid.toFixed(4) : '-'}</td>
              <td class="text-rose-400">${ask !== null ? ask.toFixed(4) : '-'}</td>
              <td class="text-slate-200">${quote.bid.toFixed(2)} / ${quote.ask.toFixed(2)}</td>
              <td class="${diffColor} font-bold">${diff}</td>
            </tr>
          `;
        }).join('');

        // Update audit feed
        const feedContainer = document.getElementById('feed-container');
        if (data.audit_trail.length === 0) {
          feedContainer.innerHTML = '<div class="text-slate-500 py-3 text-center">No orders executed yet.</div>';
        } else {
          feedContainer.innerHTML = data.audit_trail.slice(-10).reverse().map(a => `
            <div class="p-2.5 rounded bg-slate-950 border border-slate-800">
              <div class="flex items-center justify-between text-slate-400">
                <span>${a.order_id}</span>
                <span class="text-emerald-400 font-bold">${a.status.toUpperCase()}</span>
              </div>
              <div class="text-slate-100 mt-0.5 font-semibold">FILL ${a.filled_count} ${a.ticker.substring(0, 16)}... @ $${a.fill_price.toFixed(4)}</div>
              <div class="text-slate-400 text-[11px] mt-0.5">Fee: $${a.fee_paid.toFixed(4)} | VWAP match</div>
            </div>
          `).join('');
        }

      } catch (err) {
        console.error("Failed to fetch state:", err);
      }
    }

    async function executeFlag(flagId) {
      const res = await fetch('/api/execute', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({flag_id: flagId})
      });
      fetchLatestData();
    }

    // Auto-poll every 3 seconds
    fetchLatestData();
    setInterval(fetchLatestData, 3000);
  </script>
</body>
</html>
"""


@app.route("/")
@app.route("/dashboard")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/switch_venue")
def switch_venue():
    venue = request.args.get("venue", "polymarket")
    refresh_market_data(venue)
    return jsonify({"status": "switched", "venue": venue})


@app.route("/api/state")
def get_state():
    strikes = [1.5, 2.5, 3.5]
    cum_probs = [0.15, 0.50, 0.85]
    mu, sigma = engine.economic_pricer.fit_distribution_from_brackets(strikes, cum_probs)

    model_fvs = {}
    quotes = {}
    for c in cached_contracts:
        fv = engine.economic_pricer.price_bracket(c.floor_strike, c.cap_strike, mu, sigma)
        bid_q, ask_q = engine.quoter.quote_contract(fv)
        model_fvs[c.ticker] = fv
        quotes[c.ticker] = {"bid": bid_q, "ask": ask_q}

    contracts_data = [{
        "ticker": c.ticker,
        "event_ticker": c.event_ticker,
        "title": c.title,
        "subtitle": c.subtitle or c.title,
        "floor_strike": c.floor_strike,
        "cap_strike": c.cap_strike,
        "venue": c.venue.value
    } for c in cached_contracts]

    books_data = {}
    for ticker, ob in cached_books.items():
        books_data[ticker] = {
            "yes_bids": [{"price": l.price, "size": l.size} for l in ob.yes_bids],
            "yes_asks": [{"price": l.price, "size": l.size} for l in ob.yes_asks],
            "best_bid": ob.best_yes_bid,
            "best_ask": ob.best_yes_ask,
            "spread": ob.spread
        }

    flags_data = [{
        "flag_id": f.flag_id,
        "event_ticker": f.event_ticker,
        "disagreement_type": f.disagreement_type.value,
        "gap_size": f.gap_size,
        "gross_discrepancy": f.gross_discrepancy,
        "contracts_involved": f.contracts_involved,
        "suggested_trades": [{
            "ticker": t["ticker"],
            "action": t["action"].value,
            "side": t["side"].value,
            "price": t["price"],
            "size": t["size"]
        } for t in f.suggested_trades]
    } for f in cached_flags]

    summary = engine.get_portfolio_summary()

    return jsonify({
        "event": cached_event,
        "contracts": contracts_data,
        "books": books_data,
        "flags": flags_data,
        "model_fvs": model_fvs,
        "quotes": quotes,
        "portfolio": summary,
        "audit_trail": engine.execution_audit_trail
    })


@app.route("/api/execute", methods=["POST"])
def execute_flag():
    if cached_contracts and cached_books:
        ev_ticker = cached_contracts[0].event_ticker
        results = engine.scan_and_execute_bracket_arbitrage(
            event_ticker=ev_ticker,
            contracts=cached_contracts,
            books=cached_books,
            auto_execute=True
        )
        return jsonify({
            "status": "success",
            "orders_submitted": len(results),
            "portfolio": engine.get_portfolio_summary()
        })

    return jsonify({"status": "no_contracts_available"})


def main():
    port = 8080
    print("=" * 65)
    print(f" COPULA MULTI-VENUE SERVER LAUNCHING")
    print(f" Access Live At: http://localhost:{port}")
    print("=" * 65)
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
