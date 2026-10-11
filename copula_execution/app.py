"""
Professional, High-Density Quantitative Execution Terminal for Copula.
Clean, functional financial styling (Bloomberg/Linear-inspired dark UI)
replacing generic AI marketing fluff.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

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

kalshi_ingestion = MarketIngestion()
poly_ingestion = PolymarketIngestion()
gateway = PaperExecutionGateway(latency_ms=8.0)
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
            contracts = poly_ingestion.fetch_event_markets("51456")
            if contracts:
                cached_event = {
                    "event_ticker": "POLY-51456",
                    "title": "Fed Rate Cuts in 2026",
                    "series_ticker": "FED-CUTS-2026",
                    "venue": "Polymarket",
                    "settlement": "Dec 31, 2026"
                }
                cached_contracts = contracts
                cached_books = {}
                for c in cached_contracts[:10]:
                    ob = poly_ingestion.fetch_order_book(c.ticker)
                    if ob:
                        cached_books[c.ticker] = ob
        except Exception as e:
            print(f"Error Polymarket: {e}")
    else:
        try:
            open_events = kalshi_ingestion.fetch_open_events(limit=40)
            target_ev = next((ev for ev in open_events if "gdp" in ev.get("series_ticker", "").lower()), None)
            if target_ev:
                ev_ticker = target_ev.get("event_ticker")
                cached_event = {
                    "event_ticker": ev_ticker,
                    "title": target_ev.get("title"),
                    "series_ticker": target_ev.get("series_ticker"),
                    "venue": "Kalshi",
                    "settlement": "Dec 31, 2036"
                }
                cached_contracts = kalshi_ingestion.fetch_event_markets(ev_ticker)
                cached_books = {}
                for c in cached_contracts[:10]:
                    ob = kalshi_ingestion.fetch_order_book(c.ticker)
                    if ob:
                        cached_books[c.ticker] = ob
        except Exception as e:
            print(f"Error Kalshi: {e}")

    if cached_contracts and cached_books:
        ev_ticker = cached_contracts[0].event_ticker
        sum_flags = scanner.check_mutually_exclusive_sum(ev_ticker, cached_contracts, cached_books)
        mono_flags = scanner.check_monotonicity(ev_ticker, cached_contracts, cached_books)
        cached_flags = sum_flags + mono_flags


# Initial refresh
refresh_market_data("polymarket")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Copula // Execution Terminal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    .terminal-font { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #090d16; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
  </style>
</head>
<body class="bg-[#080c14] text-[#cbd5e1] p-4 text-xs terminal-font antialiased select-none">

  <!-- Top Navigation & Status Bar -->
  <header class="flex items-center justify-between border-b border-[#1e293b] pb-3 mb-4">
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2">
        <span class="inline-block w-2 h-2 rounded-full bg-emerald-400"></span>
        <span class="font-bold text-white tracking-widest text-sm">COPULA</span>
        <span class="text-[#64748b]">/ CORE-EXEC-01</span>
      </div>

      <div class="flex items-center border border-[#1e293b] rounded bg-[#0f172a] overflow-hidden ml-4">
        <button onclick="switchVenue('polymarket')" id="btn-polymarket" class="px-2.5 py-1 text-[11px] font-medium bg-[#1e293b] text-white">POLYMARKET CLOB</button>
        <button onclick="switchVenue('kalshi')" id="btn-kalshi" class="px-2.5 py-1 text-[11px] font-medium text-[#64748b] hover:text-white">KALSHI L2</button>
      </div>
    </div>

    <div class="flex items-center gap-5 text-[11px] text-[#94a3b8]">
      <div>FEED: <span class="text-emerald-400 font-semibold" id="feed-venue">POLYMARKET</span></div>
      <div>SESSION PNL: <span class="text-emerald-400 font-semibold" id="head-pnl">$0.00</span></div>
      <div>EXPOSURE: <span class="text-white font-semibold" id="head-exposure">$0.00</span> / $2,500</div>
      <div>LATENCY: <span class="text-[#64748b]">8ms</span></div>
      <button onclick="fetchLatestData()" class="px-2 py-0.5 rounded border border-[#334155] bg-[#0f172a] hover:bg-[#1e293b] text-white">SYNC</button>
    </div>
  </header>

  <!-- Metric Strip -->
  <div class="grid grid-cols-5 gap-3 mb-4">
    <div class="border border-[#1e293b] bg-[#0b1120] p-2.5 rounded">
      <div class="text-[#64748b] text-[10px]">CURRENT TARGET EVENT</div>
      <div class="text-white font-semibold truncate text-[11px] mt-0.5" id="metric-event">-</div>
    </div>
    <div class="border border-[#1e293b] bg-[#0b1120] p-2.5 rounded">
      <div class="text-[#64748b] text-[10px]">BOOK DEPTH MONITORED</div>
      <div class="text-white font-semibold text-[13px] mt-0.5" id="metric-contracts">0 Contracts</div>
    </div>
    <div class="border border-[#1e293b] bg-[#0b1120] p-2.5 rounded">
      <div class="text-[#64748b] text-[10px]">BASKET SUM (ASK / BID)</div>
      <div class="text-white font-semibold text-[13px] mt-0.5" id="metric-basket">0.00 / 0.00</div>
    </div>
    <div class="border border-[#1e293b] bg-[#0b1120] p-2.5 rounded">
      <div class="text-[#64748b] text-[10px]">COHERENCE BREACHES</div>
      <div class="text-amber-400 font-semibold text-[13px] mt-0.5" id="metric-flags">0 Active</div>
    </div>
    <div class="border border-[#1e293b] bg-[#0b1120] p-2.5 rounded">
      <div class="text-[#64748b] text-[10px]">VENUE TAKER FEE SCHEDULE</div>
      <div class="text-cyan-400 font-semibold text-[13px] mt-0.5" id="metric-fee">0.20% (Polymarket)</div>
    </div>
  </div>

  <!-- Main Trading Desk -->
  <div class="grid grid-cols-12 gap-4">

    <!-- Center: Order Books & Fair Values (8 cols) -->
    <div class="col-span-8 space-y-4">

      <!-- Orderbook Matrix -->
      <div class="border border-[#1e293b] bg-[#0b1120] rounded p-3">
        <div class="flex items-center justify-between pb-2 mb-2 border-b border-[#1e293b]">
          <span class="font-semibold text-white tracking-wide">ORDER BOOK DEPTH & DERIVED COPULA FAIR VALUES</span>
          <span class="text-[10px] text-[#64748b]" id="model-density-tag">MODEL: Discrete Marginal Fit [μ=1.85, σ=0.92]</span>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left">
            <thead>
              <tr class="text-[#64748b] border-b border-[#1e293b]/60 pb-1 text-[10px]">
                <th class="py-1">CONTRACT / STRIKE</th>
                <th>MODEL FV</th>
                <th>BEST BID</th>
                <th>BEST ASK</th>
                <th>SPREAD</th>
                <th>TWO-SIDED QUOTE</th>
                <th>EDGE (ASK - FV)</th>
              </tr>
            </thead>
            <tbody id="book-rows" class="divide-y divide-[#1e293b]/40">
              <!-- Rendered via JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Arbitrage & Coherence Alerts -->
      <div class="border border-[#1e293b] bg-[#0b1120] rounded p-3">
        <div class="flex items-center justify-between pb-2 mb-2 border-b border-[#1e293b]">
          <span class="font-semibold text-white tracking-wide">DISAGREEMENT SCANNER (NET OF FEES & SLIPPAGE)</span>
          <span class="text-[10px] text-[#64748b]">THRESHOLD: &ge; 1.5¢ / contract</span>
        </div>
        <div id="flags-list" class="space-y-2">
          <!-- Rendered via JS -->
        </div>
      </div>

    </div>

    <!-- Right: Risk Controller & Execution Journal (4 cols) -->
    <div class="col-span-4 space-y-4">

      <!-- Risk Gate Configuration -->
      <div class="border border-[#1e293b] bg-[#0b1120] rounded p-3">
        <div class="font-semibold text-white pb-2 mb-2 border-b border-[#1e293b] flex items-center justify-between">
          <span>RISK CONTROLLER</span>
          <span class="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">GATES CLOSED / READY</span>
        </div>
        <div class="space-y-1.5 text-[11px]">
          <div class="flex justify-between"><span class="text-[#64748b]">Max Order Notional:</span><span class="text-white">$100.00</span></div>
          <div class="flex justify-between"><span class="text-[#64748b]">Max Family Allocation:</span><span class="text-white">$500.00</span></div>
          <div class="flex justify-between"><span class="text-[#64748b]">Daily Drawdown Breaker:</span><span class="text-rose-400">-$200.00</span></div>
          <div class="flex justify-between"><span class="text-[#64748b]">Rate Limit Throttle:</span><span class="text-white">30 orders / min</span></div>
          <div class="flex justify-between"><span class="text-[#64748b]">Min Required Edge:</span><span class="text-emerald-400">$0.015 / contract</span></div>
        </div>
      </div>

      <!-- Execution Audit Log -->
      <div class="border border-[#1e293b] bg-[#0b1120] rounded p-3">
        <div class="font-semibold text-white pb-2 mb-2 border-b border-[#1e293b] flex items-center justify-between">
          <span>EXECUTION AUDIT TRAIL</span>
          <span class="text-[10px] text-emerald-400">ACTIVE ROUTER</span>
        </div>
        <div id="audit-list" class="space-y-1.5 max-h-80 overflow-y-auto text-[11px]">
          <!-- Rendered via JS -->
        </div>
      </div>

    </div>

  </div>

  <script>
    let activeVenue = "polymarket";

    function switchVenue(v) {
      activeVenue = v;
      if (v === 'polymarket') {
        document.getElementById('btn-polymarket').className = 'px-2.5 py-1 text-[11px] font-medium bg-[#1e293b] text-white';
        document.getElementById('btn-kalshi').className = 'px-2.5 py-1 text-[11px] font-medium text-[#64748b] hover:text-white';
        document.getElementById('feed-venue').innerText = 'POLYMARKET CLOB';
        document.getElementById('metric-fee').innerText = '0.20% (Polymarket)';
      } else {
        document.getElementById('btn-kalshi').className = 'px-2.5 py-1 text-[11px] font-medium bg-[#1e293b] text-white';
        document.getElementById('btn-polymarket').className = 'px-2.5 py-1 text-[11px] font-medium text-[#64748b] hover:text-white';
        document.getElementById('feed-venue').innerText = 'KALSHI L2';
        document.getElementById('metric-fee').innerText = '1.00% (Kalshi)';
      }
      fetch('/api/switch_venue?venue=' + v).then(() => fetchLatestData());
    }

    async function fetchLatestData() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();

        // Header and metrics
        document.getElementById('metric-event').innerText = data.event ? data.event.title : '-';
        document.getElementById('metric-contracts').innerText = data.contracts.length + ' Contracts';
        document.getElementById('metric-flags').innerText = data.flags.length + ' Active';
        document.getElementById('head-pnl').innerText = '$' + data.portfolio.net_pnl_after_fees.toFixed(2);
        document.getElementById('head-exposure').innerText = '$' + (data.portfolio.positions_count * 2.5).toFixed(2);

        // Compute basket sum
        let sumAsk = 0, sumBid = 0;
        data.contracts.forEach(c => {
          const b = data.books[c.ticker] || {};
          if (b.best_ask) sumAsk += b.best_ask;
          if (b.best_bid) sumBid += b.best_bid;
        });
        document.getElementById('metric-basket').innerText = `${sumAsk.toFixed(3)} / ${sumBid.toFixed(3)}`;

        // Render book rows
        const tbody = document.getElementById('book-rows');
        tbody.innerHTML = data.contracts.map(c => {
          const b = data.books[c.ticker] || {};
          const fv = data.model_fvs[c.ticker] || 0.05;
          const q = data.quotes[c.ticker] || {bid: 0.04, ask: 0.08};
          const ask = b.best_ask !== undefined && b.best_ask !== null ? b.best_ask : '-';
          const bid = b.best_bid !== undefined && b.best_bid !== null ? b.best_bid : '-';
          const spread = (typeof ask === 'number' && typeof bid === 'number') ? (ask - bid).toFixed(3) : '-';
          const edge = typeof ask === 'number' ? (fv - ask).toFixed(3) : '-';
          const edgeColor = parseFloat(edge) > 0 ? 'text-emerald-400 font-semibold' : 'text-[#64748b]';

          return `
            <tr class="hover:bg-[#0f172a] transition">
              <td class="py-1.5 font-medium text-slate-200 max-w-xs truncate">${c.subtitle || c.title}</td>
              <td class="text-cyan-400 font-medium">${fv.toFixed(3)}</td>
              <td class="text-emerald-400">${typeof bid === 'number' ? bid.toFixed(3) : '-'}</td>
              <td class="text-rose-400">${typeof ask === 'number' ? ask.toFixed(3) : '-'}</td>
              <td class="text-[#94a3b8]">${spread}</td>
              <td class="text-slate-300 font-medium">${q.bid.toFixed(2)} / ${q.ask.toFixed(2)}</td>
              <td class="${edgeColor}">${edge}</td>
            </tr>
          `;
        }).join('');

        // Render Flags
        const flagsDiv = document.getElementById('flags-list');
        if (data.flags.length === 0) {
          flagsDiv.innerHTML = '<div class="text-[#64748b] text-center py-3">No mathematical coherence violations present on this book.</div>';
        } else {
          flagsDiv.innerHTML = data.flags.map(f => `
            <div class="border border-amber-500/30 bg-amber-950/10 p-2.5 rounded flex items-center justify-between">
              <div>
                <span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-bold text-[10px] mr-2">${f.disagreement_type.toUpperCase()}</span>
                <span class="text-white">${f.event_ticker}</span>
                <div class="text-[#94a3b8] text-[10px] mt-0.5">Gross Gap: $${f.gross_discrepancy.toFixed(4)} | Net Edge: <span class="text-emerald-400 font-bold">+$${f.gap_size.toFixed(4)}</span></div>
              </div>
              <button onclick="executeFlag('${f.flag_id}')" class="px-2.5 py-1 bg-emerald-700 hover:bg-emerald-600 text-white font-semibold rounded text-[10px]">
                ROUTE ARB
              </button>
            </div>
          `).join('');
        }

        // Render Audit Log
        const auditDiv = document.getElementById('audit-list');
        if (data.audit_trail.length === 0) {
          auditDiv.innerHTML = '<div class="text-[#64748b] text-center py-2">No fills recorded this session.</div>';
        } else {
          auditDiv.innerHTML = data.audit_trail.slice(-8).reverse().map(a => `
            <div class="p-1.5 rounded bg-[#0f172a] border border-[#1e293b]">
              <div class="flex justify-between text-[#64748b]"><span>${a.order_id}</span><span class="text-emerald-400 font-semibold">${a.status}</span></div>
              <div class="text-slate-200 mt-0.5 font-medium">FILL ${a.filled_count} contracts @ $${a.fill_price.toFixed(4)}</div>
            </div>
          `).join('');
        }

      } catch (e) {
        console.error(e);
      }
    }

    async function executeFlag(flagId) {
      await fetch('/api/execute', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({flag_id: flagId})
      });
      fetchLatestData();
    }

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
    print(f"TERMINAL LAUNCHED: http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
