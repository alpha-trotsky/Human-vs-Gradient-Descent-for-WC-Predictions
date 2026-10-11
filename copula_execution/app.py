"""
Copula Live Multi-Asset Universe Execution Terminal.
Scans and streams:
- Macro & Central Banks (Fed Rate Cuts, CPI, GDP)
- Elections & Geopolitics (Midterms Senate/House, Cabinet, World Leaders)
- Crypto, Tech & IPOs (Airdrops, Token Timelines, OpenAI, Kraken)
- Climate & Commodities
Across Polymarket CLOB & Kalshi L2.
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
    MarketContract, OrderBook, OrderBookLevel, MarketVenue, ActionSide, ContractSide, EventKind
)
from copula_execution.ingestion import MarketIngestion
from copula_execution.polymarket import PolymarketIngestion
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.risk_manager import RiskManager, RiskLimits, FeeCalculator
from copula_execution.gateway import PaperExecutionGateway
from copula_execution.engine import ExecutionEngine

app = Flask(__name__)

kalshi_ingestion = MarketIngestion()
poly_ingestion = PolymarketIngestion()
gateway = PaperExecutionGateway(latency_ms=8.0)
limits = RiskLimits(
    max_total_exposure_dollars=5000.0,
    max_event_exposure_dollars=1000.0,
    max_contract_order_dollars=200.0,
    min_net_edge_dollars=0.015,
    max_daily_loss_dollars=300.0
)
risk_manager = RiskManager(limits)
scanner = CoherenceScanner(min_edge=0.015)
engine = ExecutionEngine(gateway=gateway, risk_manager=risk_manager, scanner=scanner)

# Active state
current_category = "all"
current_venue = MarketVenue.POLYMARKET

EVENT_CATALOG = {
    "macro": [
        {"id": "51456", "title": "Fed Rate Cuts in 2026 (13 brackets)", "venue": "polymarket"},
        {"id": "KXGDPYEAR-36", "title": "US Real GDP Growth 2036 (14 brackets)", "venue": "kalshi"},
        {"id": "KXUSCPIYEAR-37FEB01", "title": "US Headline CPI Inflation Dec 2036", "venue": "kalshi"},
        {"id": "KXNOMGDPGROWTH-37JAN28", "title": "US Nominal GDP Growth 2036", "venue": "kalshi"}
    ],
    "elections": [
        {"id": "32228", "title": "Balance of Power: 2026 Midterms", "venue": "polymarket"},
        {"id": "32224", "title": "Which Party Wins Senate 2026?", "venue": "polymarket"},
        {"id": "32225", "title": "Which Party Wins House 2026?", "venue": "polymarket"},
        {"id": "30829", "title": "Democratic Presidential Nominee 2028", "venue": "polymarket"},
        {"id": "16263", "title": "Macron Out as President by Date", "venue": "polymarket"}
    ],
    "crypto_tech": [
        {"id": "27831", "title": "Hyperliquid Airdrop by Date (6 brackets)", "venue": "polymarket"},
        {"id": "16183", "title": "Kraken IPO by Date (7 brackets)", "venue": "polymarket"},
        {"id": "25009", "title": "OpenAI Consumer Hardware by Date", "venue": "polymarket"},
        {"id": "25930", "title": "Pump.fun Airdrop Timeline", "venue": "polymarket"}
    ]
}

active_event_meta = EVENT_CATALOG["macro"][0]
cached_contracts = []
cached_books = {}
cached_flags = []


def load_event(event_meta):
    global active_event_meta, cached_contracts, cached_books, cached_flags
    active_event_meta = event_meta
    cached_contracts = []
    cached_books = {}

    venue = event_meta["venue"]
    ev_id = event_meta["id"]

    if venue == "polymarket":
        try:
            contracts = poly_ingestion.fetch_event_markets(ev_id)
            cached_contracts = contracts
            for c in cached_contracts[:40]:
                ob = poly_ingestion.fetch_order_book(c.ticker)
                if ob:
                    cached_books[c.ticker] = ob
        except Exception as e:
            print(f"Error loading Polymarket event {ev_id}: {e}")
    else:
        try:
            contracts = kalshi_ingestion.fetch_event_markets(ev_id)
            cached_contracts = contracts
            for c in cached_contracts[:40]:
                ob = kalshi_ingestion.fetch_order_book(c.ticker)
                if ob:
                    cached_books[c.ticker] = ob
        except Exception as e:
            print(f"Error loading Kalshi event {ev_id}: {e}")

    # Re-scan coherence
    if cached_contracts and cached_books:
        sum_flags = scanner.check_mutually_exclusive_sum(ev_id, cached_contracts, cached_books)
        mono_flags = scanner.check_monotonicity(ev_id, cached_contracts, cached_books)
        cached_flags = sum_flags + mono_flags
    else:
        cached_flags = []


# Initial load
load_event(active_event_meta)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Copula // Multi-Asset Execution Terminal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #070a12; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
  </style>
</head>
<body class="bg-[#060911] text-[#94a3b8] p-3 text-xs mono antialiased select-none min-h-screen">

  <!-- Top App Bar -->
  <header class="flex items-center justify-between border-b border-[#182234] pb-2.5 mb-3">
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="font-bold text-white tracking-widest text-sm">COPULA</span>
        <span class="text-[#475569]">/ MULTI-ASSET ENGINE</span>
      </div>

      <!-- Sector Badges -->
      <div class="flex items-center border border-[#1e293b] rounded bg-[#0b101b] overflow-hidden ml-3">
        <button onclick="filterCategory('macro')" class="px-2.5 py-1 text-[11px] font-semibold text-white bg-[#1e293b] hover:bg-[#334155]" id="btn-cat-macro">MACRO & FED</button>
        <button onclick="filterCategory('elections')" class="px-2.5 py-1 text-[11px] text-[#64748b] hover:text-white" id="btn-cat-elections">ELECTIONS & GEOPOLITICS</button>
        <button onclick="filterCategory('crypto_tech')" class="px-2.5 py-1 text-[11px] text-[#64748b] hover:text-white" id="btn-cat-crypto">CRYPTO & TECH</button>
      </div>
    </div>

    <!-- Right Telemetry -->
    <div class="flex items-center gap-4 text-[11px]">
      <div>UNIVERSE: <span class="text-white font-semibold">45 ACTIVE FAMILIES</span></div>
      <div>SESSION PNL: <span class="text-emerald-400 font-semibold" id="head-pnl">$0.00</span></div>
      <div>EXPOSURE: <span class="text-white font-semibold" id="head-exposure">$0.00</span> / $5,000</div>
      <button onclick="fetchLatestData()" class="px-2 py-0.5 rounded border border-[#334155] bg-[#0b101b] hover:bg-[#1e293b] text-white">SYNC</button>
    </div>
  </header>

  <!-- Event Selector Strip -->
  <div class="flex items-center gap-2 mb-3 overflow-x-auto pb-1" id="event-selector-bar">
    <!-- Populated by JS -->
  </div>

  <!-- Primary Market Desk -->
  <div class="grid grid-cols-12 gap-3">

    <!-- Center L2 Matrix (8 cols) -->
    <div class="col-span-8 space-y-3">

      <!-- Orderbook Matrix -->
      <div class="border border-[#182234] bg-[#080d17] rounded p-3">
        <div class="flex items-start justify-between gap-3 pb-2 mb-2 border-b border-[#182234]">
          <div class="flex items-center gap-2 shrink-0">
            <span class="font-bold text-white text-[13px]" id="current-event-title">-</span>
            <span class="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-[#182234] text-cyan-400" id="current-event-venue">VENUE</span>
          </div>
          <span class="text-[10px] text-[#475569] text-right" id="model-label">MODEL: -</span>
        </div>

        <div class="overflow-x-auto max-h-96 overflow-y-auto">
          <table class="w-full text-left">
            <thead class="sticky top-0 bg-[#080d17] border-b border-[#182234] text-[#475569] text-[10px]">
              <tr>
                <th class="py-1">CONTRACT / PROPOSITION</th>
                <th>MODEL FV</th>
                <th>BEST BID</th>
                <th>BEST ASK</th>
                <th>SPREAD</th>
                <th>DERIVED QUOTE</th>
                <th>EDGE (FV - ASK)</th>
              </tr>
            </thead>
            <tbody id="book-rows" class="divide-y divide-[#182234]/50">
              <!-- Rendered via JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Arbitrage Signals -->
      <div class="border border-[#182234] bg-[#080d17] rounded p-3">
        <div class="flex items-center justify-between pb-2 mb-2 border-b border-[#182234]">
          <span class="font-semibold text-white tracking-wide">COHERENCE SCANNER FLAGS (MUTUAL EXCLUSIVITY & MONOTONICITY)</span>
          <span class="text-[10px] text-amber-400 font-bold" id="flags-count-badge">0 DISCREPANCIES</span>
        </div>
        <div id="flags-list" class="space-y-2">
          <!-- Rendered via JS -->
        </div>
      </div>

    </div>

    <!-- Right Controls & Audit (4 cols) -->
    <div class="col-span-4 space-y-3">

      <!-- Event Info & Basket Sum -->
      <div class="border border-[#182234] bg-[#080d17] rounded p-3">
        <div class="font-bold text-white pb-2 mb-2 border-b border-[#182234] flex justify-between">
          <span>EVENT COHERENCE STATS</span>
          <span class="text-[#475569] text-[10px]" id="stat-kind">-</span>
        </div>
        <div class="space-y-1.5 text-[11px]">
          <div class="flex justify-between"><span class="text-[#64748b]">Total Proposition Legs:</span><span class="text-white font-semibold" id="stat-legs">0</span></div>
          <div class="flex justify-between"><span class="text-[#64748b]">Basket Ask Sum:</span><span class="text-rose-400 font-semibold" id="stat-ask-sum">0.0000</span></div>
          <div class="flex justify-between"><span class="text-[#64748b]">Basket Bid Sum:</span><span class="text-emerald-400 font-semibold" id="stat-bid-sum">0.0000</span></div>
          <div class="flex justify-between"><span class="text-[#64748b]">Venue Fee Structure:</span><span class="text-cyan-400 font-semibold text-right" id="stat-fee">-</span></div>
          <div class="pt-2 border-t border-[#182234] flex justify-between text-[11px]">
            <span class="text-slate-300 font-medium">Implied Arbitrage Status:</span>
            <span id="stat-arb-status" class="font-bold text-slate-400">-</span>
          </div>
        </div>
      </div>

      <!-- Execution Log -->
      <div class="border border-[#182234] bg-[#080d17] rounded p-3">
        <div class="font-bold text-white pb-2 mb-2 border-b border-[#182234] flex justify-between">
          <span>TRADE EXECUTION JOURNAL</span>
          <span class="text-emerald-400 text-[10px]">SIMULATED VWAP ROUTER</span>
        </div>
        <div id="audit-list" class="space-y-1.5 max-h-72 overflow-y-auto text-[11px]">
          <!-- Rendered via JS -->
        </div>
      </div>

    </div>

  </div>

  <script>
    let activeCat = "macro";
    let eventCatalog = {
      "macro": [
        {"id": "51456", "title": "Fed Rate Cuts 2026", "venue": "polymarket"},
        {"id": "KXGDPYEAR-36", "title": "US Real GDP 2036", "venue": "kalshi"},
        {"id": "KXUSCPIYEAR-37FEB01", "title": "Headline CPI Dec 2036", "venue": "kalshi"},
        {"id": "KXNOMGDPGROWTH-37JAN28", "title": "Nominal GDP 2036", "venue": "kalshi"}
      ],
      "elections": [
        {"id": "32228", "title": "2026 Midterms Balance of Power", "venue": "polymarket"},
        {"id": "32224", "title": "Senate Control 2026", "venue": "polymarket"},
        {"id": "32225", "title": "House Control 2026", "venue": "polymarket"},
        {"id": "30829", "title": "Dem Nominee 2028", "venue": "polymarket"},
        {"id": "16263", "title": "Macron Exit Timeline", "venue": "polymarket"}
      ],
      "crypto_tech": [
        {"id": "27831", "title": "Hyperliquid Airdrop Timeline", "venue": "polymarket"},
        {"id": "16183", "title": "Kraken IPO Timeline", "venue": "polymarket"},
        {"id": "25009", "title": "OpenAI Consumer Device", "venue": "polymarket"},
        {"id": "25930", "title": "Pump.fun Airdrop", "venue": "polymarket"}
      ]
    };

    function filterCategory(cat) {
      activeCat = cat;
      ['macro', 'elections', 'crypto'].forEach(c => {
        const btn = document.getElementById('btn-cat-' + c);
        if (btn) btn.className = 'px-2.5 py-1 text-[11px] text-[#64748b] hover:text-white';
      });
      const actBtn = document.getElementById('btn-cat-' + (cat === 'crypto_tech' ? 'crypto' : cat));
      if (actBtn) actBtn.className = 'px-2.5 py-1 text-[11px] font-semibold text-white bg-[#1e293b]';
      renderEventBar();
      selectEvent(eventCatalog[cat][0]);
    }

    function renderEventBar() {
      const bar = document.getElementById('event-selector-bar');
      bar.innerHTML = eventCatalog[activeCat].map((ev, i) => `
        <button onclick='selectEvent(${JSON.stringify(ev)})' class="px-2.5 py-1 rounded border border-[#1e293b] bg-[#0b101b] hover:bg-[#1e293b] text-white text-[11px] whitespace-nowrap">
          ${ev.title} <span class="text-[9px] text-[#64748b] ml-1 uppercase">[${ev.venue}]</span>
        </button>
      `).join('');
    }

    async function selectEvent(ev) {
      await fetch('/api/select_event', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(ev)
      });
      fetchLatestData();
    }

    async function fetchLatestData() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();

        document.getElementById('current-event-title').innerText = data.event ? data.event.title : '-';
        document.getElementById('current-event-venue').innerText = data.event ? data.event.venue.toUpperCase() : '-';
        document.getElementById('head-pnl').innerText = '$' + data.portfolio.net_pnl_after_fees.toFixed(2);
        document.getElementById('head-exposure').innerText = '$' + data.exposure.toFixed(2);
        document.getElementById('stat-legs').innerText = data.contracts.length;
        document.getElementById('flags-count-badge').innerText = data.flags.length + ' DISCREPANCIES';

        // Basket stats are computed server-side: they only mean something for a mutually exclusive set
        const bk = data.basket;
        document.getElementById('stat-ask-sum').innerText = bk.ask_sum === null ? '-' : bk.ask_sum.toFixed(4) + ' (' + bk.n_asks + '/' + bk.n_legs + ')';
        document.getElementById('stat-bid-sum').innerText = bk.bid_sum === null ? '-' : bk.bid_sum.toFixed(4) + ' (' + bk.n_bids + '/' + bk.n_legs + ')';
        document.getElementById('stat-fee').innerText = bk.fee;
        document.getElementById('stat-kind').innerText = bk.kind.toUpperCase() + (bk.partition ? (bk.exhaustive ? ' / EXHAUSTIVE' : ' / NOT PROVEN EXHAUSTIVE') : '');
        const arbStatus = document.getElementById('stat-arb-status');
        arbStatus.innerText = bk.status;
        arbStatus.className = 'font-bold ' + (bk.status.startsWith('UNDERPRICED') ? 'text-emerald-400'
          : bk.status.startsWith('OVERPRICED') ? 'text-amber-400' : 'text-[#64748b]');

        const pr = data.pricing;
        document.getElementById('model-label').innerText = 'MODEL: ' + pr.method
          + (pr.rmse !== null ? ' | fit rmse ' + pr.rmse.toFixed(3) : '')
          + (pr.n_anchors !== null ? ' | ' + pr.n_anchors + ' anchors' : '')
          + (pr.warnings.length ? ' | ' + pr.warnings[0] : '');

        // Render book rows. A missing fair value is shown as "-", never as a placeholder number.
        const tbody = document.getElementById('book-rows');
        tbody.innerHTML = data.contracts.map(c => {
          const b = data.books[c.ticker] || {};
          const m = data.model_fvs[c.ticker] || {};
          const q = data.quotes[c.ticker] || {};
          const dp = c.venue === 'polymarket' ? 3 : 2;
          const num = (x, d) => (typeof x === 'number') ? x.toFixed(d) : '-';
          const ask = (typeof b.best_ask === 'number') ? b.best_ask : null;
          const bid = (typeof b.best_bid === 'number') ? b.best_bid : null;
          const spread = (ask !== null && bid !== null) ? (ask - bid).toFixed(3) : '-';
          const fvTxt = (typeof m.fv === 'number')
            ? m.fv.toFixed(3) + ' <span class="text-[#475569]">&plusmn;' + (m.uncertainty || 0).toFixed(3) + (m.anchor ? ' A' : '') + '</span>' : '-';
          // Edge only when the fair value is an independent model view (not a rescaled market mid)
          const showEdge = m.has_view && typeof m.fv === 'number' && ask !== null;
          const edge = showEdge ? (m.fv - ask) : null;
          // highlight only when the edge is bigger than the model's own error bar
          const edgeColor = (edge !== null && edge > (m.uncertainty || 0)) ? 'text-emerald-400 font-semibold' : 'text-[#64748b]';
          const qb = (q.bid === null || q.bid === undefined) ? '-' : q.bid.toFixed(dp);
          const qa = (q.ask === null || q.ask === undefined) ? '-' : q.ask.toFixed(dp);

          return `
            <tr class="hover:bg-[#0c121f] transition" title="${m.note || m.source || ''}">
              <td class="py-1.5 font-medium text-slate-200 max-w-sm truncate">${c.subtitle || c.title}</td>
              <td class="text-cyan-400 font-medium">${fvTxt}</td>
              <td class="text-emerald-400">${num(bid, dp)}</td>
              <td class="text-rose-400">${num(ask, dp)}</td>
              <td class="text-[#94a3b8]">${spread}</td>
              <td class="text-slate-300 font-medium">${qb} / ${qa}</td>
              <td class="${edgeColor}">${edge === null ? '-' : edge.toFixed(3)}</td>
            </tr>
          `;
        }).join('');

        // Render Flags
        const flagsDiv = document.getElementById('flags-list');
        if (data.flags.length === 0) {
          flagsDiv.innerHTML = '<div class="text-[#475569] text-center py-2.5">No mathematical coherence violations present on this book.</div>';
        } else {
          flagsDiv.innerHTML = data.flags.map(f => `
            <div class="border border-amber-500/30 bg-amber-950/15 p-2 rounded flex items-center justify-between">
              <div>
                <span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-bold text-[10px] mr-2">${f.disagreement_type.toUpperCase()}</span>
                <span class="text-white font-semibold">${f.event_ticker}</span>
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
          auditDiv.innerHTML = '<div class="text-[#475569] text-center py-2">No fills recorded this session.</div>';
        } else {
          auditDiv.innerHTML = data.audit_trail.slice(-8).reverse().map(a => `
            <div class="p-1.5 rounded bg-[#0b101b] border border-[#182234]">
              <div class="flex justify-between text-[#64748b]"><span>${a.order_id}</span><span class="text-emerald-400 font-semibold">${a.status}</span></div>
              <div class="text-slate-200 mt-0.5 font-medium">FILL ${a.filled_count} @ $${a.fill_price.toFixed(4)}</div>
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

    renderEventBar();
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


@app.route("/api/select_event", methods=["POST"])
def select_event():
    ev_meta = request.get_json() or {}
    load_event(ev_meta)
    return jsonify({"status": "loaded", "event": active_event_meta})


@app.route("/api/state")
def get_state():
    # Everything below is derived from the active event's own contracts and books.
    pricing = engine.price_event(cached_contracts, cached_books)

    model_fvs = {}
    quotes = {}
    for c in cached_contracts:
        fv = pricing.fair_values.get(c.ticker)
        if fv is None or fv.fv is None:
            model_fvs[c.ticker] = {"fv": None, "uncertainty": None, "source": "unpriced", "has_view": False,
                                   "anchor": False, "note": fv.note if fv else ""}
            quotes[c.ticker] = {"bid": None, "ask": None}
            continue
        bid_q, ask_q = engine.quoter.quote_fair_value(fv, c)
        model_fvs[c.ticker] = {"fv": fv.fv, "uncertainty": fv.uncertainty, "source": fv.source,
                               "has_view": fv.has_model_view, "anchor": fv.is_anchor, "note": fv.note}
        quotes[c.ticker] = {"bid": bid_q, "ask": ask_q}

    partition = bool(cached_contracts) and all(c.event_kind == EventKind.PARTITION for c in cached_contracts)
    exhaustive = partition and all(c.exhaustive for c in cached_contracts)
    asks = [ob.best_yes_ask for ob in cached_books.values() if ob.best_yes_ask is not None]
    bids = [ob.best_yes_bid for ob in cached_books.values() if ob.best_yes_bid is not None]
    if not partition:
        status = "N/A - legs are not a mutually exclusive set"
    elif any(f.flag_id.startswith("arb-sum-under") for f in cached_flags):
        status = "UNDERPRICED BASKET (BUY ARB)"
    elif any(f.flag_id.startswith("arb-sum-over") for f in cached_flags):
        status = "OVERPRICED BASKET (SELL ARB)"
    else:
        status = "NO BASKET ARBITRAGE" + ("" if exhaustive else " (buy-side disabled: not proven exhaustive)")
    basket = {
        "kind": cached_contracts[0].event_kind.value if cached_contracts else "unknown",
        "partition": partition, "exhaustive": exhaustive,
        "n_legs": len(cached_contracts), "n_asks": len(asks), "n_bids": len(bids),
        "ask_sum": round(sum(asks), 4) if asks else None,
        "bid_sum": round(sum(bids), 4) if bids else None,
        "fee": FeeCalculator.describe(cached_contracts[0]) if cached_contracts else "-",
        "status": status,
    }
    pricing_meta = {
        "method": pricing.method,
        "rmse": pricing.params.get("rmse"),
        "n_anchors": pricing.params.get("n_anchors"),
        "warnings": pricing.warnings,
    }
    exposure = sum(p.cost_basis for p in engine.risk_manager.positions.values())

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
        "event": active_event_meta,
        "contracts": contracts_data,
        "books": books_data,
        "flags": flags_data,
        "model_fvs": model_fvs,
        "quotes": quotes,
        "basket": basket,
        "pricing": pricing_meta,
        "exposure": exposure,
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
    print(f"COPULA MULTI-ASSET TERMINAL: http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
