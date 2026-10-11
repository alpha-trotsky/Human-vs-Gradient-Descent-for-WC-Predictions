"""
Real-time Ingestion Layer and Order Book Monitor for Kalshi and Prediction Markets.

Fetches open events, markets, order books, and produces structured OrderBook snapshots.
"""
import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Generator

from copula_execution.types import OrderBook, OrderBookLevel, MarketVenue, MarketContract


class MarketIngestion:
    """
    Client for Kalshi's public trade-api/v2 endpoints.
    Fetches open events, brackets, and full order books.
    """

    def __init__(self, base_url: str = "https://external-api.kalshi.com/trade-api/v2"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "copula-execution-ingestion/1.0"

    def get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        for attempt in range(4):
            try:
                r = self.session.get(f"{self.base_url}{path}", params=params, timeout=12)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == 3:
                    return None
                time.sleep(1.0)
        return None

    def fetch_open_events(self, limit: int = 100) -> List[dict]:
        """Fetch active open events."""
        res = self.get("/events", {"status": "open", "limit": limit})
        return (res or {}).get("events") or []

    def fetch_event_markets(self, event_ticker: str) -> List[MarketContract]:
        """Fetch all individual contracts under an event."""
        res = self.get("/markets", {"event_ticker": event_ticker, "limit": 100})
        raw_mkts = (res or {}).get("markets") or []
        contracts = []
        for m in raw_mkts:
            c = MarketContract(
                ticker=m.get("ticker"),
                event_ticker=m.get("event_ticker"),
                venue=MarketVenue.KALSHI,
                title=m.get("title", ""),
                subtitle=m.get("yes_sub_title") or m.get("subtitle"),
                floor_strike=float(m["floor_strike"]) if m.get("floor_strike") is not None else None,
                cap_strike=float(m["cap_strike"]) if m.get("cap_strike") is not None else None,
                strike_type=m.get("strike_type"),
                mutually_exclusive_group=bool(m.get("mutually_exclusive", False))
            )
            contracts.append(c)
        return contracts

    def fetch_order_book(self, ticker: str) -> Optional[OrderBook]:
        """Fetch L2 orderbook for a contract ticker."""
        res = self.get(f"/markets/{ticker}/orderbook")
        if not res:
            return None

        ob_data = res.get("orderbook_fp") or res.get("orderbook") or {}
        yes_raw = ob_data.get("yes_dollars") or []
        no_raw = ob_data.get("no_dollars") or []

        # Parse yes levels
        # In Kalshi: yes_dollars is [[price_str, size_str], ...] (bids)
        yes_bids = []
        for p_str, s_str in yes_raw:
            try:
                yes_bids.append(OrderBookLevel(price=float(p_str), size=float(s_str)))
            except (ValueError, TypeError):
                continue
        yes_bids.sort(key=lambda x: x.price, reverse=True)

        # In Kalshi: NO bids correspond to YES asks:
        # A NO bid at price p implies a YES ask at (1.0 - p)
        yes_asks = []
        no_bids = []
        for p_str, s_str in no_raw:
            try:
                no_p = float(p_str)
                no_s = float(s_str)
                no_bids.append(OrderBookLevel(price=no_p, size=no_s))
                # YES ask price is 1.0 - NO bid price
                yes_asks.append(OrderBookLevel(price=round(1.0 - no_p, 4), size=no_s))
            except (ValueError, TypeError):
                continue
        no_bids.sort(key=lambda x: x.price, reverse=True)
        yes_asks.sort(key=lambda x: x.price)

        return OrderBook(
            ticker=ticker,
            venue=MarketVenue.KALSHI,
            timestamp=datetime.now(timezone.utc),
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            no_bids=no_bids,
            no_asks=[]
        )
