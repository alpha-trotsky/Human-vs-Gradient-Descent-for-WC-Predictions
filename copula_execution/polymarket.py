"""
Polymarket Ingestion Layer & CLOB Client.
Fetches Gamma API events, markets, and L2 order books from clob.polymarket.com.
"""
import json
import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from copula_execution.types import OrderBook, OrderBookLevel, MarketVenue, MarketContract
from copula_execution.classify import classify_polymarket_event


class PolymarketIngestion:
    """
    Connects to Polymarket Gamma API and CLOB Order Book API.
    """

    def __init__(self,
                 gamma_base: str = "https://gamma-api.polymarket.com",
                 clob_base: str = "https://clob.polymarket.com"):
        self.gamma_base = gamma_base
        self.clob_base = clob_base
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) copula-execution/1.0"

    def fetch_open_events(self, limit: int = 50) -> List[dict]:
        """Fetch active events from Gamma API."""
        try:
            r = self.session.get(
                f"{self.gamma_base}/events",
                params={"limit": limit, "closed": "false", "active": "true"},
                timeout=12
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"Error fetching Polymarket events: {e}")
        return []

    @staticmethod
    def _parse_time(s) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _load_list(raw) -> list:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except ValueError:
                return []
        return raw or []

    def contracts_from_event(self, ev: dict) -> List[MarketContract]:
        """
        Build MarketContracts for the *open, tradable* markets of a Gamma event and classify the
        event. Closed markets are dropped (they are resolved and have no book), the YES token is
        located through each market's `outcomes` list rather than assumed to be index 0, and the
        per-market fee schedule and tick size are copied from the venue.
        """
        raw_mkts = ev.get("markets") or []
        contracts: List[MarketContract] = []
        n_unmapped = 0
        for m in raw_mkts:
            if m.get("closed") or m.get("active") is False or m.get("acceptingOrders") is False:
                continue
            tokens = self._load_list(m.get("clobTokenIds"))
            outcomes = [str(o).lower() for o in self._load_list(m.get("outcomes"))]
            if "yes" in outcomes and len(tokens) == len(outcomes):
                yes_token = tokens[outcomes.index("yes")]
            else:
                n_unmapped += 1   # not a plain Yes/No market: cannot be mapped safely
                continue

            sched = m.get("feeSchedule") or {}
            contracts.append(MarketContract(
                ticker=yes_token,
                event_ticker=str(ev.get("id")),
                venue=MarketVenue.POLYMARKET,
                title=m.get("question", ""),
                subtitle=m.get("groupItemTitle") or m.get("question"),
                strike_type="binary",
                mutually_exclusive_group=bool(ev.get("negRisk")),
                expiration_time=self._parse_time(m.get("endDate")),
                tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
                fees_enabled=bool(m.get("feesEnabled", True)),
                fee_rate=float(sched["rate"]) if sched.get("rate") is not None else None,
                fee_exponent=float(sched.get("exponent", 1.0)),
            ))
        classify_polymarket_event(contracts, bool(ev.get("negRisk")),
                                  bool(ev.get("negRiskAugmented")), n_unmapped=n_unmapped)
        return contracts

    def fetch_event_markets(self, event_id: str) -> List[MarketContract]:
        """Fetch the open markets under a specific Polymarket event."""
        try:
            r = self.session.get(f"{self.gamma_base}/events/{event_id}", timeout=12)
            if r.status_code == 200:
                return self.contracts_from_event(r.json())
        except Exception as e:
            print(f"Error fetching event {event_id} from Polymarket: {e}")
        return []

    def fetch_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch L2 order book from Polymarket CLOB."""
        try:
            r = self.session.get(f"{self.clob_base}/book", params={"token_id": token_id}, timeout=10)
            if r.status_code != 200:
                return None

            data = r.json()
            raw_bids = data.get("bids") or []
            raw_asks = data.get("asks") or []

            # Parse bids (sorted descending)
            yes_bids = []
            for b in raw_bids:
                try:
                    p = float(b["price"])
                    s = float(b["size"])
                    yes_bids.append(OrderBookLevel(price=p, size=s))
                except (ValueError, KeyError, TypeError):
                    continue
            yes_bids.sort(key=lambda x: x.price, reverse=True)

            # Parse asks (sorted ascending)
            yes_asks = []
            for a in raw_asks:
                try:
                    p = float(a["price"])
                    s = float(a["size"])
                    yes_asks.append(OrderBookLevel(price=p, size=s))
                except (ValueError, KeyError, TypeError):
                    continue
            yes_asks.sort(key=lambda x: x.price)

            return OrderBook(
                ticker=token_id,
                venue=MarketVenue.POLYMARKET,
                timestamp=datetime.now(timezone.utc),
                yes_bids=yes_bids,
                yes_asks=yes_asks,
                last_price=float(data["last_trade_price"]) if data.get("last_trade_price") else None
            )
        except Exception as e:
            print(f"Error fetching CLOB book for {token_id}: {e}")
            return None
