"""
Core data types and interfaces for the Copula Execution Layer.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class MarketVenue(str, Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"
    SIMULATED = "simulated"


class ContractSide(str, Enum):
    YES = "yes"
    NO = "no"


class ActionSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DisagreementType(str, Enum):
    SUM_TO_ONE = "sum_to_one"                   # ME brackets sum exceeds 1 + fee or < 1 - fee
    MONOTONICITY = "monotonicity"               # P(Above X) < P(Above Y) when X < Y
    COPULA_ARBITRAGE = "copula_arbitrage"       # Model FV divergence beyond fee + threshold
    CROSS_EVENT_DISAGREEMENT = "cross_event"    # Linked event disagreement (e.g. CPI vs Fed)


@dataclass
class OrderBookLevel:
    price: float  # In dollars, e.g. 0.54
    size: float   # Quantity/contracts


@dataclass
class OrderBook:
    ticker: str
    venue: MarketVenue
    timestamp: datetime
    yes_bids: List[OrderBookLevel] = field(default_factory=list)  # Sorted descending by price
    yes_asks: List[OrderBookLevel] = field(default_factory=list)  # Sorted ascending by price
    no_bids: List[OrderBookLevel] = field(default_factory=list)
    no_asks: List[OrderBookLevel] = field(default_factory=list)
    last_price: Optional[float] = None
    volume: Optional[float] = None

    @property
    def best_yes_bid(self) -> Optional[float]:
        return self.yes_bids[0].price if self.yes_bids else None

    @property
    def best_yes_ask(self) -> Optional[float]:
        return self.yes_asks[0].price if self.yes_asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_yes_bid is not None and self.best_yes_ask is not None:
            return round(self.best_yes_ask - self.best_yes_bid, 4)
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_yes_bid is not None and self.best_yes_ask is not None:
            return round((self.best_yes_bid + self.best_yes_ask) / 2.0, 4)
        return self.last_price


@dataclass
class MarketContract:
    ticker: str
    event_ticker: str
    venue: MarketVenue
    title: str
    subtitle: Optional[str] = None
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None
    strike_type: Optional[str] = None
    mutually_exclusive_group: bool = False
    expiration_time: Optional[datetime] = None


@dataclass
class DisagreementFlag:
    flag_id: str
    event_ticker: str
    disagreement_type: DisagreementType
    contracts_involved: List[str]
    gap_size: float              # Net edge in dollars after fees
    gross_discrepancy: float     # Raw difference before fees
    model_fair_values: Dict[str, float]
    market_prices: Dict[str, Dict[str, float]] # ticker -> {bid, ask, mid}
    detected_at: datetime
    suggested_trades: List[Dict[str, Any]]     # List of {ticker, side, action, price, size, expected_edge}
    resolved_at: Optional[datetime] = None
    resolution_status: str = "open"            # open, converged, expired, discarded


@dataclass
class OrderRequest:
    order_id: str
    ticker: str
    venue: MarketVenue
    action: ActionSide          # BUY or SELL
    side: ContractSide          # YES or NO
    order_type: OrderType
    price: float                # In dollars (0.01 - 0.99)
    count: int                  # Number of contracts
    flag_id: Optional[str] = None
    client_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderResult:
    order_id: str
    client_order_id: Optional[str]
    status: OrderStatus
    filled_count: int = 0
    remaining_count: int = 0
    average_fill_price: float = 0.0
    fee_paid: float = 0.0
    exchange_order_id: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    ticker: str
    venue: MarketVenue
    yes_shares: int = 0
    no_shares: int = 0
    realized_pnl: float = 0.0
    cost_basis: float = 0.0
    fees_paid: float = 0.0
    last_mid_price: float = 0.0

    @property
    def net_exposure(self) -> float:
        # Net dollars at risk: in prediction markets each YES share pays $1 on YES, 0 on NO
        return float(self.yes_shares - self.no_shares)

    @property
    def unrealized_pnl(self) -> float:
        current_value = (self.yes_shares * self.last_mid_price) + (self.no_shares * (1.0 - self.last_mid_price))
        return round(current_value - self.cost_basis, 4)
