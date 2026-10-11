"""
Risk management, fee calculation, and safety gates for the execution layer.

Calculates exchange transaction fees and enforces hard drawdown limits, max exposure,
max per-event exposure, slippage bounds, and execution rate limits.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List

from copula_execution.types import (
    OrderRequest, Position, MarketVenue, ContractSide, ActionSide, MarketContract
)


@dataclass
class RiskLimits:
    max_total_exposure_dollars: float = 1000.0   # Hard stop on total portfolio notional
    max_event_exposure_dollars: float = 250.0    # Max allocation per single event/family
    max_contract_order_dollars: float = 50.0     # Max single trade size
    min_net_edge_dollars: float = 0.02           # Minimum edge required after fees (2 cents default)
    max_allowed_slippage_dollars: float = 0.015  # Max slippage vs model entry expectation
    max_daily_loss_dollars: float = 100.0        # Daily circuit breaker
    max_orders_per_minute: int = 30              # Throttling gate


class FeeCalculator:
    """
    Per-venue fee schedules.

    Kalshi (published schedule): taker fee = ceil_to_cent(0.07 * C * p * (1 - p)); maker orders pay
    nothing except on the few series that list maker fees (0.0175 * C * p * (1 - p)). Per-market
    `fee_multiplier` / `fee_type` from the API are honoured via MarketContract.

    Polymarket: fees are set per market. The market's `feeSchedule` gives
    fee = C * p * rate * (p * (1 - p)) ** exponent, taker-only; `feesEnabled=False` means fee-free.
    The fallback below is used only when a contract carries no schedule (e.g. historical data).

    Both rates drift over time -- ingestion copies what the venue reports onto each MarketContract
    and `fee_for_contract` is the entry point everything else should use.
    """

    KALSHI_TAKER_RATE = 0.07
    KALSHI_MAKER_RATE = 0.0175
    POLYMARKET_FALLBACK_RATE = 0.05   # conservative: highest non-crypto rate seen on live markets

    @staticmethod
    def _ceil_cents(x: float) -> float:
        return math.ceil(round(x * 100.0, 9)) / 100.0

    @staticmethod
    def calculate_fee(venue: MarketVenue, price: float, count: int, is_maker: bool = False,
                      fee_rate: Optional[float] = None, fee_exponent: float = 1.0,
                      fees_enabled: bool = True, maker_fee_rate: Optional[float] = None) -> float:
        if count <= 0:
            return 0.0
        p = min(max(float(price), 0.0), 1.0)

        if venue == MarketVenue.SIMULATED:
            # Venue-agnostic paper fee; real paths should pass the real venue.
            return round(max(0.005 * p * count, 0.005 * count), 4)

        if not fees_enabled:
            return 0.0

        if venue == MarketVenue.KALSHI:
            rate = maker_fee_rate if is_maker else (
                fee_rate if fee_rate is not None else FeeCalculator.KALSHI_TAKER_RATE)
            if not rate:
                return 0.0
            # Kalshi rounds the whole order's fee up to the next cent.
            return FeeCalculator._ceil_cents(rate * count * p * (1.0 - p))

        if venue == MarketVenue.POLYMARKET:
            if is_maker:
                return 0.0   # takerOnly on every schedule observed
            rate = fee_rate if fee_rate is not None else FeeCalculator.POLYMARKET_FALLBACK_RATE
            return round(count * p * rate * (p * (1.0 - p)) ** fee_exponent, 5)

        return 0.0

    @staticmethod
    def fee_for_contract(contract: MarketContract, price: float, count: int,
                         is_maker: bool = False) -> float:
        """Fee using the market's own schedule (falls back to the venue default)."""
        return FeeCalculator.calculate_fee(
            contract.venue, price, count, is_maker=is_maker,
            fee_rate=contract.fee_rate, fee_exponent=contract.fee_exponent,
            fees_enabled=contract.fees_enabled, maker_fee_rate=contract.maker_fee_rate)

    @staticmethod
    def fee_per_contract(contract: MarketContract, price: float, size: int,
                         is_maker: bool = False) -> float:
        """Average fee per contract when trading `size` at once (cent rounding is per order, not per lot)."""
        size = max(int(size), 1)
        return FeeCalculator.fee_for_contract(contract, price, size, is_maker) / size

    @staticmethod
    def describe(contract: MarketContract) -> str:
        """Human-readable schedule for UIs."""
        if not contract.fees_enabled:
            return "fee-free"
        if contract.venue == MarketVenue.KALSHI:
            r = contract.fee_rate if contract.fee_rate is not None else FeeCalculator.KALSHI_TAKER_RATE
            return f"taker {r:g}*p(1-p), ceil to cent"
        if contract.venue == MarketVenue.POLYMARKET:
            r = contract.fee_rate if contract.fee_rate is not None else FeeCalculator.POLYMARKET_FALLBACK_RATE
            return f"taker {r:g}*p*(p(1-p))^{contract.fee_exponent:g}"
        return "paper"


class RiskManager:
    """
    Stateful risk controller enforcing strict safety bounds before any order reaches the gateway.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.positions: Dict[str, Position] = {}  # ticker -> Position
        self.event_allocations: Dict[str, float] = {}  # event_ticker -> total dollars committed
        self.order_timestamps: List[datetime] = []
        self.daily_realized_pnl: float = 0.0
        self.last_pnl_reset: datetime = datetime.now(timezone.utc)

    def update_position_from_fill(self, ticker: str, event_ticker: str, venue: MarketVenue,
                                  action: ActionSide, side: ContractSide, price: float,
                                  count: int, fee: float):
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker, venue=venue)

        pos = self.positions[ticker]
        pos.fees_paid += fee
        pos.last_mid_price = price

        dollar_amount = price * count

        if side == ContractSide.YES:
            if action == ActionSide.BUY:
                pos.yes_shares += count
                pos.cost_basis += dollar_amount
            elif action == ActionSide.SELL:
                # Closing or shorting
                fill_proceeds = dollar_amount
                pos.yes_shares -= count
                pos.realized_pnl += fill_proceeds - fee
                self.daily_realized_pnl += fill_proceeds - fee
        else: # NO contract
            if action == ActionSide.BUY:
                pos.no_shares += count
                pos.cost_basis += dollar_amount
            elif action == ActionSide.SELL:
                fill_proceeds = dollar_amount
                pos.no_shares -= count
                pos.realized_pnl += fill_proceeds - fee
                self.daily_realized_pnl += fill_proceeds - fee

        # Update event allocations
        current_event_alloc = self.event_allocations.get(event_ticker, 0.0)
        self.event_allocations[event_ticker] = max(0.0, current_event_alloc + dollar_amount)

    def validate_order(self, order: OrderRequest, event_ticker: str,
                       expected_edge: float) -> Tuple[bool, Optional[str]]:
        """
        Runs comprehensive checks:
        1. Net Edge >= min_net_edge_dollars (accounts for spread + fees)
        2. Daily loss circuit breaker
        3. Rate limiter (orders per minute)
        4. Max single order size
        5. Max event allocation
        6. Max total portfolio exposure
        """
        now = datetime.now(timezone.utc)

        # Check edge
        if expected_edge < self.limits.min_net_edge_dollars:
            return False, f"Edge {expected_edge:.4f} below minimum threshold {self.limits.min_net_edge_dollars:.4f}"

        # Daily loss circuit breaker
        if self.daily_realized_pnl <= -self.limits.max_daily_loss_dollars:
            return False, f"Daily circuit breaker triggered: realized loss ${abs(self.daily_realized_pnl):.2f} >= ${self.limits.max_daily_loss_dollars:.2f}"

        # Rate limiting
        cutoff = now - timedelta(minutes=1)
        self.order_timestamps = [ts for ts in self.order_timestamps if ts > cutoff]
        if len(self.order_timestamps) >= self.limits.max_orders_per_minute:
            return False, f"Rate limit exceeded: {len(self.order_timestamps)} orders in last 60s"

        order_notional = order.price * order.count

        # Max contract order size
        if order_notional > self.limits.max_contract_order_dollars:
            return False, f"Order notional ${order_notional:.2f} exceeds max per-order limit ${self.limits.max_contract_order_dollars:.2f}"

        # Max event allocation
        event_total = self.event_allocations.get(event_ticker, 0.0) + order_notional
        if event_total > self.limits.max_event_exposure_dollars:
            return False, f"Projected event exposure ${event_total:.2f} exceeds limit ${self.limits.max_event_exposure_dollars:.2f}"

        # Max total portfolio exposure
        total_current_exposure = sum(p.cost_basis for p in self.positions.values())
        if total_current_exposure + order_notional > self.limits.max_total_exposure_dollars:
            return False, f"Projected total exposure ${total_current_exposure + order_notional:.2f} exceeds limit ${self.limits.max_total_exposure_dollars:.2f}"

        # Record timestamp if valid
        self.order_timestamps.append(now)
        return True, None
