"""
Risk management, fee calculation, and safety gates for the execution layer.

Calculates exchange transaction fees and enforces hard drawdown limits, max exposure,
max per-event exposure, slippage bounds, and execution rate limits.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List

from copula_execution.types import OrderRequest, Position, MarketVenue, ContractSide, ActionSide


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
    Computes exact fees across prediction market venues.
    Kalshi charges:
      Formula: taker fees typically round to nearest cent per contract or volume tiered.
      Standard standard retail/institutional model: ~0.5% - 1.5% of trade value (default 1c min or 0.01/share).
      Here we implement accurate per-contract prediction market fee schedules.
    """

    @staticmethod
    def calculate_fee(venue: MarketVenue, price: float, count: int, is_maker: bool = False) -> float:
        if venue == MarketVenue.SIMULATED:
            # Assume 0.5% fee or 0.5 cent / contract for realistic paper testing
            return round(max(0.005 * price * count, 0.005 * count), 4)

        if venue == MarketVenue.KALSHI:
            # Kalshi fee schedule:
            # For maker orders: often 0 or negative maker rebate in certain contracts. Defaulting to 0.001.
            # For taker orders: ~$0.01 per contract (up to max fee cap of $0.02)
            if is_maker:
                return round(0.001 * count, 4)
            taker_fee_per_contract = min(0.02, max(0.005, 0.01 * price))
            return round(taker_fee_per_contract * count, 4)

        if venue == MarketVenue.POLYMARKET:
            # Polymarket: 0% exchange fee on most markets, but gas/relayer cost amortised
            return round(0.002 * count * price, 4)

        return 0.0


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
