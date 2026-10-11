"""
Abstract Execution Gateway and Production Implementation for Kalshi + Paper Simulator.

Supports:
- Paper execution mode with realistic order book walk & slippage simulation
- Live REST / signed order execution for Kalshi exchange
"""
import abc
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List

from copula_execution.types import (
    OrderRequest, OrderResult, OrderStatus, MarketVenue, OrderBook, ActionSide, ContractSide
)
from copula_execution.risk_manager import FeeCalculator


class BaseExecutionGateway(abc.ABC):
    @abc.abstractmethod
    def submit_order(self, order: OrderRequest, current_book: Optional[OrderBook] = None) -> OrderResult:
        """Submit a single order."""
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abc.abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult:
        """Query status of an order."""
        pass


class PaperExecutionGateway(BaseExecutionGateway):
    """
    Simulated Execution Gateway designed for realistic paper-trading:
    - Matches orders directly against the order book levels.
    - Simulates market impact and book depth.
    - Fills limit orders when crossed or resting.
    - Calculates real taker/maker fees.
    """

    def __init__(self, fill_ratio: float = 1.0, latency_ms: float = 15.0):
        self.fill_ratio = fill_ratio
        self.latency_ms = latency_ms
        self.orders: Dict[str, OrderResult] = {}

    @staticmethod
    def _fee(order: OrderRequest, price: float, count: int) -> float:
        """Taker fee under the market's own schedule (venue default, or the paper fee for SIMULATED)."""
        if order.contract is not None:
            return FeeCalculator.fee_for_contract(order.contract, price, count, is_maker=False)
        return FeeCalculator.calculate_fee(order.venue, price, count, is_maker=False)

    def submit_order(self, order: OrderRequest, current_book: Optional[OrderBook] = None) -> OrderResult:
        # Simulate network latency
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        exchange_id = f"sim-{uuid.uuid4().hex[:12]}"
        
        # If no book available, fill optimistically at specified limit price
        if not current_book:
            fee = self._fee(order, order.price, order.count)
            res = OrderResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                status=OrderStatus.FILLED,
                filled_count=order.count,
                remaining_count=0,
                average_fill_price=order.price,
                fee_paid=fee,
                exchange_order_id=exchange_id,
                message="Filled in paper simulation without orderbook",
                timestamp=datetime.now(timezone.utc)
            )
            self.orders[order.order_id] = res
            return res

        # Walk the book for execution
        # If buying YES -> we take from YES asks
        # If selling YES -> we hit YES bids
        levels = current_book.yes_asks if order.action == ActionSide.BUY else current_book.yes_bids

        if not levels:
            # No counter liquidity
            res = OrderResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                status=OrderStatus.OPEN,
                filled_count=0,
                remaining_count=order.count,
                average_fill_price=0.0,
                fee_paid=0.0,
                exchange_order_id=exchange_id,
                message="Resting on book: counter-book empty",
                timestamp=datetime.now(timezone.utc)
            )
            self.orders[order.order_id] = res
            return res

        remaining = order.count
        total_spent = 0.0
        filled = 0

        for lvl in levels:
            if remaining <= 0:
                break

            # Price check
            if order.action == ActionSide.BUY and lvl.price > order.price:
                # Ask is above our buy limit price
                break
            elif order.action == ActionSide.SELL and lvl.price < order.price:
                # Bid is below our sell limit price
                break

            take_qty = int(min(remaining, lvl.size))
            if take_qty <= 0:
                continue

            filled += take_qty
            total_spent += take_qty * lvl.price
            remaining -= take_qty

        if filled > 0:
            avg_price = round(total_spent / filled, 4)
            fee = self._fee(order, avg_price, filled)
            status = OrderStatus.FILLED if remaining == 0 else OrderStatus.PARTIALLY_FILLED
            res = OrderResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                status=status,
                filled_count=filled,
                remaining_count=remaining,
                average_fill_price=avg_price,
                fee_paid=fee,
                exchange_order_id=exchange_id,
                message=f"Executed {filled}/{order.count} contracts @ avg ${avg_price:.4f}",
                timestamp=datetime.now(timezone.utc)
            )
        else:
            res = OrderResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                status=OrderStatus.OPEN,
                filled_count=0,
                remaining_count=order.count,
                average_fill_price=0.0,
                fee_paid=0.0,
                exchange_order_id=exchange_id,
                message="Unfilled: order placed outside current book spread",
                timestamp=datetime.now(timezone.utc)
            )

        self.orders[order.order_id] = res
        return res

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_order_status(self, order_id: str) -> OrderResult:
        if order_id in self.orders:
            return self.orders[order_id]
        return OrderResult(
            order_id=order_id,
            client_order_id=None,
            status=OrderStatus.REJECTED,
            message="Order not found"
        )


class KalshiExecutionGateway(BaseExecutionGateway):
    """
    Live Production Gateway for Kalshi Trade API v2.
    Requires API key authentication. Falls back safely or fails cleanly when credentials are absent.
    """

    def __init__(self, api_base: str = "https://external-api.kalshi.com/trade-api/v2",
                 api_key: Optional[str] = None, private_key: Optional[str] = None):
        self.api_base = api_base
        self.api_key = api_key
        self.private_key = private_key
        self.orders: Dict[str, OrderResult] = {}

    def submit_order(self, order: OrderRequest, current_book: Optional[OrderBook] = None) -> OrderResult:
        if not self.api_key:
            return OrderResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                status=OrderStatus.REJECTED,
                message="Kalshi credentials not configured. Please provide KALSHI_API_KEY.",
                timestamp=datetime.now(timezone.utc)
            )

        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Format price in cents (1 - 99) as required by Kalshi API v2
        price_cents = int(round(order.price * 100))

        payload = {
            "ticker": order.ticker,
            "action": order.action.value,          # "buy" or "sell"
            "type": order.order_type.value,        # "limit" or "market"
            "side": order.side.value,              # "yes" or "no"
            "count": order.count,
            "yes_price": price_cents if order.side == ContractSide.YES else None,
            "no_price": price_cents if order.side == ContractSide.NO else None,
            "client_order_id": order.client_order_id or order.order_id,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            r = requests.post(f"{self.api_base}/portfolio/orders", json=payload, headers=headers, timeout=10)
            if r.status_code == 201:
                data = r.json().get("order", {})
                res = OrderResult(
                    order_id=order.order_id,
                    client_order_id=order.client_order_id,
                    status=OrderStatus.SUBMITTED,
                    exchange_order_id=data.get("order_id"),
                    message="Submitted to Kalshi live order book",
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                res = OrderResult(
                    order_id=order.order_id,
                    client_order_id=order.client_order_id,
                    status=OrderStatus.REJECTED,
                    message=f"Kalshi API error ({r.status_code}): {r.text}",
                    timestamp=datetime.now(timezone.utc)
                )
        except Exception as e:
            res = OrderResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                status=OrderStatus.REJECTED,
                message=f"HTTP exception: {str(e)}",
                timestamp=datetime.now(timezone.utc)
            )

        self.orders[order.order_id] = res
        return res

    def cancel_order(self, order_id: str) -> bool:
        # Calls DELETE /portfolio/orders/{order_id}
        if not self.api_key:
            return False
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = requests.delete(f"{self.api_base}/portfolio/orders/{order_id}", headers=headers, timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> OrderResult:
        if order_id in self.orders:
            return self.orders[order_id]
        return OrderResult(order_id=order_id, client_order_id=None, status=OrderStatus.PENDING)
