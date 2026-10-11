"""
Stress and Edge-Case Tests for Copula Execution Layer:
- Monotonicity inversions across multiple ladder thresholds
- Circuit breaker trips on consecutive simulated losses
- Rate-limiting boundary checks
- Heavy orderbook depth exhaustion & slippage penalty
- Scoreline copula normalization and 1X2 consistency checks
- Negative edge suppression (fee-dominated traps)
"""
import unittest
import time
from datetime import datetime, timezone
import numpy as np

from copula_execution.types import (
    OrderBook, OrderBookLevel, MarketVenue, MarketContract, OrderRequest,
    OrderType, ActionSide, ContractSide, OrderStatus, DisagreementType, EventKind
)
from copula_execution.risk_manager import RiskManager, RiskLimits, FeeCalculator
from copula_execution.gateway import PaperExecutionGateway
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.pricer import EconomicBracketPricer, SportsCopulaPricer, MarketMakerQuoter
from copula_execution.engine import ExecutionEngine


class TestExecutionStressAndEdgeCases(unittest.TestCase):

    def test_fee_dominated_trap_suppression(self):
        """
        Gross edge exists (e.g. 1.2 cents), but fee is 1.5 cents.
        The system MUST suppress the flag and not execute a money-losing trade.
        """
        scanner = CoherenceScanner(min_edge=0.02)
        # Declared as an exhaustive partition so the *only* thing suppressing the flag is the fee.
        c1 = MarketContract("T-1", "EV-1", MarketVenue.KALSHI, "Bracket 1", floor_strike=1.0,
                            event_kind=EventKind.PARTITION, exhaustive=True)
        c2 = MarketContract("T-2", "EV-1", MarketVenue.KALSHI, "Bracket 2", floor_strike=2.0,
                            event_kind=EventKind.PARTITION, exhaustive=True)

        # Sum of asks is 0.985 (1.5c gross profit).
        # But fee on 2 contracts is ~2c. Net edge is negative!
        books = {
            "T-1": OrderBook("T-1", MarketVenue.KALSHI, datetime.now(timezone.utc),
                             yes_asks=[OrderBookLevel(price=0.49, size=10)]),
            "T-2": OrderBook("T-2", MarketVenue.KALSHI, datetime.now(timezone.utc),
                             yes_asks=[OrderBookLevel(price=0.495, size=10)])
        }

        flags = scanner.check_mutually_exclusive_sum("EV-1", [c1, c2], books)
        self.assertEqual(len(flags), 0, "Fee-dominated opportunity was not suppressed")

    def test_circuit_breaker_locks_trading_after_drawdown(self):
        """
        Verify that hitting the daily drawdown limit shuts down execution completely.
        """
        limits = RiskLimits(max_daily_loss_dollars=50.0, min_net_edge_dollars=0.01)
        rm = RiskManager(limits)

        req = OrderRequest(
            order_id="ord-safe",
            ticker="T-TEST",
            venue=MarketVenue.SIMULATED,
            action=ActionSide.BUY,
            side=ContractSide.YES,
            order_type=OrderType.LIMIT,
            price=0.40,
            count=10
        )

        valid, _ = rm.validate_order(req, "EV-1", expected_edge=0.05)
        self.assertTrue(valid)

        # Simulate a series of losing fills totaling $60 loss
        rm.daily_realized_pnl = -55.0

        valid_post_loss, reason = rm.validate_order(req, "EV-1", expected_edge=0.05)
        self.assertFalse(valid_post_loss, "Circuit breaker failed to trip after max drawdown")
        self.assertIn("Daily circuit breaker triggered", reason)

    def test_order_rate_limiter_boundary(self):
        """
        Verify orders are throttled when exceeding max_orders_per_minute.
        """
        limits = RiskLimits(max_orders_per_minute=5, min_net_edge_dollars=0.01)
        rm = RiskManager(limits)

        req = OrderRequest(
            order_id="ord-rate",
            ticker="T-TEST",
            venue=MarketVenue.SIMULATED,
            action=ActionSide.BUY,
            side=ContractSide.YES,
            order_type=OrderType.LIMIT,
            price=0.20,
            count=5
        )

        # First 5 orders should pass
        for i in range(5):
            valid, _ = rm.validate_order(req, "EV-1", expected_edge=0.05)
            self.assertTrue(valid, f"Order {i+1} should pass rate limiter")

        # 6th order in same window must fail
        valid_6th, reason = rm.validate_order(req, "EV-1", expected_edge=0.05)
        self.assertFalse(valid_6th)
        self.assertIn("Rate limit exceeded", reason)

    def test_orderbook_walking_and_partial_fills(self):
        """
        Verify paper gateway walks order book depth and correctly calculates VWAP fill price.
        """
        gw = PaperExecutionGateway(latency_ms=0)
        book = OrderBook(
            ticker="T-DEPTH",
            venue=MarketVenue.KALSHI,
            timestamp=datetime.now(timezone.utc),
            yes_asks=[
                OrderBookLevel(price=0.20, size=5),   # 5 @ $0.20
                OrderBookLevel(price=0.25, size=10),  # 10 @ $0.25
                OrderBookLevel(price=0.30, size=20)   # 20 @ $0.30
            ]
        )

        # Request 12 contracts with limit price $0.25
        # Should take 5 @ 0.20 ($1.00) + 7 @ 0.25 ($1.75) = $2.75 for 12 contracts (avg = $0.2292)
        req = OrderRequest(
            order_id="ord-vwap",
            ticker="T-DEPTH",
            venue=MarketVenue.SIMULATED,
            action=ActionSide.BUY,
            side=ContractSide.YES,
            order_type=OrderType.LIMIT,
            price=0.25,
            count=12
        )

        res = gw.submit_order(req, current_book=book)
        self.assertEqual(res.status, OrderStatus.FILLED)
        self.assertEqual(res.filled_count, 12)
        self.assertAlmostEqual(res.average_fill_price, 2.75 / 12, places=3)

    def test_sports_copula_grid_sums_to_one(self):
        """
        Verify score matrix generated by SportsCopulaPricer maintains probability axiom sum = 1.0.
        """
        pricer = SportsCopulaPricer()
        grid = pricer.calculate_scoreline_grid(mu_h=1.45, mu_a=1.10, r_h=7.5, r_a=7.5, theta=0.18, max_goals=10)

        self.assertEqual(grid.shape, (11, 11))
        self.assertAlmostEqual(float(np.sum(grid)), 1.0, places=3)
        self.assertTrue(np.all(grid >= 0.0))

        # Check exact score probability retrieval
        p_0_0 = pricer.price_exact_score(0, 0, grid)
        p_1_0 = pricer.price_exact_score(1, 0, grid)
        p_2_1 = pricer.price_exact_score(2, 1, grid)
        self.assertGreater(p_0_0, 0.01)
        self.assertGreater(p_1_0, 0.05)
        self.assertGreater(p_2_1, 0.03)

    def test_convergence_delta_calculation(self):
        """
        Verify convergence metric calculation updates properly when prices drift toward fair value.
        """
        engine = ExecutionEngine()
        flag = scanner_flag = CoherenceScanner().check_mutually_exclusive_sum(
            "EV-CONV",
            [
                MarketContract("C1", "EV-CONV", MarketVenue.KALSHI, "Leg 1",
                               event_kind=EventKind.PARTITION, exhaustive=True),
                MarketContract("C2", "EV-CONV", MarketVenue.KALSHI, "Leg 2",
                               event_kind=EventKind.PARTITION, exhaustive=True)
            ],
            {
                "C1": OrderBook("C1", MarketVenue.KALSHI, datetime.now(timezone.utc),
                                yes_bids=[OrderBookLevel(0.28, 10)], yes_asks=[OrderBookLevel(0.32, 10)]),
                "C2": OrderBook("C2", MarketVenue.KALSHI, datetime.now(timezone.utc),
                                yes_bids=[OrderBookLevel(0.28, 10)], yes_asks=[OrderBookLevel(0.32, 10)])
            }
        )[0]

        # Fair value is 0.50 each. Initial mid price is 0.30 (gap = 0.20)
        # Price moves to 0.45 (gap reduces to 0.05)
        engine.record_convergence_observation(flag, {"C1": 0.45, "C2": 0.45})
        self.assertEqual(len(engine.convergence_records), 2)
        rec = engine.convergence_records[0]
        self.assertTrue(rec["converged"])
        self.assertAlmostEqual(rec["convergence_delta"], 0.15, places=3)


if __name__ == "__main__":
    unittest.main()
