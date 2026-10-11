"""
Comprehensive Unit and Integration Test Suite for Copula Execution Layer.
"""
import unittest
from datetime import datetime, timezone

from copula_execution.types import (
    OrderBook, OrderBookLevel, MarketVenue, MarketContract, OrderRequest,
    OrderType, ActionSide, ContractSide, OrderStatus, DisagreementType
)
from copula_execution.risk_manager import RiskManager, RiskLimits, FeeCalculator
from copula_execution.gateway import PaperExecutionGateway
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.pricer import EconomicBracketPricer, MarketMakerQuoter
from copula_execution.engine import ExecutionEngine


class TestFeeCalculator(unittest.TestCase):
    def test_fee_calculation(self):
        # 10 contracts @ $0.50
        fee_kalshi = FeeCalculator.calculate_fee(MarketVenue.KALSHI, 0.50, 10, is_maker=False)
        self.assertGreater(fee_kalshi, 0.0)
        self.assertLessEqual(fee_kalshi, 0.20)

        fee_sim = FeeCalculator.calculate_fee(MarketVenue.SIMULATED, 0.50, 10)
        self.assertGreater(fee_sim, 0.0)


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.limits = RiskLimits(
            max_total_exposure_dollars=500.0,
            max_event_exposure_dollars=100.0,
            max_contract_order_dollars=30.0,
            min_net_edge_dollars=0.02,
            max_daily_loss_dollars=50.0
        )
        self.rm = RiskManager(self.limits)

    def test_edge_threshold_check(self):
        req = OrderRequest(
            order_id="test-1",
            ticker="TICKER-1",
            venue=MarketVenue.SIMULATED,
            action=ActionSide.BUY,
            side=ContractSide.YES,
            order_type=OrderType.LIMIT,
            price=0.50,
            count=10
        )
        # Edge 0.01 is below min 0.02
        valid, reason = self.rm.validate_order(req, "EVENT-1", expected_edge=0.01)
        self.assertFalse(valid)
        self.assertIn("below minimum threshold", reason)

        # Edge 0.03 is above min
        valid, reason = self.rm.validate_order(req, "EVENT-1", expected_edge=0.03)
        self.assertTrue(valid)

    def test_order_size_limit(self):
        req = OrderRequest(
            order_id="test-2",
            ticker="TICKER-1",
            venue=MarketVenue.SIMULATED,
            action=ActionSide.BUY,
            side=ContractSide.YES,
            order_type=OrderType.LIMIT,
            price=0.80,
            count=50  # 50 * 0.80 = $40 > $30 max
        )
        valid, reason = self.rm.validate_order(req, "EVENT-1", expected_edge=0.05)
        self.assertFalse(valid)
        self.assertIn("exceeds max per-order limit", reason)


class TestCoherenceScanner(unittest.TestCase):
    def setUp(self):
        self.scanner = CoherenceScanner(min_edge=0.02)

    def test_sum_to_one_arbitrage_detection(self):
        # Set up 3 mutually exclusive contracts where sum of YES asks is 0.70 (gross gap = $0.30!)
        c1 = MarketContract("T-A", "EV-1", MarketVenue.KALSHI, "A", floor_strike=1.0)
        c2 = MarketContract("T-B", "EV-1", MarketVenue.KALSHI, "B", floor_strike=2.0)
        c3 = MarketContract("T-C", "EV-1", MarketVenue.KALSHI, "C", floor_strike=3.0)

        books = {
            "T-A": OrderBook("T-A", MarketVenue.KALSHI, datetime.now(timezone.utc),
                             yes_asks=[OrderBookLevel(price=0.20, size=50)]),
            "T-B": OrderBook("T-B", MarketVenue.KALSHI, datetime.now(timezone.utc),
                             yes_asks=[OrderBookLevel(price=0.25, size=50)]),
            "T-C": OrderBook("T-C", MarketVenue.KALSHI, datetime.now(timezone.utc),
                             yes_asks=[OrderBookLevel(price=0.25, size=50)])
        }

        flags = self.scanner.check_mutually_exclusive_sum("EV-1", [c1, c2, c3], books)
        self.assertEqual(len(flags), 1)
        flag = flags[0]
        self.assertEqual(flag.disagreement_type, DisagreementType.SUM_TO_ONE)
        self.assertGreater(flag.gap_size, 0.20)
        self.assertEqual(len(flag.suggested_trades), 3)

    def test_monotonicity_arbitrage_detection(self):
        # Strike 2.0 has Ask = 0.40, Strike 4.0 has Bid = 0.55
        # Clearly impossible since P(X > 4) cannot exceed P(X > 2)!
        c_low = MarketContract("T-LOW", "EV-1", MarketVenue.KALSHI, "Low", floor_strike=2.0)
        c_high = MarketContract("T-HIGH", "EV-1", MarketVenue.KALSHI, "High", floor_strike=4.0)

        books = {
            "T-LOW": OrderBook("T-LOW", MarketVenue.KALSHI, datetime.now(timezone.utc),
                               yes_bids=[OrderBookLevel(price=0.35, size=20)],
                               yes_asks=[OrderBookLevel(price=0.40, size=20)]),
            "T-HIGH": OrderBook("T-HIGH", MarketVenue.KALSHI, datetime.now(timezone.utc),
                                yes_bids=[OrderBookLevel(price=0.55, size=20)],
                                yes_asks=[OrderBookLevel(price=0.60, size=20)])
        }

        flags = self.scanner.check_monotonicity("EV-1", [c_low, c_high], books)
        self.assertEqual(len(flags), 1)
        flag = flags[0]
        self.assertEqual(flag.disagreement_type, DisagreementType.MONOTONICITY)
        self.assertGreater(flag.gap_size, 0.10)


class TestEconomicPricerAndQuoter(unittest.TestCase):
    def test_economic_distribution_fitting(self):
        pricer = EconomicBracketPricer()
        # Test normal fitting from cumulative probabilities
        strikes = [2.0, 3.0, 4.0]
        cum_probs = [0.1587, 0.5000, 0.8413]  # Standard normal around mu=3, sigma=1
        mu, sigma = pricer.fit_distribution_from_brackets(strikes, cum_probs)
        self.assertAlmostEqual(mu, 3.0, delta=0.1)
        self.assertAlmostEqual(sigma, 1.0, delta=0.1)

        # Price middle bracket
        p_mid = pricer.price_bracket(2.5, 3.5, mu, sigma)
        self.assertGreater(p_mid, 0.3)
        self.assertLess(p_mid, 0.5)

    def test_two_sided_quoting(self):
        quoter = MarketMakerQuoter(target_spread=0.06)
        bid, ask = quoter.quote_contract(0.40, current_inventory=0)
        self.assertEqual(round(ask - bid, 2), 0.06)
        self.assertEqual(bid, 0.37)
        self.assertEqual(ask, 0.43)


class TestExecutionEngineIntegration(unittest.TestCase):
    def test_full_pipeline_arbitrage_execution(self):
        engine = ExecutionEngine()

        c1 = MarketContract("T-A", "EV-TEST", MarketVenue.KALSHI, "A", floor_strike=1.0)
        c2 = MarketContract("T-B", "EV-TEST", MarketVenue.KALSHI, "B", floor_strike=2.0)
        books = {
            "T-A": OrderBook("T-A", MarketVenue.KALSHI, datetime.now(timezone.utc),
                             yes_asks=[OrderBookLevel(price=0.25, size=50)]),
            "T-B": OrderBook("T-B", MarketVenue.KALSHI, datetime.now(timezone.utc),
                             yes_asks=[OrderBookLevel(price=0.30, size=50)])
        }

        # Sum of asks is 0.55 (< 1.0) -> free money arb
        order_results = engine.scan_and_execute_bracket_arbitrage("EV-TEST", [c1, c2], books, auto_execute=True)

        self.assertEqual(len(order_results), 2)
        for r in order_results:
            self.assertEqual(r.status, OrderStatus.FILLED)
            self.assertEqual(r.filled_count, 10)

        summary = engine.get_portfolio_summary()
        self.assertEqual(summary["positions_count"], 2)
        self.assertEqual(summary["flags_detected"], 1)
        self.assertEqual(summary["orders_executed"], 2)


if __name__ == "__main__":
    unittest.main()
