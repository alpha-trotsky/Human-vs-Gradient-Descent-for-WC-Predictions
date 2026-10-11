"""
Central Execution Orchestrator.

Wires together:
1. Ingestion layer (fetching order books & contracts)
2. Coherence scanner (arbitrage & model violation detection)
3. Pricing & quoting engines
4. Risk gates & fee calculator
5. Order execution gateway (paper & live)
6. Convergence tracking & metrics
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from copula_execution.types import (
    DisagreementFlag, OrderRequest, OrderResult, OrderBook, MarketContract,
    OrderType, ActionSide, ContractSide, MarketVenue
)
from copula_execution.gateway import BaseExecutionGateway, PaperExecutionGateway
from copula_execution.risk_manager import RiskManager, FeeCalculator
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.pricer import EconomicBracketPricer, SportsCopulaPricer, MarketMakerQuoter


class ExecutionEngine:
    def __init__(self,
                 gateway: Optional[BaseExecutionGateway] = None,
                 risk_manager: Optional[RiskManager] = None,
                 scanner: Optional[CoherenceScanner] = None):
        self.gateway = gateway or PaperExecutionGateway()
        self.risk_manager = risk_manager or RiskManager()
        self.scanner = scanner or CoherenceScanner()
        self.economic_pricer = EconomicBracketPricer()
        self.sports_pricer = SportsCopulaPricer()
        self.quoter = MarketMakerQuoter()

        # Audit logs & tracking
        self.flags_history: List[DisagreementFlag] = []
        self.execution_audit_trail: List[Dict[str, Any]] = []
        self.convergence_records: List[Dict[str, Any]] = []

    def scan_and_execute_bracket_arbitrage(self, event_ticker: str,
                                           contracts: List[MarketContract],
                                           books: Dict[str, OrderBook],
                                           auto_execute: bool = True) -> List[OrderResult]:
        """
        Executes arbitrage opportunities detected by the scanner:
        1. Evaluates sum-to-one brackets and monotonicity bounds.
        2. Routes suggested trades through risk gating.
        3. Fires orders to gateway and updates portfolio positions.
        """
        results = []

        # Run scanner
        sum_flags = self.scanner.check_mutually_exclusive_sum(event_ticker, contracts, books)
        mono_flags = self.scanner.check_monotonicity(event_ticker, contracts, books)
        all_flags = sum_flags + mono_flags

        for flag in all_flags:
            self.flags_history.append(flag)

            if not auto_execute:
                continue

            for trade in flag.suggested_trades:
                ticker = trade["ticker"]
                action = trade["action"]
                side = trade["side"]
                price = trade["price"]
                size = trade["size"]
                book = books.get(ticker)

                order_req = OrderRequest(
                    order_id=f"ord-{uuid.uuid4().hex[:10]}",
                    ticker=ticker,
                    venue=MarketVenue.SIMULATED,
                    action=action,
                    side=side,
                    order_type=OrderType.LIMIT,
                    price=price,
                    count=size,
                    flag_id=flag.flag_id
                )

                # Validate with Risk Gate
                is_valid, reject_reason = self.risk_manager.validate_order(
                    order=order_req,
                    event_ticker=event_ticker,
                    expected_edge=flag.gap_size
                )

                if not is_valid:
                    self.execution_audit_trail.append({
                        "order_id": order_req.order_id,
                        "ticker": ticker,
                        "status": "REJECTED_BY_RISK_GATE",
                        "reason": reject_reason,
                        "timestamp": datetime.now(timezone.utc)
                    })
                    continue

                # Submit to Execution Gateway
                order_res = self.gateway.submit_order(order_req, current_book=book)
                results.append(order_res)

                if order_res.filled_count > 0:
                    self.risk_manager.update_position_from_fill(
                        ticker=ticker,
                        event_ticker=event_ticker,
                        venue=MarketVenue.SIMULATED,
                        action=action,
                        side=side,
                        price=order_res.average_fill_price,
                        count=order_res.filled_count,
                        fee=order_res.fee_paid
                    )

                self.execution_audit_trail.append({
                    "order_id": order_req.order_id,
                    "ticker": ticker,
                    "status": order_res.status.value,
                    "fill_price": order_res.average_fill_price,
                    "filled_count": order_res.filled_count,
                    "fee_paid": order_res.fee_paid,
                    "flag_id": flag.flag_id,
                    "timestamp": datetime.now(timezone.utc)
                })

        return results

    def record_convergence_observation(self, flag: DisagreementFlag,
                                       current_prices: Dict[str, float]):
        """
        Tracks convergence rate:
        Does market price move toward model fair value post-detection?
        """
        for ticker in flag.contracts_involved:
            orig_price_data = flag.market_prices.get(ticker, {})
            orig_mid = orig_price_data.get("mid")
            new_price = current_prices.get(ticker)
            model_fv = flag.model_fair_values.get(ticker)

            if orig_mid is not None and new_price is not None and model_fv is not None:
                initial_gap = abs(orig_mid - model_fv)
                current_gap = abs(new_price - model_fv)
                converged = current_gap < initial_gap
                self.convergence_records.append({
                    "flag_id": flag.flag_id,
                    "ticker": ticker,
                    "initial_gap": initial_gap,
                    "current_gap": current_gap,
                    "converged": converged,
                    "convergence_delta": initial_gap - current_gap,
                    "observed_at": datetime.now(timezone.utc)
                })

    def get_portfolio_summary(self) -> Dict[str, Any]:
        total_realized_pnl = sum(p.realized_pnl for p in self.risk_manager.positions.values())
        total_fees = sum(p.fees_paid for p in self.risk_manager.positions.values())
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.risk_manager.positions.values())

        return {
            "positions_count": len(self.risk_manager.positions),
            "total_realized_pnl": round(total_realized_pnl, 4),
            "total_fees_paid": round(total_fees, 4),
            "total_unrealized_pnl": round(total_unrealized_pnl, 4),
            "net_pnl_after_fees": round(total_realized_pnl + total_unrealized_pnl - total_fees, 4),
            "flags_detected": len(self.flags_history),
            "orders_executed": len(self.execution_audit_trail)
        }
