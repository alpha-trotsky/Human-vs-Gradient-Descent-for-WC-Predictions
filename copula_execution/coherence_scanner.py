"""
Arbitrage and Coherence Scanner for Prediction Markets.

Implements mathematical and structural consistency checks:
1. Mutually Exclusive Brackets sum to 1.0 (checking both bid-sum and ask-sum bounds after fees).
2. Monotonicity: 'Above X' contracts must strictly decrease in price as strike X increases.
3. Model-to-Market Copula Divergence: flags contracts where calibrated joint model fair value
   differs from market quotes by more than spread + fees.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from copula_execution.types import (
    DisagreementFlag, DisagreementType, OrderBook, MarketContract, ActionSide, ContractSide
)
from copula_execution.risk_manager import FeeCalculator, MarketVenue


class CoherenceScanner:
    def __init__(self, fee_buffer: float = 0.02, min_edge: float = 0.02):
        """
        fee_buffer: safety buffer for exchange transaction costs
        min_edge: minimum net profit required to raise a tradable flag
        """
        self.fee_buffer = fee_buffer
        self.min_edge = min_edge

    def check_mutually_exclusive_sum(self, event_ticker: str,
                                     contracts: List[MarketContract],
                                     books: Dict[str, OrderBook]) -> List[DisagreementFlag]:
        """
        In a mutually exclusive, collectively exhaustive set of brackets (e.g. GDP brackets,
        1X2 match odds, exact scores):
          - Buying all YES asks guarantees exactly $1.00 payout. If sum(YES asks) + fees < $1.00,
            we have a guaranteed BUY arbitrage!
          - Selling all YES bids (or buying all NO asks) guarantees paying $1.00.
            If sum(YES bids) - fees > $1.00, we have an over-priced SELL arbitrage!
        """
        flags = []
        valid_contracts = [c for c in contracts if c.ticker in books and books[c.ticker] is not None]
        if len(valid_contracts) < 2:
            return flags

        # Check if set is complete or collectively exhaustive
        # Calculate sum of best YES asks and sum of best YES bids
        sum_asks = 0.0
        sum_bids = 0.0
        all_asks_present = True
        all_bids_present = True
        total_fee = 0.0

        market_prices = {}
        for c in valid_contracts:
            ob = books[c.ticker]
            bid = ob.best_yes_bid
            ask = ob.best_yes_ask
            market_prices[c.ticker] = {
                "bid": bid if bid is not None else 0.0,
                "ask": ask if ask is not None else 1.0,
                "mid": ob.mid_price or 0.5
            }
            if ask is not None:
                sum_asks += ask
                total_fee += FeeCalculator.calculate_fee(MarketVenue.KALSHI, ask, 1, is_maker=False)
            else:
                all_asks_present = False

            if bid is not None:
                sum_bids += bid
            else:
                all_bids_present = False

        # Scenario 1: Underpriced basket (Sum of Asks + Fees < 1.0)
        # We can buy 1 YES in each bracket for sum_asks. Exactly one bracket resolves YES, yielding $1.00.
        if all_asks_present and (sum_asks + total_fee) < (1.0 - self.min_edge):
            net_edge = round(1.0 - (sum_asks + total_fee), 4)
            suggested_trades = []
            for c in valid_contracts:
                ask_p = books[c.ticker].best_yes_ask
                suggested_trades.append({
                    "ticker": c.ticker,
                    "action": ActionSide.BUY,
                    "side": ContractSide.YES,
                    "price": ask_p,
                    "size": 10,
                    "reason": "Basket sum < $1 arbitrage leg"
                })

            flags.append(DisagreementFlag(
                flag_id=f"arb-sum-under-{uuid.uuid4().hex[:8]}",
                event_ticker=event_ticker,
                disagreement_type=DisagreementType.SUM_TO_ONE,
                contracts_involved=[c.ticker for c in valid_contracts],
                gap_size=net_edge,
                gross_discrepancy=round(1.0 - sum_asks, 4),
                model_fair_values={c.ticker: round(1.0 / len(valid_contracts), 4) for c in valid_contracts},
                market_prices=market_prices,
                detected_at=datetime.now(timezone.utc),
                suggested_trades=suggested_trades
            ))

        # Scenario 2: Overpriced basket (Sum of Bids - Fees > 1.0)
        # We can sell (or short/buy NO) across all brackets. Total payoff cost is $1.00, proceeds = sum_bids.
        if all_bids_present and (sum_bids - total_fee) > (1.0 + self.min_edge):
            net_edge = round((sum_bids - total_fee) - 1.0, 4)
            suggested_trades = []
            for c in valid_contracts:
                bid_p = books[c.ticker].best_yes_bid
                suggested_trades.append({
                    "ticker": c.ticker,
                    "action": ActionSide.SELL,
                    "side": ContractSide.YES,
                    "price": bid_p,
                    "size": 10,
                    "reason": "Basket sum > $1 overpriced sell leg"
                })

            flags.append(DisagreementFlag(
                flag_id=f"arb-sum-over-{uuid.uuid4().hex[:8]}",
                event_ticker=event_ticker,
                disagreement_type=DisagreementType.SUM_TO_ONE,
                contracts_involved=[c.ticker for c in valid_contracts],
                gap_size=net_edge,
                gross_discrepancy=round(sum_bids - 1.0, 4),
                model_fair_values={c.ticker: round(1.0 / len(valid_contracts), 4) for c in valid_contracts},
                market_prices=market_prices,
                detected_at=datetime.now(timezone.utc),
                suggested_trades=suggested_trades
            ))

        return flags

    def check_monotonicity(self, event_ticker: str,
                           threshold_contracts: List[MarketContract],
                           books: Dict[str, OrderBook]) -> List[DisagreementFlag]:
        """
        For cumulative 'Above X' contracts:
        If strike X1 < strike X2, then mathematically P(Value > X1) >= P(Value > X2).
        Therefore:
          - If Best YES Bid(X2) > Best YES Ask(X1) + fees:
            Buy YES(X1) and Sell YES(X2).
            Net cashflow at inception = Bid(X2) - Ask(X1) - fees > 0.
            At expiration:
              - If Value <= X1: both pay $0. Net payout = 0.
              - If X1 < Value <= X2: YES(X1) pays $1, YES(X2) pays $0. Net payout = +$1.
              - If Value > X2: both pay $1. Net payout = $1 - $1 = 0.
            This is a 100% risk-free arbitrage with strictly non-negative payout!
        """
        flags = []
        # Filter for contracts that have a floor_strike
        sorted_contracts = [c for c in threshold_contracts if c.floor_strike is not None and c.ticker in books]
        sorted_contracts.sort(key=lambda c: c.floor_strike)

        for i in range(len(sorted_contracts)):
            for j in range(i + 1, len(sorted_contracts)):
                c_low = sorted_contracts[i]   # Lower strike X1
                c_high = sorted_contracts[j]  # Higher strike X2

                book_low = books[c_low.ticker]
                book_high = books[c_high.ticker]
                if not book_low or not book_high:
                    continue

                ask_low = book_low.best_yes_ask
                bid_high = book_high.best_yes_bid

                if ask_low is None or bid_high is None:
                    continue

                fee_low = FeeCalculator.calculate_fee(MarketVenue.KALSHI, ask_low, 1)
                fee_high = FeeCalculator.calculate_fee(MarketVenue.KALSHI, bid_high, 1)
                total_fees = fee_low + fee_high

                gross_edge = bid_high - ask_low
                net_edge = gross_edge - total_fees

                if net_edge >= self.min_edge:
                    flags.append(DisagreementFlag(
                        flag_id=f"arb-mono-{uuid.uuid4().hex[:8]}",
                        event_ticker=event_ticker,
                        disagreement_type=DisagreementType.MONOTONICITY,
                        contracts_involved=[c_low.ticker, c_high.ticker],
                        gap_size=round(net_edge, 4),
                        gross_discrepancy=round(gross_edge, 4),
                        model_fair_values={},
                        market_prices={
                            c_low.ticker: {"bid": book_low.best_yes_bid or 0.0, "ask": ask_low, "mid": book_low.mid_price or ask_low},
                            c_high.ticker: {"bid": bid_high, "ask": book_high.best_yes_ask or 1.0, "mid": book_high.mid_price or bid_high}
                        },
                        detected_at=datetime.now(timezone.utc),
                        suggested_trades=[
                            {
                                "ticker": c_low.ticker,
                                "action": ActionSide.BUY,
                                "side": ContractSide.YES,
                                "price": ask_low,
                                "size": 10,
                                "reason": f"Buy lower strike {c_low.floor_strike}"
                            },
                            {
                                "ticker": c_high.ticker,
                                "action": ActionSide.SELL,
                                "side": ContractSide.YES,
                                "price": bid_high,
                                "size": 10,
                                "reason": f"Sell higher strike {c_high.floor_strike}"
                            }
                        ]
                    ))

        return flags

    def check_model_disagreement(self, event_ticker: str,
                                 contract: MarketContract,
                                 model_fair_value: float,
                                 book: OrderBook) -> Optional[DisagreementFlag]:
        """
        Detects when a model's calibrated joint probability diverges from the market book
        by more than spread + fees.
        """
        bid = book.best_yes_bid
        ask = book.best_yes_ask
        if bid is None and ask is None:
            return None

        # If model expects price much higher than market Ask: BUY signal
        if ask is not None:
            fee = FeeCalculator.calculate_fee(MarketVenue.KALSHI, ask, 1)
            net_edge = model_fair_value - ask - fee
            if net_edge >= self.min_edge:
                return DisagreementFlag(
                    flag_id=f"arb-copula-buy-{uuid.uuid4().hex[:8]}",
                    event_ticker=event_ticker,
                    disagreement_type=DisagreementType.COPULA_ARBITRAGE,
                    contracts_involved=[contract.ticker],
                    gap_size=round(net_edge, 4),
                    gross_discrepancy=round(model_fair_value - ask, 4),
                    model_fair_values={contract.ticker: round(model_fair_value, 4)},
                    market_prices={contract.ticker: {"bid": bid or 0.0, "ask": ask, "mid": book.mid_price or ask}},
                    detected_at=datetime.now(timezone.utc),
                    suggested_trades=[{
                        "ticker": contract.ticker,
                        "action": ActionSide.BUY,
                        "side": ContractSide.YES,
                        "price": ask,
                        "size": 10,
                        "reason": f"Model FV {model_fair_value:.3f} > Ask {ask:.3f}"
                    }]
                )

        # If model expects price much lower than market Bid: SELL signal
        if bid is not None:
            fee = FeeCalculator.calculate_fee(MarketVenue.KALSHI, bid, 1)
            net_edge = bid - model_fair_value - fee
            if net_edge >= self.min_edge:
                return DisagreementFlag(
                    flag_id=f"arb-copula-sell-{uuid.uuid4().hex[:8]}",
                    event_ticker=event_ticker,
                    disagreement_type=DisagreementType.COPULA_ARBITRAGE,
                    contracts_involved=[contract.ticker],
                    gap_size=round(net_edge, 4),
                    gross_discrepancy=round(bid - model_fair_value, 4),
                    model_fair_values={contract.ticker: round(model_fair_value, 4)},
                    market_prices={contract.ticker: {"bid": bid, "ask": ask or 1.0, "mid": book.mid_price or bid}},
                    detected_at=datetime.now(timezone.utc),
                    suggested_trades=[{
                        "ticker": contract.ticker,
                        "action": ActionSide.SELL,
                        "side": ContractSide.YES,
                        "price": bid,
                        "size": 10,
                        "reason": f"Bid {bid:.3f} > Model FV {model_fair_value:.3f}"
                    }]
                )

        return None
