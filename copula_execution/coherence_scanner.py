"""
Arbitrage and Coherence Scanner for Prediction Markets.

Implements mathematical and structural consistency checks. Each check is only valid for certain
kinds of events, so contracts must be classified first (see copula_execution.classify):

1. Mutually exclusive baskets (EventKind.PARTITION only)
     - SELL every leg: at most one leg pays $1, so sum(bids) - fees > 1 is a guaranteed profit.
       Valid for any mutually exclusive set, exhaustive or not.
     - BUY every leg: guaranteed $1 payout only if the legs are *exhaustive* (one of them must win)
       and we hold *all* of them. Requires proven exhaustiveness and a live ask on every leg.
2. Monotonicity (EventKind.CUMULATIVE_THRESHOLD / NESTED_BY_DATE only)
     - Nested events: P(superset) >= P(subset). Bracket ("between") contracts are NOT nested and
       are never checked here.
3. Model-to-Market divergence: model fair value vs. the book, net of fees *and* model uncertainty.

All fees are computed per leg with the contract's own venue schedule at the size we would trade.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from copula_execution.classify import contract_edges
from copula_execution.types import (
    DisagreementFlag, DisagreementType, OrderBook, MarketContract, ActionSide, ContractSide, EventKind
)
from copula_execution.risk_manager import FeeCalculator


class CoherenceScanner:
    def __init__(self, fee_buffer: float = 0.02, min_edge: float = 0.02,
                 trade_size: int = 10, uncertainty_mult: float = 1.0):
        """
        min_edge: minimum net profit per contract required to raise a tradable flag
        trade_size: lot size suggested on every leg (fees are priced at this size, since
                    exchanges round per order)
        uncertainty_mult: model-disagreement edges must also clear this many fair-value sigmas
        """
        self.fee_buffer = fee_buffer
        self.min_edge = min_edge
        self.trade_size = trade_size
        self.uncertainty_mult = uncertainty_mult

    def _fee(self, contract: MarketContract, price: float) -> float:
        return FeeCalculator.fee_per_contract(contract, price, self.trade_size)

    # ------------------------------------------------------------------------------------------
    def check_mutually_exclusive_sum(self, event_ticker: str,
                                     contracts: List[MarketContract],
                                     books: Dict[str, OrderBook]) -> List[DisagreementFlag]:
        """
        Basket arbitrage over a mutually exclusive set.
          Under-priced (sum asks + fees < 1): buy one YES in every leg. Needs EXHAUSTIVE legs and a
            tradable ask on every one of them -- otherwise "exactly one leg pays $1" is unproven.
          Over-priced (sum bids - fees > 1): sell YES in every leg that has a bid. At most one leg can
            pay out, so the liability is capped at $1 whether or not the set is exhaustive.
        """
        flags: List[DisagreementFlag] = []
        if len(contracts) < 2:
            return flags
        if len({c.venue for c in contracts}) != 1:
            return flags   # legs on different venues are not one event with one resolution rule
        if any(c.event_kind != EventKind.PARTITION for c in contracts):
            return flags   # nested / unknown relationships do not sum to one

        exhaustive = all(c.exhaustive for c in contracts)
        with_books = [c for c in contracts if books.get(c.ticker) is not None]
        mid_of = {}
        for c in with_books:
            ob = books[c.ticker]
            mid_of[c.ticker] = ob.mid_price

        market_prices = {}
        for c in with_books:
            ob = books[c.ticker]
            market_prices[c.ticker] = {
                "bid": ob.best_yes_bid if ob.best_yes_bid is not None else 0.0,
                "ask": ob.best_yes_ask if ob.best_yes_ask is not None else 1.0,
                "mid": ob.mid_price if ob.mid_price is not None else 0.5,
            }

        def coherent_fvs(legs: List[MarketContract]) -> Dict[str, float]:
            """Market mids rescaled to sum to 1 -- the nearest coherent prices (used for convergence tracking)."""
            mids = {c.ticker: mid_of[c.ticker] for c in legs if mid_of.get(c.ticker) is not None}
            tot = sum(mids.values())
            if len(mids) != len(legs) or tot <= 0:
                return {}
            return {t: round(m / tot, 4) for t, m in mids.items()}

        # Scenario 1: buy every leg
        ask_legs = [c for c in contracts if books.get(c.ticker) is not None and books[c.ticker].best_yes_ask is not None]
        if exhaustive and len(ask_legs) == len(contracts):
            sum_asks = sum(books[c.ticker].best_yes_ask for c in ask_legs)
            total_fee = sum(self._fee(c, books[c.ticker].best_yes_ask) for c in ask_legs)
            if (sum_asks + total_fee) < (1.0 - self.min_edge):
                net_edge = round(1.0 - (sum_asks + total_fee), 4)
                flags.append(DisagreementFlag(
                    flag_id=f"arb-sum-under-{uuid.uuid4().hex[:8]}",
                    event_ticker=event_ticker,
                    disagreement_type=DisagreementType.SUM_TO_ONE,
                    contracts_involved=[c.ticker for c in ask_legs],
                    gap_size=net_edge,
                    gross_discrepancy=round(1.0 - sum_asks, 4),
                    model_fair_values=coherent_fvs(ask_legs),
                    market_prices={c.ticker: market_prices[c.ticker] for c in ask_legs},
                    detected_at=datetime.now(timezone.utc),
                    suggested_trades=[{
                        "ticker": c.ticker, "action": ActionSide.BUY, "side": ContractSide.YES,
                        "price": books[c.ticker].best_yes_ask, "size": self.trade_size,
                        "reason": "Basket sum < $1 arbitrage leg"} for c in ask_legs]
                ))

        # Scenario 2: sell every leg that has a bid (liability capped at $1 by mutual exclusivity)
        bid_legs = [c for c in with_books if books[c.ticker].best_yes_bid is not None]
        if len(bid_legs) >= 2:
            sum_bids = sum(books[c.ticker].best_yes_bid for c in bid_legs)
            total_fee = sum(self._fee(c, books[c.ticker].best_yes_bid) for c in bid_legs)
            if (sum_bids - total_fee) > (1.0 + self.min_edge):
                net_edge = round((sum_bids - total_fee) - 1.0, 4)
                flags.append(DisagreementFlag(
                    flag_id=f"arb-sum-over-{uuid.uuid4().hex[:8]}",
                    event_ticker=event_ticker,
                    disagreement_type=DisagreementType.SUM_TO_ONE,
                    contracts_involved=[c.ticker for c in bid_legs],
                    gap_size=net_edge,
                    gross_discrepancy=round(sum_bids - 1.0, 4),
                    model_fair_values=coherent_fvs(bid_legs),
                    market_prices={c.ticker: market_prices[c.ticker] for c in bid_legs},
                    detected_at=datetime.now(timezone.utc),
                    suggested_trades=[{
                        "ticker": c.ticker, "action": ActionSide.SELL, "side": ContractSide.YES,
                        "price": books[c.ticker].best_yes_bid, "size": self.trade_size,
                        "reason": "Basket sum > $1 overpriced sell leg"} for c in bid_legs]
                ))

        return flags

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _nested_chains(contracts: List[MarketContract]) -> List[List[MarketContract]]:
        """
        Chains ordered so that probability is non-increasing along the chain, i.e. every contract
        is a (weak) superset of the ones after it:
          'above k' ladders : ascending k          P(X > k) falls as k rises
          'below k' ladders : descending k         P(X < k) falls as k falls
          by-date ladders   : latest deadline first
        Anything that is not a true nested ladder (brackets, partitions, unknowns) yields no chain.
        """
        chains: List[List[MarketContract]] = []
        thr = [c for c in contracts if c.event_kind == EventKind.CUMULATIVE_THRESHOLD]
        above = [c for c in thr if (contract_edges(c) or (None, None))[0] is not None
                 and (contract_edges(c) or (None, None))[1] is None]
        below = [c for c in thr if (contract_edges(c) or (None, None))[0] is None
                 and (contract_edges(c) or (None, None))[1] is not None]
        if len(above) >= 2:
            chains.append(sorted(above, key=lambda c: contract_edges(c)[0]))
        if len(below) >= 2:
            chains.append(sorted(below, key=lambda c: contract_edges(c)[1], reverse=True))
        dated = [c for c in contracts if c.event_kind == EventKind.NESTED_BY_DATE and c.expiration_time is not None]
        if len(dated) >= 2:
            chains.append(sorted(dated, key=lambda c: c.expiration_time, reverse=True))
        return chains

    @staticmethod
    def _label(c: MarketContract) -> str:
        if c.event_kind == EventKind.NESTED_BY_DATE and c.expiration_time is not None:
            return f"deadline {c.expiration_time.date()}"
        e = contract_edges(c)
        return f"strike {e[0] if e and e[0] is not None else (e[1] if e else '?')}"

    def check_monotonicity(self, event_ticker: str,
                           threshold_contracts: List[MarketContract],
                           books: Dict[str, OrderBook]) -> List[DisagreementFlag]:
        """
        For nested contracts, P(superset) >= P(subset). If Bid(subset) > Ask(superset) + fees:
        buy the superset YES and sell the subset YES.
          - neither happens: both pay $0
          - only the superset happens: +$1
          - both happen: $1 - $1 = 0
        so the payoff is never negative and the inception cash flow Bid - Ask - fees is locked in.
        """
        flags: List[DisagreementFlag] = []
        for chain in self._nested_chains(threshold_contracts):
            for i in range(len(chain)):
                for j in range(i + 1, len(chain)):
                    big, small = chain[i], chain[j]
                    if big.venue != small.venue:
                        continue   # different venues may resolve differently
                    book_big, book_small = books.get(big.ticker), books.get(small.ticker)
                    if not book_big or not book_small:
                        continue
                    ask_big, bid_small = book_big.best_yes_ask, book_small.best_yes_bid
                    if ask_big is None or bid_small is None:
                        continue

                    total_fees = self._fee(big, ask_big) + self._fee(small, bid_small)
                    gross_edge = bid_small - ask_big
                    net_edge = gross_edge - total_fees
                    if net_edge < self.min_edge:
                        continue
                    flags.append(DisagreementFlag(
                        flag_id=f"arb-mono-{uuid.uuid4().hex[:8]}",
                        event_ticker=event_ticker,
                        disagreement_type=DisagreementType.MONOTONICITY,
                        contracts_involved=[big.ticker, small.ticker],
                        gap_size=round(net_edge, 4),
                        gross_discrepancy=round(gross_edge, 4),
                        model_fair_values={},
                        market_prices={
                            big.ticker: {"bid": book_big.best_yes_bid or 0.0, "ask": ask_big,
                                         "mid": book_big.mid_price or ask_big},
                            small.ticker: {"bid": bid_small, "ask": book_small.best_yes_ask or 1.0,
                                           "mid": book_small.mid_price or bid_small},
                        },
                        detected_at=datetime.now(timezone.utc),
                        suggested_trades=[
                            {"ticker": big.ticker, "action": ActionSide.BUY, "side": ContractSide.YES,
                             "price": ask_big, "size": self.trade_size,
                             "reason": f"Buy superset leg ({self._label(big)})"},
                            {"ticker": small.ticker, "action": ActionSide.SELL, "side": ContractSide.YES,
                             "price": bid_small, "size": self.trade_size,
                             "reason": f"Sell subset leg ({self._label(small)})"},
                        ]
                    ))
        return flags

    # ------------------------------------------------------------------------------------------
    def check_model_disagreement(self, event_ticker: str,
                                 contract: MarketContract,
                                 model_fair_value: float,
                                 book: OrderBook,
                                 uncertainty: float = 0.0) -> Optional[DisagreementFlag]:
        """
        Detects when a model's fair value diverges from the book by more than fees plus
        `uncertainty_mult` model sigmas. Only call with fair values that are an independent model
        view (FairValue.has_model_view); a rescaled market mid is not one.
        """
        bid = book.best_yes_bid
        ask = book.best_yes_ask
        if bid is None and ask is None:
            return None
        haircut = self.uncertainty_mult * max(uncertainty, 0.0)

        if ask is not None:
            fee = self._fee(contract, ask)
            net_edge = model_fair_value - ask - fee - haircut
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
                        "ticker": contract.ticker, "action": ActionSide.BUY, "side": ContractSide.YES,
                        "price": ask, "size": self.trade_size,
                        "reason": f"Model FV {model_fair_value:.3f} > Ask {ask:.3f}"}]
                )

        if bid is not None:
            fee = self._fee(contract, bid)
            net_edge = bid - model_fair_value - fee - haircut
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
                        "ticker": contract.ticker, "action": ActionSide.SELL, "side": ContractSide.YES,
                        "price": bid, "size": self.trade_size,
                        "reason": f"Bid {bid:.3f} > Model FV {model_fair_value:.3f}"}]
                )

        return None
