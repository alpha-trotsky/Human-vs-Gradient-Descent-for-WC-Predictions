"""
Tests for the event-aware pricing engine: classification, scanner guards, fee schedules,
event pricing (partition / ladder / by-date), and the quoter.
Offline: books are synthetic but mirror structures seen on the live Kalshi / Polymarket APIs.
"""
import math
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import norm

from copula_execution.types import (
    EventKind, MarketContract, MarketVenue, OrderBook, OrderBookLevel
)
from copula_execution.classify import classify_kalshi_event, classify_polymarket_event
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.pricer import (
    CalibrationError, EconomicBracketPricer, EventPricer, FairValue, MarketMakerQuoter,
    SportsCopulaPricer, pava_nondecreasing, SOURCE_NORMALIZED_MID,
)
from copula_execution.risk_manager import FeeCalculator

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def book(ticker, bid, ask, venue=MarketVenue.KALSHI):
    return OrderBook(ticker, venue, NOW,
                     yes_bids=[OrderBookLevel(bid, 100)] if bid is not None else [],
                     yes_asks=[OrderBookLevel(ask, 100)] if ask is not None else [])


def kalshi(ticker, strike_type, floor=None, cap=None, event="EV", **kw):
    return MarketContract(ticker, event, MarketVenue.KALSHI, ticker, strike_type=strike_type,
                          floor_strike=floor, cap_strike=cap, **kw)


def gdp_style_event(step=0.1, drop=None):
    """less 0.1 | between [0.1,0.5] [0.6,1.0] ... | greater 6 -- the live KXGDPYEAR structure."""
    cs = [kalshi("L", "less", cap=0.1)]
    f = 0.1
    while f < 5.9:
        cs.append(kalshi(f"B{f:.1f}", "between", floor=round(f, 1), cap=round(f + 0.4, 1)))
        f = round(f + 0.5, 1)
    cs.append(kalshi("G", "greater", floor=round(f - 0.1, 1)))
    if drop:
        cs = [c for c in cs if c.ticker != drop]
    return cs


# --------------------------------------------------------------------------------------------
class TestClassification(unittest.TestCase):
    def test_quantised_brackets_tile_and_are_exhaustive(self):
        cs = gdp_style_event()
        classify_kalshi_event(cs, event_mutually_exclusive=True)
        self.assertTrue(all(c.event_kind == EventKind.PARTITION for c in cs))
        self.assertTrue(all(c.exhaustive for c in cs))
        self.assertAlmostEqual(cs[0].resolution_step, 0.1)

    def test_missing_bracket_breaks_exhaustiveness(self):
        cs = gdp_style_event(drop="B1.1")
        classify_kalshi_event(cs, event_mutually_exclusive=True)
        self.assertTrue(all(c.event_kind == EventKind.PARTITION for c in cs))
        self.assertFalse(any(c.exhaustive for c in cs))

    def test_regular_grid_without_open_ends_is_not_exhaustive(self):
        # brackets are perfectly regular, but nothing covers values above the top bracket (or below the bottom)
        for drop in ("G", "L"):
            cs = gdp_style_event(drop=drop)
            classify_kalshi_event(cs, event_mutually_exclusive=True)
            self.assertTrue(all(c.event_kind == EventKind.PARTITION for c in cs))
            self.assertFalse(any(c.exhaustive for c in cs), f"dropped {drop}")

    def test_truncated_listing_is_never_exhaustive(self):
        cs = gdp_style_event()
        classify_kalshi_event(cs, event_mutually_exclusive=True, truncated=True)
        self.assertFalse(any(c.exhaustive for c in cs))

    def test_threshold_ladder_is_not_a_partition(self):
        cs = [kalshi(f"G{k}", "greater", floor=k) for k in (1.0, 2.0, 3.0)]
        classify_kalshi_event(cs, event_mutually_exclusive=False)
        self.assertTrue(all(c.event_kind == EventKind.CUMULATIVE_THRESHOLD for c in cs))
        self.assertEqual(cs[1].model_edges, (2.0, None))

    def test_polymarket_negrisk_augmented_is_not_exhaustive(self):
        cs = [MarketContract(f"P{i}", "E", MarketVenue.POLYMARKET, f"Will {n} win?", subtitle=n)
              for i, n in enumerate(["Democratic Party", "Republican Party"])]
        classify_polymarket_event(cs, neg_risk=True, neg_risk_augmented=True)
        self.assertTrue(all(c.event_kind == EventKind.PARTITION and not c.exhaustive for c in cs))
        classify_polymarket_event(cs, neg_risk=True, neg_risk_augmented=False)
        self.assertTrue(all(c.exhaustive for c in cs))

    def test_polymarket_by_date_is_nested_not_partition(self):
        cs = [MarketContract(f"D{i}", "E", MarketVenue.POLYMARKET, f"Kraken IPO by {m} 2027?",
                             expiration_time=NOW + timedelta(days=30 * (i + 1)))
              for i, m in enumerate(["March 31,", "June 30,", "December 31,"])]
        classify_polymarket_event(cs, neg_risk=False, neg_risk_augmented=False)
        self.assertTrue(all(c.event_kind == EventKind.NESTED_BY_DATE for c in cs))

    def test_polymarket_unrelated_markets_stay_unknown(self):
        cs = [MarketContract("A", "E", MarketVenue.POLYMARKET, "Will A happen?", expiration_time=NOW),
              MarketContract("B", "E", MarketVenue.POLYMARKET, "Will B happen?",
                             expiration_time=NOW + timedelta(days=1))]
        classify_polymarket_event(cs, neg_risk=False, neg_risk_augmented=False)
        self.assertTrue(all(c.event_kind == EventKind.UNKNOWN for c in cs))

    def test_ordinal_titles_get_numeric_edges(self):
        titles = ["0 (0 bps)", "1 (25 bps)", "2 (50 bps)", "3+ (75+ bps)"]
        cs = [MarketContract(f"F{i}", "E", MarketVenue.POLYMARKET, "q", subtitle=t) for i, t in enumerate(titles)]
        classify_polymarket_event(cs, neg_risk=True, neg_risk_augmented=False)
        self.assertEqual(cs[0].model_edges, (None, 0.5))
        self.assertEqual(cs[1].model_edges, (0.5, 1.5))
        self.assertEqual(cs[3].model_edges, (2.5, None))


# --------------------------------------------------------------------------------------------
class TestScannerGuards(unittest.TestCase):
    def setUp(self):
        self.sc = CoherenceScanner(min_edge=0.01)

    def test_nested_by_date_never_flags_sum_to_one(self):
        # sum of bids is far above 1 -- but "by date" markets are nested, not exclusive
        cs = [MarketContract(f"D{i}", "E", MarketVenue.POLYMARKET, "t", event_kind=EventKind.NESTED_BY_DATE,
                             expiration_time=NOW + timedelta(days=i + 1)) for i in range(3)]
        bk = {c.ticker: book(c.ticker, 0.60, 0.62, MarketVenue.POLYMARKET) for c in cs}
        self.assertEqual(self.sc.check_mutually_exclusive_sum("E", cs, bk), [])

    def test_unknown_kind_never_flags(self):
        cs = [MarketContract(f"U{i}", "E", MarketVenue.KALSHI, "t") for i in range(3)]
        bk = {c.ticker: book(c.ticker, 0.1, 0.12) for c in cs}
        self.assertEqual(self.sc.check_mutually_exclusive_sum("E", cs, bk), [])

    def test_buy_basket_requires_exhaustive_but_sell_basket_does_not(self):
        def legs(exhaustive):
            return [MarketContract(f"M{i}", "E", MarketVenue.KALSHI, "t", event_kind=EventKind.PARTITION,
                                   exhaustive=exhaustive) for i in range(3)]
        cheap = lambda cs: {c.ticker: book(c.ticker, 0.15, 0.20) for c in cs}   # asks sum 0.60
        rich = lambda cs: {c.ticker: book(c.ticker, 0.45, 0.50) for c in cs}    # bids sum 1.35

        self.assertEqual(len(self.sc.check_mutually_exclusive_sum("E", legs(True), cheap(legs(True)))), 1)
        self.assertEqual(self.sc.check_mutually_exclusive_sum("E", legs(False), cheap(legs(False))), [])
        for ex in (True, False):
            ls = legs(ex)
            flags = self.sc.check_mutually_exclusive_sum("E", ls, rich(ls))
            self.assertEqual([f.flag_id.startswith("arb-sum-over") for f in flags], [True])

    def test_buy_basket_needs_a_live_ask_on_every_leg(self):
        cs = [MarketContract(f"M{i}", "E", MarketVenue.KALSHI, "t", event_kind=EventKind.PARTITION,
                             exhaustive=True) for i in range(3)]
        bk = {c.ticker: book(c.ticker, 0.15, 0.20) for c in cs}
        bk["M2"] = book("M2", 0.15, None)
        self.assertEqual(self.sc.check_mutually_exclusive_sum("E", cs, bk), [])
        del bk["M2"]
        self.assertEqual(self.sc.check_mutually_exclusive_sum("E", cs, bk), [])

    def test_mixed_venues_are_not_one_basket(self):
        cs = [MarketContract("K", "E", MarketVenue.KALSHI, "t", event_kind=EventKind.PARTITION, exhaustive=True),
              MarketContract("P", "E", MarketVenue.POLYMARKET, "t", event_kind=EventKind.PARTITION, exhaustive=True)]
        bk = {"K": book("K", 0.1, 0.2), "P": book("P", 0.1, 0.2, MarketVenue.POLYMARKET)}
        self.assertEqual(self.sc.check_mutually_exclusive_sum("E", cs, bk), [])

    def test_brackets_are_never_checked_for_monotonicity(self):
        # adjacent brackets: a higher floor with a higher price is perfectly legal
        cs = gdp_style_event()
        classify_kalshi_event(cs, event_mutually_exclusive=True)
        bk = {c.ticker: book(c.ticker, 0.30, 0.31) for c in cs}
        bk["B0.6"] = book("B0.6", 0.05, 0.06)    # lower-floor bracket cheap ...
        bk["B1.1"] = book("B1.1", 0.40, 0.41)    # ... higher-floor bracket rich
        self.assertEqual(self.sc.check_monotonicity("EV", cs, bk), [])

    def test_threshold_ladder_inversion_is_flagged(self):
        cs = [kalshi("G1", "greater", floor=1.0), kalshi("G2", "greater", floor=2.0)]
        classify_kalshi_event(cs, event_mutually_exclusive=False)
        bk = {"G1": book("G1", 0.30, 0.35), "G2": book("G2", 0.55, 0.60)}   # P(>2) priced above P(>1)
        flags = self.sc.check_monotonicity("EV", cs, bk)
        self.assertEqual(len(flags), 1)
        buy, sell = flags[0].suggested_trades
        self.assertEqual((buy["ticker"], sell["ticker"]), ("G1", "G2"))

    def test_by_date_inversion_is_flagged(self):
        cs = [MarketContract("EARLY", "E", MarketVenue.POLYMARKET, "t", event_kind=EventKind.NESTED_BY_DATE,
                             expiration_time=NOW + timedelta(days=10)),
              MarketContract("LATE", "E", MarketVenue.POLYMARKET, "t", event_kind=EventKind.NESTED_BY_DATE,
                             expiration_time=NOW + timedelta(days=100))]
        # P(by early) = 0.50 > P(by late) = 0.30 is impossible
        bk = {"EARLY": book("EARLY", 0.50, 0.52, MarketVenue.POLYMARKET),
              "LATE": book("LATE", 0.28, 0.30, MarketVenue.POLYMARKET)}
        flags = self.sc.check_monotonicity("E", cs, bk)
        self.assertEqual(len(flags), 1)
        buy, sell = flags[0].suggested_trades
        self.assertEqual((buy["ticker"], sell["ticker"]), ("LATE", "EARLY"))

    def test_model_disagreement_must_clear_uncertainty(self):
        c = MarketContract("X", "E", MarketVenue.KALSHI, "t")
        b = book("X", 0.28, 0.30)
        self.assertIsNotNone(self.sc.check_model_disagreement("E", c, 0.45, b, uncertainty=0.01))
        self.assertIsNone(self.sc.check_model_disagreement("E", c, 0.45, b, uncertainty=0.15))


# --------------------------------------------------------------------------------------------
class TestFees(unittest.TestCase):
    def test_kalshi_matches_published_schedule(self):
        # ceil(0.07 * C * p * (1-p)) to the cent, per order
        self.assertEqual(FeeCalculator.calculate_fee(MarketVenue.KALSHI, 0.50, 1), 0.02)
        self.assertEqual(FeeCalculator.calculate_fee(MarketVenue.KALSHI, 0.50, 100), 1.75)
        self.assertEqual(FeeCalculator.calculate_fee(MarketVenue.KALSHI, 0.05, 1), 0.01)
        self.assertEqual(FeeCalculator.calculate_fee(MarketVenue.KALSHI, 0.50, 10, is_maker=True), 0.0)

    def test_cent_rounding_is_per_order_not_per_contract(self):
        c = MarketContract("K", "E", MarketVenue.KALSHI, "t")
        one = FeeCalculator.fee_per_contract(c, 0.5, 1)
        ten = FeeCalculator.fee_per_contract(c, 0.5, 10)
        self.assertAlmostEqual(one, 0.02)
        self.assertAlmostEqual(ten, 0.018)

    def test_polymarket_uses_the_markets_own_schedule(self):
        c = MarketContract("P", "E", MarketVenue.POLYMARKET, "t", fee_rate=0.05, fee_exponent=1.0)
        self.assertAlmostEqual(FeeCalculator.fee_for_contract(c, 0.5, 100), 100 * 0.5 * 0.05 * 0.25, places=5)
        free = MarketContract("P", "E", MarketVenue.POLYMARKET, "t", fees_enabled=False)
        self.assertEqual(FeeCalculator.fee_for_contract(free, 0.5, 100), 0.0)
        self.assertEqual(FeeCalculator.calculate_fee(MarketVenue.POLYMARKET, 0.5, 100, is_maker=True), 0.0)

    def test_fees_peak_at_fifty_cents_and_vanish_at_extremes(self):
        f = lambda p: FeeCalculator.calculate_fee(MarketVenue.POLYMARKET, p, 1000, fee_rate=0.05)
        self.assertGreater(f(0.5), f(0.2))
        self.assertGreater(f(0.2), f(0.02))


# --------------------------------------------------------------------------------------------
class TestDistributionFitting(unittest.TestCase):
    def test_pava_repairs_non_monotone_cdf(self):
        out = pava_nondecreasing([0.2, 0.5, 0.4, 0.9])
        self.assertTrue(np.all(np.diff(out) >= -1e-12))
        self.assertAlmostEqual(out[1], out[2])

    def test_fit_recovers_known_normal(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ps = [norm.cdf((x - 3.2) / 0.8) for x in xs]
        mu, sigma = EconomicBracketPricer().fit_distribution_from_brackets(xs, ps)
        self.assertAlmostEqual(mu, 3.2, places=3)
        self.assertAlmostEqual(sigma, 0.8, places=3)

    def test_unidentifiable_fit_raises_instead_of_defaulting(self):
        p = EconomicBracketPricer()
        with self.assertRaises(CalibrationError):
            p.fit_distribution_from_brackets([1, 2, 3], [0.996, 0.998, 0.999])   # all in the far tail
        with self.assertRaises(CalibrationError):
            p.fit_distribution_from_brackets([1, 2, 3], [0.5, 0.5, 0.5])         # flat CDF
        with self.assertRaises(CalibrationError):
            p.fit_distribution_from_brackets([1.0], [0.5])

    def test_noisy_illiquid_point_is_downweighted(self):
        xs = [1, 2, 3, 4, 5]
        ps = [norm.cdf((x - 3) / 1.0) for x in xs]
        ps[1] = 0.45   # a bad print at x=2
        p = EconomicBracketPricer()
        _, s_eq = p.fit_distribution_from_brackets(xs, ps)
        _, s_w = p.fit_distribution_from_brackets(xs, ps, weights=[1, 0.01, 1, 1, 1])
        self.assertLess(abs(s_w - 1.0), abs(s_eq - 1.0))

    def test_tails_are_not_clipped(self):
        p = EconomicBracketPricer()
        far = p.price_bracket(8.0, None, 3.0, 1.0)       # true value ~ 2.9e-7
        self.assertLess(far, 1e-5)
        self.assertGreater(far, 0.0)

    def test_no_strike_means_unpriceable_not_certain(self):
        self.assertIsNone(EconomicBracketPricer().price_bracket(None, None, 0.0, 1.0))

    def test_student_t_has_fatter_tails_than_normal(self):
        n = EconomicBracketPricer("normal").price_bracket(5.0, None, 0.0, 1.0)
        t = EconomicBracketPricer("student_t", df=4).price_bracket(5.0, None, 0.0, 1.0)
        self.assertGreater(t, n * 50)

    def test_lognormal_never_prices_negative_values(self):
        p = EconomicBracketPricer("lognormal")
        self.assertEqual(p.price_bracket(None, 0.0, 1.0, 0.5), 0.0)
        mu, s = p.fit_distribution_from_brackets([50, 100, 200], [norm.cdf((math.log(x) - 4.6) / 0.6) for x in (50, 100, 200)])
        self.assertAlmostEqual(mu, 4.6, places=2)


# --------------------------------------------------------------------------------------------
class TestEventPricer(unittest.TestCase):
    def tight_books(self, cs, probs, half=0.01):
        return {c.ticker: book(c.ticker, round(probs[c.ticker] - half, 4), round(probs[c.ticker] + half, 4))
                for c in cs if c.ticker in probs}

    def test_partition_fit_recovers_the_underlying_and_prices_missing_brackets(self):
        cs = gdp_style_event()
        classify_kalshi_event(cs, event_mutually_exclusive=True)
        true_mu, true_sigma = 2.6, 1.1
        def region(c):
            lo, hi = c.model_edges
            upper = 1.0 if hi is None else norm.cdf((hi - true_mu) / true_sigma)
            lower = 0.0 if lo is None else norm.cdf((lo - true_mu) / true_sigma)
            return upper - lower
        truth = {c.ticker: region(c) for c in cs}
        # only the central brackets are liquid; tails have no book at all
        liquid = {t: p for t, p in truth.items() if p > 0.05}
        books = self.tight_books(cs, liquid)
        res = EventPricer().price_event(cs, books, as_of=NOW)
        self.assertIn("partition fit", res.method)
        self.assertAlmostEqual(res.params["mu"], true_mu, delta=0.1)
        self.assertAlmostEqual(res.params["sigma"], true_sigma, delta=0.15)
        for c in cs:
            fv = res.fair_values[c.ticker]
            self.assertIsNotNone(fv.fv)
            self.assertAlmostEqual(fv.fv, truth[c.ticker], delta=0.02)
        self.assertAlmostEqual(sum(f.fv for f in res.fair_values.values()), 1.0, places=6)
        # unbooked tail brackets are priced but flagged as non-anchors with wider error bars
        tail = res.fair_values["L"]
        self.assertFalse(tail.is_anchor)
        self.assertGreater(tail.uncertainty, res.fair_values["B2.6"].uncertainty)

    def test_too_few_liquid_brackets_gives_no_price(self):
        cs = gdp_style_event()
        classify_kalshi_event(cs, event_mutually_exclusive=True)
        books = {"B2.1": book("B2.1", 0.20, 0.22), "B2.6": book("B2.6", 0.25, 0.27)}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        self.assertTrue(all(f.fv is None for f in res.fair_values.values()))

    def test_wide_spread_books_are_not_used_as_anchors(self):
        cs = gdp_style_event()
        classify_kalshi_event(cs, event_mutually_exclusive=True)
        books = {c.ticker: book(c.ticker, 0.01, 0.30) for c in cs}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        self.assertTrue(all(f.fv is None for f in res.fair_values.values()))

    def test_strikeless_partition_is_normalized_and_not_a_model_view(self):
        cs = [MarketContract(f"E{i}", "E", MarketVenue.POLYMARKET, "t", event_kind=EventKind.PARTITION,
                             exhaustive=True) for i in range(3)]
        books = {"E0": book("E0", 0.48, 0.52, MarketVenue.POLYMARKET),
                 "E1": book("E1", 0.28, 0.32, MarketVenue.POLYMARKET),
                 "E2": book("E2", 0.23, 0.27, MarketVenue.POLYMARKET)}   # mids sum to 1.05
        res = EventPricer().price_event(cs, books, as_of=NOW)
        fvs = res.fair_values
        self.assertAlmostEqual(sum(f.fv for f in fvs.values()), 1.0, places=9)
        self.assertTrue(all(f.source == SOURCE_NORMALIZED_MID and not f.has_model_view for f in fvs.values()))

    def test_strikeless_partition_with_a_missing_book_is_unpriced_not_guessed(self):
        cs = [MarketContract(f"E{i}", "E", MarketVenue.POLYMARKET, "t", event_kind=EventKind.PARTITION)
              for i in range(3)]
        books = {"E0": book("E0", 0.48, 0.52, MarketVenue.POLYMARKET)}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        self.assertTrue(all(f.fv is None for f in res.fair_values.values()))

    def test_unknown_event_is_never_priced(self):
        cs = [MarketContract(f"U{i}", "E", MarketVenue.POLYMARKET, "t") for i in range(3)]
        books = {c.ticker: book(c.ticker, 0.4, 0.42, MarketVenue.POLYMARKET) for c in cs}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        self.assertTrue(all(f.fv is None for f in res.fair_values.values()))

    def test_threshold_ladder_prices_illiquid_strikes(self):
        strikes = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        cs = [kalshi(f"G{k}", "greater", floor=k) for k in strikes]
        classify_kalshi_event(cs, event_mutually_exclusive=False)
        surv = {f"G{k}": 1.0 - norm.cdf((k - 2.0) / 0.9) for k in strikes}
        liquid = {t: p for t, p in surv.items() if t in ("G1.0", "G1.5", "G2.0", "G2.5", "G3.0")}
        res = EventPricer().price_event(cs, self.tight_books(cs, liquid), as_of=NOW)
        for t, p in surv.items():
            self.assertAlmostEqual(res.fair_values[t].fv, p, delta=0.02, msg=t)
        self.assertTrue(res.fair_values["G1.0"].is_anchor)
        self.assertFalse(res.fair_values["G4.0"].is_anchor)

    def test_by_date_curve_is_monotone_and_interpolates_with_constant_hazard(self):
        days = [90, 180, 270, 360]
        cs = [MarketContract(f"D{d}", "E", MarketVenue.POLYMARKET, "Foo by date?", event_kind=EventKind.NESTED_BY_DATE,
                             expiration_time=NOW + timedelta(days=d)) for d in days]
        h = -math.log(0.8) / 180.0     # P(by 180d) = 0.20
        p = lambda d: 1 - math.exp(-h * d)
        books = {"D90": book("D90", p(90) - 0.01, p(90) + 0.01, MarketVenue.POLYMARKET),
                 "D360": book("D360", p(360) - 0.01, p(360) + 0.01, MarketVenue.POLYMARKET)}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        fv = {t: res.fair_values[t].fv for t in res.fair_values}
        self.assertAlmostEqual(fv["D180"], p(180), delta=0.005)   # exact for a constant hazard
        self.assertAlmostEqual(fv["D270"], p(270), delta=0.005)
        ordered = [fv[f"D{d}"] for d in days]
        self.assertEqual(ordered, sorted(ordered))
        self.assertFalse(res.fair_values["D180"].is_anchor)
        self.assertTrue(res.fair_values["D90"].is_anchor)

    def test_by_date_extrapolation_beyond_last_liquid_deadline_is_not_a_model_view(self):
        # Two liquid early deadlines and a far deadline whose book says something very different.
        # A constant-hazard extrapolation must not be presented as evidence against that book.
        days = [90, 180, 360, 720]
        cs = [MarketContract(f"D{d}", "E", MarketVenue.POLYMARKET, "Foo by date?", event_kind=EventKind.NESTED_BY_DATE,
                             expiration_time=NOW + timedelta(days=d)) for d in days]
        books = {"D90": book("D90", 0.04, 0.06, MarketVenue.POLYMARKET),
                 "D180": book("D180", 0.09, 0.11, MarketVenue.POLYMARKET),
                 "D720": book("D720", 0.10, 0.90, MarketVenue.POLYMARKET)}   # illiquid, not an anchor
        res = EventPricer().price_event(cs, books, as_of=NOW)
        far = res.fair_values["D720"]
        self.assertFalse(far.has_model_view)
        self.assertGreaterEqual(far.fv, res.fair_values["D180"].fv)          # monotone lower bound respected
        self.assertGreater(far.uncertainty, 0.05)                            # and honestly wide
        between = res.fair_values["D360"]
        self.assertFalse(between.has_model_view)   # D360 is also beyond the last *anchor* (D180)

    def test_by_date_between_now_and_first_anchor_is_bounded_and_a_view(self):
        cs = [MarketContract(f"D{d}", "E", MarketVenue.POLYMARKET, "Foo by date?", event_kind=EventKind.NESTED_BY_DATE,
                             expiration_time=NOW + timedelta(days=d)) for d in (30, 90, 180)]
        books = {"D90": book("D90", 0.19, 0.21, MarketVenue.POLYMARKET),
                 "D180": book("D180", 0.29, 0.31, MarketVenue.POLYMARKET)}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        early = res.fair_values["D30"]
        self.assertTrue(early.has_model_view)
        self.assertTrue(0.0 < early.fv < res.fair_values["D90"].fv)

    def test_poor_fit_withholds_the_model_view(self):
        # a bimodal partition no single bell curve can reproduce
        cs = gdp_style_event()
        classify_kalshi_event(cs, event_mutually_exclusive=True)
        mass = {c.ticker: 0.0 for c in cs}
        mass["B0.1"], mass["B1.1"], mass["B2.1"], mass["B3.1"], mass["B4.1"], mass["B5.1"] = 0.34, 0.01, 0.01, 0.01, 0.01, 0.34
        mass["L"], mass["G"] = 0.04, 0.04
        mass["B2.6"], mass["B3.6"] = 0.0, 0.0
        books = {t: book(t, max(p - 0.01, 0.0), p + 0.01) for t, p in mass.items()}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        self.assertGreater(res.params["rmse"], 0.04)
        self.assertTrue(res.warnings)
        self.assertFalse(any(f.has_model_view for f in res.fair_values.values()))

    def test_fat_tailed_data_selects_student_t(self):
        from scipy.stats import t as student_t
        edges = [(None, 0.0)] + [(x, x + 0.5) for x in np.arange(0.0, 4.0, 0.5)] + [(4.0, None)]
        cs = []
        for i, (lo, hi) in enumerate(edges):
            c = MarketContract(f"T{i}", "E", MarketVenue.KALSHI, "t", event_kind=EventKind.PARTITION, exhaustive=True)
            c.model_edges = (lo, hi)
            cs.append(c)
        cdf = lambda x, below: (0.0 if below else 1.0) if x is None else student_t.cdf((x - 2.0) / 0.5, 4)
        probs = {c.ticker: cdf(c.model_edges[1], False) - cdf(c.model_edges[0], True) for c in cs}
        books = {t: book(t, max(p - 0.005, 0.0), p + 0.005) for t, p in probs.items()}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        self.assertEqual(res.params["dist"], "student_t")
        self.assertLess(res.params["rmse"], 0.01)

    def test_by_date_repairs_inverted_anchor_prices(self):
        cs = [MarketContract(f"D{d}", "E", MarketVenue.POLYMARKET, "Foo by date?", event_kind=EventKind.NESTED_BY_DATE,
                             expiration_time=NOW + timedelta(days=d)) for d in (30, 60, 90)]
        books = {"D30": book("D30", 0.29, 0.31, MarketVenue.POLYMARKET),
                 "D60": book("D60", 0.19, 0.21, MarketVenue.POLYMARKET),   # inverted vs D30
                 "D90": book("D90", 0.39, 0.41, MarketVenue.POLYMARKET)}
        res = EventPricer().price_event(cs, books, as_of=NOW)
        v = [res.fair_values[f"D{d}"].fv for d in (30, 60, 90)]
        self.assertEqual(v, sorted(v))

    def test_heterogeneous_event_kinds_are_refused(self):
        cs = [MarketContract("A", "E", MarketVenue.KALSHI, "t", event_kind=EventKind.PARTITION),
              MarketContract("B", "E", MarketVenue.KALSHI, "t", event_kind=EventKind.NESTED_BY_DATE)]
        res = EventPricer().price_event(cs, {}, as_of=NOW)
        self.assertTrue(all(f.fv is None for f in res.fair_values.values()))


# --------------------------------------------------------------------------------------------
class TestQuoter(unittest.TestCase):
    def test_spread_widens_with_model_uncertainty(self):
        q = MarketMakerQuoter(target_spread=0.04)
        b0, a0 = q.quote_contract(0.50, uncertainty=0.0)
        b1, a1 = q.quote_contract(0.50, uncertainty=0.03)
        self.assertGreater(a1 - b1, a0 - b0)

    def test_spread_widens_near_expiry(self):
        q = MarketMakerQuoter(target_spread=0.04)
        far = q.quote_contract(0.50, days_to_expiry=30)
        near = q.quote_contract(0.50, days_to_expiry=0.2)
        self.assertGreater(near[1] - near[0], far[1] - far[0])

    def test_longshot_tail_gets_no_bid(self):
        q = MarketMakerQuoter(target_spread=0.04)
        bid, ask = q.quote_contract(0.002, uncertainty=0.005)
        self.assertIsNone(bid)           # would pay 1c+ for something worth 0.2c
        self.assertGreaterEqual(ask, 0.01)

    def test_near_certain_contract_gets_no_ask(self):
        bid, ask = MarketMakerQuoter().quote_contract(0.995)
        self.assertIsNotNone(bid)
        self.assertIsNone(ask)

    def test_inventory_skew_is_capped_and_never_crosses_fair_value(self):
        q = MarketMakerQuoter(target_spread=0.04, inventory_skew_factor=0.001)
        for inv in (100, 10_000, -10_000):
            bid, ask = q.quote_contract(0.50, current_inventory=inv)
            self.assertLessEqual(bid, 0.50)
            self.assertGreaterEqual(ask, 0.50)
        b_long, a_long = q.quote_contract(0.50, current_inventory=10_000)
        b_flat, a_flat = q.quote_contract(0.50)
        self.assertLess(a_long, a_flat)  # long -> willing to sell cheaper
        self.assertLess(b_long, b_flat)

    def test_respects_tick_size(self):
        bid, ask = MarketMakerQuoter(target_spread=0.004).quote_contract(0.4123, tick=0.001)
        for px in (bid, ask):
            self.assertAlmostEqual(px * 1000, round(px * 1000), places=6)
        self.assertLess(bid, 0.4123)
        self.assertGreater(ask, 0.4123)

    def test_quote_from_fair_value_handles_unpriced(self):
        q = MarketMakerQuoter()
        self.assertEqual(q.quote_fair_value(FairValue("T", None, None, "unpriced", False)), (None, None))


class TestSportsGrid(unittest.TestCase):
    def test_out_of_grid_scoreline_has_zero_not_a_floor(self):
        p = SportsCopulaPricer()
        g = p.calculate_scoreline_grid(1.3, 1.1)
        self.assertEqual(p.price_exact_score(11, 0, g), 0.0)


if __name__ == "__main__":
    unittest.main()
