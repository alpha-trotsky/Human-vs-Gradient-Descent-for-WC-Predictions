"""
Pricing Models & Quoting Engine for illiquid brackets and exact scores.

Layers
------
1. EconomicBracketPricer  - distribution maths (Normal / Student-t / Log-normal): fit to liquid
                            contracts, price any bracket or tail from the fitted law.
2. EventPricer            - event-level orchestrator. Looks at what *kind* of event it is
                            (EventKind) and the live books, picks the model that is valid for it,
                            and returns a FairValue (with uncertainty and provenance) per contract.
                            Contracts it cannot price get fv=None -- never a made-up number.
3. SportsCopulaPricer     - scoreline grid for sports (Negative-Binomial + Frank copula).
4. MarketMakerQuoter      - two-sided quotes around a FairValue: spread widens with model
                            uncertainty, in the tails and near expiry; inventory skew is capped;
                            prices respect the market's tick size.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.stats import t as student_t

from copula_execution.classify import contract_edges
from copula_execution.types import EventKind, MarketContract, OrderBook


class CalibrationError(ValueError):
    """Raised when a distribution cannot be fitted from the supplied prices."""


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def pava_nondecreasing(y: Sequence[float], w: Optional[Sequence[float]] = None) -> np.ndarray:
    """Weighted isotonic regression (pool-adjacent-violators): closest non-decreasing sequence."""
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    vals: List[float] = []
    wts: List[float] = []
    cnt: List[int] = []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
            c = cnt[-2] + cnt[-1]
            vals[-2:] = [v]; wts[-2:] = [tw]; cnt[-2:] = [c]
    return np.repeat(vals, cnt)


def _mid_and_half_spread(book: Optional[OrderBook]) -> Optional[Tuple[float, float]]:
    """(mid, half-spread) for a two-sided book, else None."""
    if book is None:
        return None
    bid, ask = book.best_yes_bid, book.best_yes_ask
    if bid is None or ask is None or ask < bid:
        return None
    return (bid + ask) / 2.0, (ask - bid) / 2.0


@dataclass
class FairValue:
    ticker: str
    fv: Optional[float]               # None = could not be priced (never a placeholder number)
    uncertainty: Optional[float]      # 1-sigma of fv in probability units
    source: str                       # how it was produced, see SOURCE_*
    has_model_view: bool              # True: independent structural view (edge vs. book is meaningful)
    is_anchor: bool = False           # contract's own liquid book was used to calibrate the model
    note: str = ""


SOURCE_PARTITION_FIT = "partition_fit"
SOURCE_LADDER_FIT = "ladder_fit"
SOURCE_SURVIVAL = "survival_interp"
SOURCE_NORMALIZED_MID = "normalized_mid"
SOURCE_UNPRICED = "unpriced"


@dataclass
class EventPricing:
    event_ticker: str
    kind: str
    method: str
    fair_values: Dict[str, FairValue]
    params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------------------------
# 1. Distribution maths
# ----------------------------------------------------------------------------------------------
class EconomicBracketPricer:
    """
    Fits a continuous distribution to liquid contracts and prices any bracket / tail from it.

    assumed_dist: "normal" | "student_t" (fixed `df`, fat tails) | "lognormal" (strictly positive).
    For lognormal, `mu`/`sigma` are the parameters of ln(X).
    """

    def __init__(self, assumed_dist: str = "normal", df: float = 5.0):
        if assumed_dist not in ("normal", "student_t", "lognormal"):
            raise ValueError(f"unknown distribution {assumed_dist!r}")
        self.assumed_dist = assumed_dist
        self.df = df

    # --- standard-law primitives -------------------------------------------------------------
    def _std_cdf(self, z):
        return student_t.cdf(z, self.df) if self.assumed_dist == "student_t" else norm.cdf(z)

    def _std_ppf(self, p):
        return student_t.ppf(p, self.df) if self.assumed_dist == "student_t" else norm.ppf(p)

    def _std_pdf(self, z):
        return student_t.pdf(z, self.df) if self.assumed_dist == "student_t" else norm.pdf(z)

    def _cdf(self, x: Optional[float], mu: float, sigma: float, below: bool) -> float:
        """P(X <= x); an open edge (None) is -inf if `below` else +inf."""
        if x is None:
            return 0.0 if below else 1.0
        if self.assumed_dist == "lognormal":
            if x <= 0:
                return 0.0
            x = math.log(x)
        return float(self._std_cdf((x - mu) / sigma))

    def cdf(self, x: float, mu: float, sigma: float) -> float:
        return self._cdf(x, mu, sigma, below=True)

    def region_prob(self, lo: Optional[float], hi: Optional[float], mu: float, sigma: float) -> float:
        """P(lo < X <= hi) with None = unbounded."""
        return max(self._cdf(hi, mu, sigma, below=False) - self._cdf(lo, mu, sigma, below=True), 0.0)

    # --- fitting from a CDF ladder -----------------------------------------------------------
    def fit_distribution_from_brackets(self, strikes: Sequence[float],
                                       implied_cum_probs: Sequence[float],
                                       weights: Optional[Sequence[float]] = None) -> Tuple[float, float]:
        """
        Fit (mu, sigma) from P(X <= strike_i). Probability noise is propagated to z-space and the
        regression is z = a*x + b (errors live in p, not in x), so illiquid points count less.
        The CDF points are first made monotone (isotonic repair). Raises CalibrationError instead
        of returning a default when the data cannot identify the distribution.
        """
        xs = np.asarray(strikes, dtype=float)
        ps = np.asarray(implied_cum_probs, dtype=float)
        ws = np.ones(len(xs)) if weights is None else np.asarray(weights, dtype=float)
        if len(xs) != len(ps) or len(xs) < 2:
            raise CalibrationError("need at least two (strike, cumulative probability) points")
        order = np.argsort(xs, kind="stable")
        xs, ps, ws = xs[order], ps[order], ws[order]

        if self.assumed_dist == "lognormal":
            keep = xs > 0
            xs, ps, ws = np.log(xs[keep]), ps[keep], ws[keep]

        ps = pava_nondecreasing(ps, ws)
        mask = (ps > 0.005) & (ps < 0.995)
        if mask.sum() < 2 or len(np.unique(xs[mask])) < 2:
            raise CalibrationError(
                "fewer than two distinct strikes with cumulative probability inside (0.5%, 99.5%)")

        z = self._std_ppf(ps[mask])
        wz = ws[mask] * self._std_pdf(z) ** 2          # var(z) ~ var(p) / pdf(z)^2
        A = np.vstack([xs[mask], np.ones(mask.sum())]).T
        sw = np.sqrt(wz)
        (a, b), *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
        if not np.isfinite(a) or a <= 1e-12:
            raise CalibrationError("implied CDF is flat or decreasing in the strike")
        return float(-b / a), float(1.0 / a)

    def price_bracket(self, floor_strike: Optional[float], cap_strike: Optional[float],
                      mu: float, sigma: float) -> Optional[float]:
        """
        P(floor_strike < X <= cap_strike) under the fitted law, unclipped (tails keep their true,
        tiny probability). Returns None when the contract has no numeric strike at all -- that is
        "cannot be priced here", not "certain".
        """
        if floor_strike is None and cap_strike is None:
            return None
        return self.region_prob(floor_strike, cap_strike, mu, sigma)

    # --- fitting from a partition of brackets ------------------------------------------------
    def fit_partition(self, edges: Sequence[Tuple[Optional[float], Optional[float]]],
                      mids: Sequence[float], weights: Sequence[float]) -> Tuple[float, float, float]:
        """
        Fit (mu, sigma) so model bracket probabilities match observed mids (weighted least squares).
        Returns (mu, sigma, rmse) with rmse over the anchors in probability units.
        """
        mids = np.asarray(mids, float)
        w = np.sqrt(np.asarray(weights, float))
        lognormal = self.assumed_dist == "lognormal"
        if lognormal and any((lo is not None and lo <= 0) for lo, _ in edges):
            raise CalibrationError("lognormal needs strictly positive edges")

        # starting point from the mid-weighted moments of bracket centres
        finite = [h - l for l, h in edges if l is not None and h is not None]
        width = float(np.median(finite)) if finite else 1.0
        centres = []
        for lo, hi in edges:
            if lo is None:
                centres.append(hi - width / 2.0)
            elif hi is None:
                centres.append(lo + width / 2.0)
            else:
                centres.append((lo + hi) / 2.0)
        centres = np.asarray(centres, float)
        tot = mids.sum()
        if tot <= 0:
            raise CalibrationError("anchor mids sum to zero")
        mean0 = float((mids * centres).sum() / tot)
        var0 = float((mids * (centres - mean0) ** 2).sum() / tot) + width ** 2 / 12.0
        sd0 = math.sqrt(var0)
        if self.assumed_dist == "student_t":
            sd0 *= math.sqrt((self.df - 2.0) / self.df) if self.df > 2 else 1.0
        if lognormal:
            if mean0 <= 0:
                raise CalibrationError("lognormal start needs positive mean")
            start_mu, start_sigma = math.log(mean0), max(sd0 / mean0, 1e-3)
        else:
            start_mu, start_sigma = mean0, max(sd0, 1e-6)

        def resid(theta):
            mu, sigma = theta[0], math.exp(theta[1])
            model = np.array([self.region_prob(lo, hi, mu, sigma) for lo, hi in edges])
            return w * (model - mids)

        best = None
        for scale in (1.0, 0.5, 2.0):   # multi-start: the bracket likelihood is flat in the tails
            x0 = np.array([start_mu, math.log(start_sigma * scale)])
            try:
                sol = least_squares(resid, x0, x_scale="jac")
            except (ValueError, FloatingPointError):
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is None:
            raise CalibrationError("partition fit failed to converge")
        mu, sigma = float(best.x[0]), float(math.exp(best.x[1]))
        model = np.array([self.region_prob(lo, hi, mu, sigma) for lo, hi in edges])
        rmse = float(np.sqrt(np.mean((model - mids) ** 2)))
        return mu, sigma, rmse


# ----------------------------------------------------------------------------------------------
# 2. Event-level pricer
# ----------------------------------------------------------------------------------------------
class EventPricer:
    """
    Prices every contract of an event from that event's own live books.

      PARTITION (numeric edges)  -> fit a distribution to liquid brackets, price every bracket
      PARTITION (no numeric)     -> market-implied normalisation; flagged as *not* a model view
      CUMULATIVE_THRESHOLD       -> fit a CDF through the liquid thresholds
      NESTED_BY_DATE             -> monotone (isotonic) curve + constant-hazard interpolation
      UNKNOWN                    -> unpriced

    Uncertainty = fit error on the anchors (+ extra for extrapolated contracts) combined with a
    relative term, so tails carry wide error bars instead of a clipped floor.
    """

    def __init__(self, bracket_pricer: Optional[EconomicBracketPricer] = None,
                 max_anchor_spread: float = 0.06, min_anchors: int = 3, min_date_anchors: int = 2,
                 max_fit_rmse: float = 0.04, rel_uncertainty: float = 0.25,
                 min_uncertainty: float = 0.005, extrapolation_uncertainty: float = 0.04,
                 fat_tail_df: Optional[float] = 4.0):
        self.bp = bracket_pricer or EconomicBracketPricer()
        # With a Normal primary, a Student-t (df=fat_tail_df) is also fitted and the better fit wins.
        self.fat_tail_df = fat_tail_df
        self.max_anchor_spread = max_anchor_spread
        self.min_anchors = min_anchors
        self.min_date_anchors = min_date_anchors
        self.max_fit_rmse = max_fit_rmse
        self.rel_uncertainty = rel_uncertainty
        self.min_uncertainty = min_uncertainty
        self.extrapolation_uncertainty = extrapolation_uncertainty

    # -- public ------------------------------------------------------------------------------
    def price_event(self, contracts: List[MarketContract], books: Dict[str, OrderBook],
                    as_of: Optional[datetime] = None) -> EventPricing:
        ev = contracts[0].event_ticker if contracts else ""
        if not contracts:
            return EventPricing(ev, EventKind.UNKNOWN.value, "none", {})
        kinds = {c.event_kind for c in contracts}
        if len(kinds) != 1:
            return self._unpriced(contracts, ev, EventKind.UNKNOWN, "contracts disagree on event kind")
        kind = next(iter(kinds))
        anchors = self._anchors(contracts, books)
        as_of = as_of or datetime.now(timezone.utc)

        if kind == EventKind.PARTITION:
            if all(contract_edges(c) is not None for c in contracts):
                return self._price_partition(contracts, anchors, ev)
            return self._price_normalized(contracts, books, ev)
        if kind == EventKind.CUMULATIVE_THRESHOLD:
            return self._price_ladder(contracts, anchors, ev)
        if kind == EventKind.NESTED_BY_DATE:
            return self._price_by_date(contracts, anchors, ev, as_of)
        return self._unpriced(contracts, ev, kind, "relationship between contracts not established")

    # -- internals ---------------------------------------------------------------------------
    def _anchors(self, contracts, books) -> Dict[str, Tuple[float, float]]:
        """ticker -> (mid, half_spread) for contracts whose own book is tight enough to trust."""
        out = {}
        for c in contracts:
            ms = _mid_and_half_spread(books.get(c.ticker))
            if ms is None:
                continue
            mid, half = ms
            if 2 * half <= self.max_anchor_spread and 0.0 < mid < 1.0:
                out[c.ticker] = (mid, half)
        return out

    def _candidates(self) -> List[EconomicBracketPricer]:
        cands = [self.bp]
        if self.bp.assumed_dist == "normal" and self.fat_tail_df:
            cands.append(EconomicBracketPricer("student_t", df=self.fat_tail_df))
        return cands

    @staticmethod
    def _weight(half_spread: float) -> float:
        return 1.0 / max(half_spread, 0.005) ** 2

    def _unc(self, fv: float, rmse: float, anchor: bool, extra_abs: float = 0.0) -> float:
        base = max(rmse, self.min_uncertainty)
        if anchor:
            return base
        abs_unc = base * 1.5 + extra_abs
        return math.sqrt(abs_unc ** 2 + (self.rel_uncertainty * fv) ** 2)

    def _unpriced(self, contracts, ev, kind, why) -> EventPricing:
        fvs = {c.ticker: FairValue(c.ticker, None, None, SOURCE_UNPRICED, False, note=why) for c in contracts}
        kind_v = kind.value if isinstance(kind, EventKind) else str(kind)
        return EventPricing(ev, kind_v, "unpriced", fvs, warnings=[why])

    def _price_normalized(self, contracts, books, ev) -> EventPricing:
        mids = {}
        for c in contracts:
            ms = _mid_and_half_spread(books.get(c.ticker))
            if ms is not None:
                mids[c.ticker] = ms
        if len(mids) < len(contracts) or sum(m for m, _ in mids.values()) <= 0:
            return self._unpriced(contracts, ev, EventKind.PARTITION,
                                  "no numeric structure and not every leg has a two-sided book")
        total = sum(m for m, _ in mids.values())
        fvs = {}
        for c in contracts:
            mid, half = mids[c.ticker]
            fvs[c.ticker] = FairValue(c.ticker, mid / total, max(half, self.min_uncertainty),
                                      SOURCE_NORMALIZED_MID, False, is_anchor=True,
                                      note="market mid rescaled to sum to 1 (no independent model view)")
        return EventPricing(ev, EventKind.PARTITION.value, "normalized market mids", fvs,
                            params={"overround": total - 1.0})

    def _price_partition(self, contracts, anchors, ev) -> EventPricing:
        edges_all = {c.ticker: contract_edges(c) for c in contracts}
        anch = [c for c in contracts if c.ticker in anchors]
        if len(anch) < self.min_anchors:
            return self._unpriced(contracts, ev, EventKind.PARTITION,
                                  f"only {len(anch)} liquid brackets (need {self.min_anchors})")
        fits = []
        last_err = None
        for bp in self._candidates():
            try:
                fits.append((bp, *bp.fit_partition(
                    [edges_all[c.ticker] for c in anch],
                    [anchors[c.ticker][0] for c in anch],
                    [self._weight(anchors[c.ticker][1]) for c in anch])))
            except CalibrationError as e:
                last_err = e
        if not fits:
            return self._unpriced(contracts, ev, EventKind.PARTITION, f"calibration failed: {last_err}")
        bp, mu, sigma, rmse = min(fits, key=lambda f: f[3])
        warnings = []
        good = rmse <= self.max_fit_rmse
        if not good:
            warnings.append(f"poor fit (rmse {rmse:.3f} > {self.max_fit_rmse}): the distribution shape does "
                            "not suit this event -- no model view, uncertainties inflated")
        fvs = {}
        for c in contracts:
            lo, hi = edges_all[c.ticker]
            fv = bp.region_prob(lo, hi, mu, sigma)
            is_a = c.ticker in anchors
            fvs[c.ticker] = FairValue(c.ticker, fv, self._unc(fv, rmse, is_a), SOURCE_PARTITION_FIT,
                                      good, is_anchor=is_a,
                                      note="" if good else "poor fit: not a reliable view")
        return EventPricing(ev, EventKind.PARTITION.value, f"{bp.assumed_dist} partition fit", fvs,
                            params={"mu": mu, "sigma": sigma, "dist": bp.assumed_dist,
                                    "rmse": rmse, "n_anchors": len(anch)}, warnings=warnings)

    def _price_ladder(self, contracts, anchors, ev) -> EventPricing:
        # Every contract is P(X > k) (edges (k, None)) or P(X < k) (edges (None, k)); both are CDF points.
        pts = []   # (strike, cdf, weight, ticker)
        for c in contracts:
            e = contract_edges(c)
            if e is None or (e[0] is None) == (e[1] is None):
                return self._unpriced(contracts, ev, EventKind.CUMULATIVE_THRESHOLD,
                                      "threshold contract without a single strike")
            if c.ticker not in anchors:
                continue
            mid, half = anchors[c.ticker]
            if e[1] is None:
                pts.append((e[0], 1.0 - mid, self._weight(half), c.ticker))
            else:
                pts.append((e[1], mid, self._weight(half), c.ticker))
        if len(pts) < self.min_anchors:
            return self._unpriced(contracts, ev, EventKind.CUMULATIVE_THRESHOLD,
                                  f"only {len(pts)} liquid thresholds (need {self.min_anchors})")
        fits = []
        last_err = None
        for bp in self._candidates():
            try:
                mu, sigma = bp.fit_distribution_from_brackets(
                    [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
            except CalibrationError as e:
                last_err = e
                continue
            rmse = float(np.sqrt(np.mean([
                (bp.region_prob(*contract_edges(c), mu, sigma) - anchors[c.ticker][0]) ** 2
                for c in contracts if c.ticker in anchors])))
            fits.append((bp, mu, sigma, rmse))
        if not fits:
            return self._unpriced(contracts, ev, EventKind.CUMULATIVE_THRESHOLD, f"calibration failed: {last_err}")
        bp, mu, sigma, rmse = min(fits, key=lambda f: f[3])
        warnings = []
        good = rmse <= self.max_fit_rmse
        if not good:
            warnings.append(f"poor fit (rmse {rmse:.3f} > {self.max_fit_rmse}): the distribution shape does "
                            "not suit this event -- no model view, uncertainties inflated")
        fvs = {}
        for c in contracts:
            fv = bp.region_prob(*contract_edges(c), mu, sigma)
            is_a = c.ticker in anchors
            fvs[c.ticker] = FairValue(c.ticker, fv, self._unc(fv, rmse, is_a), SOURCE_LADDER_FIT,
                                      good, is_anchor=is_a,
                                      note="" if good else "poor fit: not a reliable view")
        return EventPricing(ev, EventKind.CUMULATIVE_THRESHOLD.value, f"{bp.assumed_dist} CDF ladder fit",
                            fvs, params={"mu": mu, "sigma": sigma, "dist": bp.assumed_dist,
                                         "rmse": rmse, "n_anchors": len(pts)}, warnings=warnings)

    def _price_by_date(self, contracts, anchors, ev, as_of) -> EventPricing:
        if any(c.expiration_time is None for c in contracts):
            return self._unpriced(contracts, ev, EventKind.NESTED_BY_DATE, "contract without a deadline")
        t0 = as_of.timestamp()
        T = {c.ticker: c.expiration_time.timestamp() for c in contracts}
        anch = sorted((c for c in contracts if c.ticker in anchors), key=lambda c: T[c.ticker])
        anch = [c for c in anch if T[c.ticker] > t0]
        if len(anch) < self.min_date_anchors:
            return self._unpriced(contracts, ev, EventKind.NESTED_BY_DATE,
                                  f"only {len(anch)} liquid deadlines (need {self.min_date_anchors})")

        raw = np.array([anchors[c.ticker][0] for c in anch])
        w = np.array([self._weight(anchors[c.ticker][1]) for c in anch])
        smooth = np.clip(pava_nondecreasing(raw, w), 1e-4, 0.999)   # P(by t) cannot fall as t grows
        rmse = float(np.sqrt(np.mean((smooth - raw) ** 2)))
        ta = np.array([T[c.ticker] for c in anch])
        lnS = np.log(1.0 - smooth)                                   # log-survival, non-increasing
        by_ticker = {c.ticker: float(smooth[i]) for i, c in enumerate(anch)}

        # constant-hazard extrapolation after the last anchor
        if len(anch) >= 2 and lnS[-1] < lnS[-2] and ta[-1] > ta[-2]:
            tail_h = (lnS[-1] - lnS[-2]) / (ta[-1] - ta[-2])
        else:
            tail_h = lnS[-1] / (ta[-1] - t0)

        fvs = {}
        for c in contracts:
            t = T[c.ticker]
            if t <= t0:
                fvs[c.ticker] = FairValue(c.ticker, None, None, SOURCE_UNPRICED, False,
                                          note="deadline already passed")
                continue
            if c.ticker in by_ticker:
                fv = by_ticker[c.ticker]
                fvs[c.ticker] = FairValue(c.ticker, fv, self._unc(fv, rmse, True), SOURCE_SURVIVAL, True,
                                          is_anchor=True)
                continue
            j = int(np.searchsorted(ta, t))
            view = True
            if j == 0:        # between "now" (P=0) and the first liquid deadline: bounded on both sides
                ls = lnS[0] * (t - t0) / (ta[0] - t0)
                extra = 0.25 * smooth[0]
                note = "interpolated between now and first liquid deadline"
            elif j >= len(ta):  # beyond the last anchor: monotonicity only gives a lower bound
                ls = lnS[-1] + tail_h * (t - ta[-1])
                fv_tmp = float(1.0 - math.exp(ls))
                extra = self.extrapolation_uncertainty + 0.5 * fv_tmp
                view = False   # a constant-hazard extrapolation is not evidence against the book
                note = (f"extrapolated beyond last liquid deadline: only P >= {smooth[-1]:.3f} is implied; "
                        "no model view")
            else:             # between two anchors
                frac = (t - ta[j - 1]) / (ta[j] - ta[j - 1])
                ls = lnS[j - 1] + (lnS[j] - lnS[j - 1]) * frac
                extra = 0.25 * (smooth[j] - smooth[j - 1])
                note = "interpolated between liquid deadlines"
            fv = float(1.0 - math.exp(ls))
            fvs[c.ticker] = FairValue(c.ticker, fv, self._unc(fv, rmse, False, extra), SOURCE_SURVIVAL,
                                      view, note=note)
        return EventPricing(ev, EventKind.NESTED_BY_DATE.value, "monotone survival interpolation", fvs,
                            params={"rmse": rmse, "n_anchors": len(anch)})


# ----------------------------------------------------------------------------------------------
# 3. Sports
# ----------------------------------------------------------------------------------------------
class SportsCopulaPricer:
    """
    Connects to models.py and core.McHale_Copula to produce full 11x11 scoreline distributions
    and derive prices for exact score, totals, spreads, and combos.
    """

    def __init__(self):
        try:
            from core.McHale_Copula import score_matrix
            self._score_matrix_fn = score_matrix
        except ImportError:
            self._score_matrix_fn = None

    def calculate_scoreline_grid(self, mu_h: float, mu_a: float,
                                 r_h: float = 8.0, r_a: float = 8.0,
                                 theta: float = 0.15, max_goals: int = 10) -> np.ndarray:
        if self._score_matrix_fn:
            return self._score_matrix_fn(mu_h, mu_a, r_h, r_a, theta, max_goals=max_goals)

        # Fallback independent Poisson grid if McHale copula not imported
        from scipy.stats import poisson
        grid = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                grid[h, a] = poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a)
        return grid / grid.sum()

    def price_exact_score(self, home_goals: int, away_goals: int, grid: np.ndarray) -> float:
        """Probability of the scoreline; 0.0 outside the truncation grid (mass there is negligible)."""
        if 0 <= home_goals < grid.shape[0] and 0 <= away_goals < grid.shape[1]:
            return float(grid[home_goals, away_goals])
        return 0.0


# ----------------------------------------------------------------------------------------------
# 4. Quoter
# ----------------------------------------------------------------------------------------------
class MarketMakerQuoter:
    """
    Two-sided quotes around a fair value.

      half_spread = target_spread/2 + uncertainty_mult * fv_uncertainty        (model risk)
                    * (1 + 0.5 * (expiry_widen_days - days_left)/expiry_widen_days)  near expiry
      inventory skew = inventory_skew_factor * inventory, capped at +-half_spread so quotes never
                       cross to the wrong side of fair value
      bid = floor_to_tick(fv - half - skew), ask = ceil_to_tick(fv + half - skew)

    A side is omitted (None) when it cannot be quoted without giving away value -- e.g. no bid when
    fair value minus the spread is below one tick (long-shot tails), no ask when it would exceed
    1 - tick.
    """

    def __init__(self, target_spread: float = 0.04, inventory_skew_factor: float = 0.001,
                 uncertainty_mult: float = 1.0, expiry_widen_days: float = 3.0):
        self.target_spread = target_spread
        self.inventory_skew_factor = inventory_skew_factor
        self.uncertainty_mult = uncertainty_mult
        self.expiry_widen_days = expiry_widen_days

    @staticmethod
    def _floor_tick(x: float, tick: float) -> float:
        return round(math.floor(round(x / tick, 6)) * tick, 6)

    @staticmethod
    def _ceil_tick(x: float, tick: float) -> float:
        return round(math.ceil(round(x / tick, 6)) * tick, 6)

    def half_spread(self, uncertainty: float = 0.0, tick: float = 0.01,
                    days_to_expiry: Optional[float] = None) -> float:
        half = self.target_spread / 2.0 + self.uncertainty_mult * max(uncertainty, 0.0)
        if days_to_expiry is not None and days_to_expiry < self.expiry_widen_days:
            left = max(days_to_expiry, 0.0)
            half *= 1.0 + 0.5 * (self.expiry_widen_days - left) / self.expiry_widen_days
        return max(half, tick)

    def quote_contract(self, fair_value: float, current_inventory: int = 0, uncertainty: float = 0.0,
                       tick: float = 0.01, days_to_expiry: Optional[float] = None
                       ) -> Tuple[Optional[float], Optional[float]]:
        half = self.half_spread(uncertainty, tick, days_to_expiry)
        skew = float(np.clip(self.inventory_skew_factor * current_inventory, -half, half))
        raw_bid = fair_value - half - skew
        raw_ask = fair_value + half - skew

        bid = self._floor_tick(raw_bid, tick)
        ask = self._ceil_tick(raw_ask, tick)
        bid_out: Optional[float] = bid if bid >= tick else None
        ask_out: Optional[float] = ask if ask <= 1.0 - tick + 1e-9 else None
        if bid_out is not None and ask_out is not None and ask_out <= bid_out:
            ask_out = round(bid_out + tick, 6)
            if ask_out > 1.0 - tick + 1e-9:
                ask_out = None
        return bid_out, ask_out

    def quote_fair_value(self, fv: FairValue, contract: Optional[MarketContract] = None,
                         inventory: int = 0, as_of: Optional[datetime] = None
                         ) -> Tuple[Optional[float], Optional[float]]:
        """Quote straight from a FairValue; (None, None) when the contract could not be priced."""
        if fv.fv is None:
            return None, None
        tick = contract.tick_size if contract else 0.01
        days = None
        if contract is not None and contract.expiration_time is not None:
            now = as_of or datetime.now(timezone.utc)
            days = (contract.expiration_time - now).total_seconds() / 86400.0
        return self.quote_contract(fv.fv, inventory, fv.uncertainty or 0.0, tick, days)
