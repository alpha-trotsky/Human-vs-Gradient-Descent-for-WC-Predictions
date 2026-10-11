"""
Pricing Models & Quoting Engine for Illiquid Brackets and Exact Scores.

Provides two-sided market making (bid/ask quotes) calibrated from liquid contracts:
1. Economics (Gaussian / Student-t / Log-normal continuous distribution calibration)
2. Sports (Negative-Binomial + Frank Copula joint grid calibration from models.py / McHale_Copula)
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple

from copula_execution.types import MarketContract, OrderBook


class EconomicBracketPricer:
    """
    Fits continuous distribution to liquid contracts (e.g. CPI/GDP/Fed target brackets)
    and prices every thin bracket and tail outcome.
    """

    def __init__(self, assumed_dist: str = "normal"):
        self.assumed_dist = assumed_dist

    def fit_distribution_from_brackets(self, strikes: List[float],
                                       implied_cum_probs: List[float]) -> Tuple[float, float]:
        """
        Fits mu and sigma of underlying economic variable from known cumulative probabilities.
        strikes: list of thresholds X
        implied_cum_probs: P(X <= threshold)
        """
        if len(strikes) < 2:
            return float(np.mean(strikes)), 1.0

        # Linear regression on standard normal quantiles:
        # z_i = Phi^{-1}(P(X <= x_i)) = (x_i - mu) / sigma
        # => x_i = mu + sigma * z_i
        valid_pairs = []
        for x, p in zip(strikes, implied_cum_probs):
            if 0.005 < p < 0.995:
                z = norm.ppf(p)
                valid_pairs.append((x, z))

        if len(valid_pairs) < 2:
            mu = float(np.mean(strikes))
            sigma = 1.0
            return mu, sigma

        xs = np.array([p[0] for p in valid_pairs])
        zs = np.array([p[1] for p in valid_pairs])

        # x = mu + sigma * z
        # Slope is sigma, intercept is mu
        A = np.vstack([zs, np.ones(len(zs))]).T
        sigma, mu = np.linalg.lstsq(A, xs, rcond=None)[0]
        sigma = max(0.05, float(sigma))
        mu = float(mu)
        return mu, sigma

    def price_bracket(self, floor_strike: Optional[float],
                      cap_strike: Optional[float],
                      mu: float, sigma: float) -> float:
        """
        Computes P(floor_strike < X <= cap_strike).
        """
        if floor_strike is None and cap_strike is not None:
            # Below cap_strike
            p = norm.cdf((cap_strike - mu) / sigma)
        elif floor_strike is not None and cap_strike is None:
            # Above floor_strike
            p = 1.0 - norm.cdf((floor_strike - mu) / sigma)
        elif floor_strike is not None and cap_strike is not None:
            p_upper = norm.cdf((cap_strike - mu) / sigma)
            p_lower = norm.cdf((floor_strike - mu) / sigma)
            p = p_upper - p_lower
        else:
            p = 1.0
        return float(np.clip(p, 0.001, 0.999))


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
        if 0 <= home_goals < grid.shape[0] and 0 <= away_goals < grid.shape[1]:
            return float(grid[home_goals, away_goals])
        return 0.001


class MarketMakerQuoter:
    """
    Long-tail two-sided quoting engine:
    Generates competitive bid/ask quotes around model fair values:
      - Bid = fair_value - half_spread - skew(inventory)
      - Ask = fair_value + half_spread - skew(inventory)
    """

    def __init__(self, target_spread: float = 0.04, min_spread: float = 0.02, inventory_skew_factor: float = 0.001):
        self.target_spread = target_spread
        self.min_spread = min_spread
        self.inventory_skew_factor = inventory_skew_factor

    def quote_contract(self, fair_value: float, current_inventory: int = 0) -> Tuple[float, float]:
        half_spread = max(self.min_spread / 2.0, self.target_spread / 2.0)
        # Skew downward if long, upward if short
        inventory_skew = self.inventory_skew_factor * current_inventory

        raw_bid = fair_value - half_spread - inventory_skew
        raw_ask = fair_value + half_spread - inventory_skew

        # Bound to valid prediction market cents range [0.01, 0.99]
        bid = round(float(np.clip(raw_bid, 0.01, 0.98)), 2)
        ask = round(float(np.clip(raw_ask, bid + 0.01, 0.99)), 2)
        return bid, ask
