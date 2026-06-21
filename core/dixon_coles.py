import numpy as np
from scipy.stats import poisson

def dixon_coles(lambda_home, lambda_away, rho, max_goals=10):
    """W/D/L probabilities from two Poissons with the Dixon-Coles low-score correction.
    lambda_home/away come from either the linear regression or the MLP — same function for both."""
    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, lambda_home),          # rows = home goals
                 poisson.pmf(g, lambda_away))          # cols = away goals

    # tau: four cells, each its own factor
    M[0, 0] *= 1 - lambda_home * lambda_away * rho
    M[0, 1] *= 1 + lambda_home * rho
    M[1, 0] *= 1 + lambda_away * rho
    M[1, 1] *= 1 - rho

    M = np.clip(M, 0, None)
    M /= M.sum()                                       # renormalise (tau + truncation shift mass)

    home = np.tril(M, -1).sum()                        # home goals > away  -> below diagonal
    draw = np.trace(M)                                 # i == j            -> the diagonal
    away = np.triu(M,  1).sum()                        # away goals > home -> above diagonal
    return home, draw, away