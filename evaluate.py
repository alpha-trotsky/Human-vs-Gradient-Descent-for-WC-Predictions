"""
Evaluation metrics for the Dixon-Coles match models.

The headline metric is the Ranked Probability Score (RPS), the standard scoring
rule for football 1X2 forecasts. Each model outputs (lambda_home, lambda_away, rho);
dixon_coles() turns those into (P_home, P_draw, P_away), and we score those
ordered probabilities against the actual outcome.
"""
import numpy as np
import torch
from dixon_coles import dixon_coles


# ---------------------------------------------------------------------------
# Turning model outputs into 1X2 probabilities
# ---------------------------------------------------------------------------
def outcome_probs_from_lambdas(lambda_home, lambda_away, rho, max_goals=10):
    """Map per-match (lambda_home, lambda_away, rho) to an (n, 3) array of
    [P(home win), P(draw), P(away win)] using the Dixon-Coles scoreline grid."""
    rho = np.broadcast_to(np.asarray(rho, dtype=float), np.shape(lambda_home))
    probs = np.array([
        dixon_coles(lh, la, r, max_goals)
        for lh, la, r in zip(lambda_home, lambda_away, rho)
    ])
    return probs  # columns ordered [home, draw, away]


def actual_outcomes(home_goals, away_goals):
    """Encode results as ordinal classes: 0 = home win, 1 = draw, 2 = away win."""
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    return np.where(home_goals > away_goals, 0,
                    np.where(home_goals == away_goals, 1, 2))


# ---------------------------------------------------------------------------
# Model -> probabilities adapters
# ---------------------------------------------------------------------------
def predict_linear(model, X):
    """(lambda_home, lambda_away, rho-per-match) for the scipy linear model."""
    lambda_home, lambda_away = model.predict(X)
    rho = np.full(len(X), model.coefficients[-1])
    return lambda_home, lambda_away, rho


def predict_mlp(model, X):
    """(lambda_home, lambda_away, rho) for the MLP, applying the SAME feature
    standardization that was fit on the training set (stored on the model)."""
    X_scaled = (X - model.feature_mean) / model.feature_std
    model.eval()
    with torch.no_grad():
        lambda_home, lambda_away, rho = model(torch.FloatTensor(X_scaled))
    return lambda_home.numpy(), lambda_away.numpy(), rho.numpy()


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------
def ranked_probability_score(probs, outcomes):
    """
    Ranked Probability Score for ordinal 1X2 outcomes.

    For each match with predicted CDF F (cumulative over [home, draw, away]) and
    observed CDF O (a step from 0->1 at the realized class):

        RPS = 1/(r-1) * sum_{k=1}^{r-1} (F_k - O_k)^2     (r = 3 classes)

    Lower is better. A perfect, certain forecast scores 0. Because it works on the
    cumulative distribution it rewards being "close" on the ordered scale: putting
    mass on draw when the result was a home win is penalized less than putting it
    on an away win.

    Returns (mean_rps, per_match_rps).
    """
    probs = np.asarray(probs, dtype=float)
    n, r = probs.shape
    cum_pred = np.cumsum(probs, axis=1)

    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1.0
    cum_obs = np.cumsum(onehot, axis=1)

    # Only the first r-1 cumulative terms are free (the last is always 1 - 1 = 0).
    per_match = ((cum_pred - cum_obs) ** 2).sum(axis=1) / (r - 1)
    return per_match.mean(), per_match


def log_loss(probs, outcomes, eps=1e-15):
    """Multiclass negative log-likelihood of the realized outcomes."""
    probs = np.asarray(probs, dtype=float)
    p_true = probs[np.arange(len(outcomes)), outcomes]
    return -np.log(np.clip(p_true, eps, 1.0)).mean()


def brier_score(probs, outcomes):
    """Multiclass Brier score (mean squared error vs. the one-hot outcome)."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return ((probs - onehot) ** 2).sum(axis=1).mean()


def accuracy(probs, outcomes):
    """Share of matches where the most likely predicted class was correct."""
    return (np.asarray(probs).argmax(axis=1) == outcomes).mean()


# ---------------------------------------------------------------------------
# Baseline + reporting
# ---------------------------------------------------------------------------
def base_rate_probs(train_home_goals, train_away_goals, n):
    """Constant baseline: predict the training-set 1X2 frequencies for every match."""
    train_out = actual_outcomes(train_home_goals, train_away_goals)
    freqs = np.bincount(train_out, minlength=3) / len(train_out)
    return np.tile(freqs, (n, 1))


def score_block(name, probs, outcomes):
    mean_rps, _ = ranked_probability_score(probs, outcomes)
    return {
        'model':    name,
        'RPS':      mean_rps,
        'log_loss': log_loss(probs, outcomes),
        'Brier':    brier_score(probs, outcomes),
        'accuracy': accuracy(probs, outcomes),
    }


def evaluate_models(linear_model, mlp_model, X_val, home_goals_val, away_goals_val,
                    train_home_goals, train_away_goals):
    """Score the linear model, the MLP, and a base-rate baseline on the holdout."""
    outcomes = actual_outcomes(home_goals_val, away_goals_val)

    lin_probs = outcome_probs_from_lambdas(*predict_linear(linear_model, X_val))
    mlp_probs = outcome_probs_from_lambdas(*predict_mlp(mlp_model, X_val))
    base_probs = base_rate_probs(train_home_goals, train_away_goals, len(outcomes))

    rows = [
        score_block('Base rate',         base_probs, outcomes),
        score_block('Linear regression', lin_probs,  outcomes),
        score_block('MLP (neural net)',  mlp_probs,  outcomes),
    ]

    print("\n" + "=" * 70)
    print(f"VALIDATION METRICS  (n = {len(outcomes)} matches)")
    print("=" * 70)
    print(f"{'model':20s} {'RPS':>8s} {'log_loss':>10s} {'Brier':>8s} {'accuracy':>9s}")
    print("-" * 70)
    for r in rows:
        print(f"{r['model']:20s} {r['RPS']:8.4f} {r['log_loss']:10.4f} "
              f"{r['Brier']:8.4f} {r['accuracy']:9.3f}")
    print("-" * 70)
    print("Lower RPS / log_loss / Brier = better; higher accuracy = better.")
    print("A model earns its keep only by beating the base-rate baseline.")
    return rows
