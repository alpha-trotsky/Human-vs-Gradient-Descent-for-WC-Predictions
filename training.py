import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy, torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson
from dixon_coles import dixon_coles

# ============================================================================
# LINEAR REGRESSION TRAINING FOR DIXON-COLES MODEL
# ============================================================================

class LinearRegressionDixonColes:
    """
    Linear regression model optimized for Dixon-Coles likelihood.
    
    Two separate linear models:
    - lambda_home = beta_0_h + sum(beta_i_h * feature_i)
    - lambda_away = beta_0_a + sum(beta_i_a * feature_i)
    
    Plus shared rho parameter for correlation between home/away goals.
    """
    
    def __init__(self, n_features):
        """
        Args:
            n_features: number of input features (excluding intercept)
        """
        self.n_features = n_features
        # Parameters: [beta_0_h, beta_1_h, ..., beta_n_h, beta_0_a, beta_1_a, ..., beta_n_a, rho]
        self.n_params = 2 * (n_features + 1) + 1
        self.coefficients = None
        self.feature_names = None
        # Feature standardization stats, fit once on the training set in fit()
        self.feature_mean = None
        self.feature_std = None
        
    def _predict_lambdas(self, X, params):
        """
        Predict lambda_home and lambda_away for each match.
        
        Args:
            X: feature matrix (n_matches, n_features)
            params: parameter vector of length 2*(n_features+1)+1
            
        Returns:
            lambda_home, lambda_away: arrays of shape (n_matches,)
        """
        beta_h = params[:self.n_features + 1]
        beta_a = params[self.n_features + 1:2 * (self.n_features + 1)]

        # Standardize features with the train-set stats stored in fit() (identity
        # if called before fit). The intercept column is added AFTER scaling so the
        # intercept coefficient is never normalized away.
        if self.feature_mean is not None:
            X = (X - self.feature_mean) / self.feature_std

        # Append intercept column (all ones) to the standardized features
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        lambda_home = np.dot(X_with_intercept, beta_h)
        lambda_away = np.dot(X_with_intercept, beta_a)
        
        # Ensure lambdas are positive (expected goals must be > 0)
        lambda_home = np.clip(lambda_home, 1e-6, 1e6)
        lambda_away = np.clip(lambda_away, 1e-6, 1e6)
        
        return lambda_home, lambda_away
    
    def _negative_log_likelihood(self, params, X, home_goals, away_goals, weights=None, alpha=0.0):
        """
        Calculate negative log-likelihood using Poisson distribution with optional weighting.
        
        For each match, given predicted lambdas and observed scores,
        calculate P(home_goals | lambda_home) * P(away_goals | lambda_away)
        using Poisson PMF.
        
        Args:
            params: parameter vector
            X: feature matrix
            home_goals: observed home goals
            away_goals: observed away goals
            weights: optional match weights for temporal decay (shape: n_matches)
            
        Returns:
            negative log-likelihood (scalar)
        """
        lambda_home, lambda_away = self._predict_lambdas(X, params)
        rho = params[-1]
        
        # Clip rho to valid range [-1, 1]
        rho = np.clip(rho, -0.99, 0.99)
        
        # Calculate Poisson likelihoods
        p_home = poisson.pmf(home_goals, lambda_home)
        p_away = poisson.pmf(away_goals, lambda_away)

        # Dixon-Coles low-score correction (tau): adjusts the four low-score cells
        # so 0-0/1-0/0-1/1-1 results aren't treated as independent. rho enters here.
        tau = np.ones_like(lambda_home)
        m00 = (home_goals == 0) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m10 = (home_goals == 1) & (away_goals == 0)
        m11 = (home_goals == 1) & (away_goals == 1)
        tau[m00] = 1 - lambda_home[m00] * lambda_away[m00] * rho
        tau[m01] = 1 + lambda_home[m01] * rho
        tau[m10] = 1 + lambda_away[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-10, None)  # keep positive so log(tau) is finite

        log_likelihood = np.log(p_home + 1e-10) + np.log(p_away + 1e-10) + np.log(tau)

        # Penalize if any likelihood is extremely low
        log_likelihood = np.where(log_likelihood < -10, -10, log_likelihood)
        
        # Apply weights if provided (for temporal decay)
        if weights is not None:
            log_likelihood = log_likelihood * weights

        nll = -np.sum(log_likelihood)

        # L2 (ridge) penalty on the slope coefficients only — NOT the two intercepts
        # (indices 0 and n_features+1) or rho (last). Features are standardized, so
        # penalizing every slope equally is well-posed. Tames multicollinearity.
        if alpha > 0:
            beta_h_slopes = params[1:self.n_features + 1]
            beta_a_slopes = params[self.n_features + 2:2 * (self.n_features + 1)]
            nll = nll + alpha * (np.sum(beta_h_slopes ** 2) + np.sum(beta_a_slopes ** 2))

        return nll
    
    def fit(self, X, home_goals, away_goals, weights=None, method='L-BFGS-B', max_iter=1000,
            alpha=0.0, verbose=True):
        """
        Optimize model parameters to maximize Dixon-Coles likelihood.
        
        Args:
            X: feature matrix (n_matches, n_features)
            home_goals: observed home team goals
            away_goals: observed away team goals
            weights: optional match weights for temporal decay
            method: scipy.optimize method
            max_iter: maximum iterations
            
        Returns:
            optimization result object
        """
        # Standardize features ONCE on the training set; stored and reused for
        # every NLL evaluation during fit and for all later predictions.
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8

        # Initial parameter guess: small positive values
        x0 = np.random.normal(0.1, 0.01, self.n_params)
        x0[-1] = 0.0  # rho starts at 0
        
        # Bounds: lambdas should be positive, rho in [-1, 1]
        bounds = [(-10, 10)] * (2 * (self.n_features + 1)) + [(-0.99, 0.99)]
        
        self.alpha = alpha
        callback = None
        if verbose:
            callback = lambda xk: print(
                f"  NLL: {self._negative_log_likelihood(xk, X, home_goals, away_goals, weights, alpha):.4f}")

        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(X, home_goals, away_goals, weights, alpha),
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter},
            callback=callback
        )

        self.coefficients = result.x
        return result
    
    def predict(self, X):
        """Predict lambdas for new matches."""
        if self.coefficients is None:
            raise ValueError("Model not trained yet")
        return self._predict_lambdas(X, self.coefficients)
    
    def get_params(self):
        """Return fitted parameters as a dict."""
        if self.coefficients is None:
            raise ValueError("Model not trained yet")
        
        beta_h = self.coefficients[:self.n_features + 1]
        beta_a = self.coefficients[self.n_features + 1:2 * (self.n_features + 1)]
        rho = self.coefficients[-1]
        
        return {
            'beta_home': beta_h,
            'beta_away': beta_a,
            'rho': rho
        }


def prepare_training_data(results_path, squad_features_path, weight_decay=0.001,
                          min_year=None, max_year=None,
                          start_date=None, end_date=None, drop_friendlies=False,
                          verbose=True):
    """
    Load and prepare training data for linear regression.
    
    Steps:
    1. Load match results and squad features
    2. Extract FIFA year from match date (year starts Oct 10)
    3. Filter matches where both teams have squad_size > 11 and exist in squad_features
    4. Normalize team names to match between datasets
    5. Create feature matrix combining ELO, last-10, and squad stats
    6. Compute temporal weights with exponential decay (recent matches weighted higher)
    
    Args:
        results_path: path to match_results_elo.csv
        squad_features_path: path to squad_features.csv
        weight_decay: decay rate for temporal weighting (higher = faster decay to past)
        
    Returns:
        X: feature matrix (n_matches, n_features)
        home_goals, away_goals: observed goals
        weights: temporal decay weights (n_matches,)
        feature_names: list of feature names
        matches_df: filtered match dataframe with predictions
    """
    
    # Load data
    match_results = pd.read_csv(results_path)
    squad_features = pd.read_csv(squad_features_path)
    
    # Convert date to datetime
    match_results['date'] = pd.to_datetime(match_results['date'])
    
    # Extract FIFA year. A new FIFA game ships in Oct and covers Oct(Y-1)..Sep(Y),
    # so a match in Oct(Y-1) or later belongs to game-year Y.
    # E.g. 2024-10-10 to 2025-09-30 is FC25 (year=2025).
    match_results['fifa_year'] = match_results['date'].dt.year
    match_results.loc[match_results['date'].dt.month >= 10, 'fifa_year'] += 1
    
    # Normalize team names for matching
    # squad_features has capitalized names, match_results might vary
    squad_features['team_normalized'] = squad_features['team'].str.lower().str.strip()
    match_results['home_team_normalized'] = match_results['home_team'].str.lower().str.strip()
    match_results['away_team_normalized'] = match_results['away_team'].str.lower().str.strip()
    
    # Merge with squad features for home team
    match_results = match_results.merge(
        squad_features.rename(columns={
            'squad_size': 'home_squad_size',
            'squad_avg': 'home_squad_avg',
            'squad_std': 'home_squad_std',
            'attack_avg': 'home_attack_avg',
            'midfield_avg': 'home_midfield_avg',
            'defence_avg': 'home_defence_avg',
            'bench_avg': 'home_bench_avg',
            'bench_std': 'home_bench_std',
            'year': 'squad_year_home'
        })[['team_normalized', 'squad_year_home', 'home_squad_size', 'home_squad_avg', 
            'home_squad_std', 'home_attack_avg', 'home_midfield_avg', 'home_defence_avg',
            'home_bench_avg', 'home_bench_std']],
        left_on=['home_team_normalized', 'fifa_year'],
        right_on=['team_normalized', 'squad_year_home'],
        how='left'
    )
    
    # Merge with squad features for away team
    match_results = match_results.merge(
        squad_features.rename(columns={
            'squad_size': 'away_squad_size',
            'squad_avg': 'away_squad_avg',
            'squad_std': 'away_squad_std',
            'attack_avg': 'away_attack_avg',
            'midfield_avg': 'away_midfield_avg',
            'defence_avg': 'away_defence_avg',
            'bench_avg': 'away_bench_avg',
            'bench_std': 'away_bench_std',
            'year': 'squad_year_away'
        })[['team_normalized', 'squad_year_away', 'away_squad_size', 'away_squad_avg',
            'away_squad_std', 'away_attack_avg', 'away_midfield_avg', 'away_defence_avg',
            'away_bench_avg', 'away_bench_std']],
        left_on=['away_team_normalized', 'fifa_year'],
        right_on=['team_normalized', 'squad_year_away'],
        how='left'
    )
    
    # Filter: both teams must have squad_size > 11
    match_results = match_results[
        (match_results['home_squad_size'] > 11) & 
        (match_results['away_squad_size'] > 11)
    ].copy()
    
    # Drop rows with NaN in critical features
    critical_cols = ['home_squad_avg', 'away_squad_avg', 'home_elo', 'away_elo',
                     'home_scored_last10', 'away_scored_last10']
    match_results = match_results.dropna(subset=critical_cols)

    # Restrict to a FIFA-year window (used for train vs. validation splitting)
    if min_year is not None:
        match_results = match_results[match_results['fifa_year'] >= min_year]
    if max_year is not None:
        match_results = match_results[match_results['fifa_year'] <= max_year]

    # Restrict to a date window (used for the train/val/test split by match date)
    if start_date is not None:
        match_results = match_results[match_results['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        match_results = match_results[match_results['date'] < pd.to_datetime(end_date)]

    # Optionally drop friendlies (lower signal than competitive matches)
    if drop_friendlies:
        match_results = match_results[match_results['tournament'] != 'Friendly']

    match_results = match_results.copy()

    # Compute temporal weights: exponential decay based on days from latest match
    latest_date = match_results['date'].max()
    days_since = (latest_date - match_results['date']).dt.days
    weights = np.exp(-weight_decay * days_since)
    weights = weights / weights.sum() * len(weights)  # Normalize so average weight = 1
    
    # Build feature matrix
    feature_cols = [
        'home_elo', 'away_elo',
        'home_scored_last10', 'home_conceded_last10', 'away_scored_last10', 'away_conceded_last10',
        'home_squad_avg', 'home_squad_std', 'home_attack_avg', 'home_midfield_avg', 'home_defence_avg',
        'home_bench_avg', 'home_bench_std',
        'away_squad_avg', 'away_squad_std', 'away_attack_avg', 'away_midfield_avg', 'away_defence_avg',
        'away_bench_avg', 'away_bench_std'
    ]
    
    # Fill any remaining NaNs (e.g., bench_std for small squads) with 0
    for col in feature_cols:
        match_results[col] = match_results[col].fillna(0)
    
    X = match_results[feature_cols].values
    home_goals = match_results['home_score'].values.astype(float)
    away_goals = match_results['away_score'].values.astype(float)
    
    # Check for NaN values — drop from BOTH the arrays and the returned dataframe
    nan_goals = np.isnan(home_goals) | np.isnan(away_goals)
    if nan_goals.any():
        if verbose:
            print(f"WARNING: {nan_goals.sum()} matches with NaN goals, removing them")
        X = X[~nan_goals]
        home_goals = home_goals[~nan_goals]
        away_goals = away_goals[~nan_goals]
        weights = weights[~nan_goals]
        match_results = match_results[~nan_goals].copy()

    if verbose:
        print(f"Training data prepared: {len(X)} matches, {X.shape[1]} features")
        print(f"  Home goals: mean={home_goals.mean():.2f}, std={home_goals.std():.2f}")
        print(f"  Away goals: mean={away_goals.mean():.2f}, std={away_goals.std():.2f}")
        print(f"  Temporal weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        print(f"  Date range: {match_results['date'].min().date()} to {latest_date.date()}")

    return X, home_goals, away_goals, weights, feature_cols, match_results


def train_linear_regression(X, home_goals, away_goals, weights, feature_names):
    """
    Train the linear-regression Dixon-Coles model on a prepared feature matrix.

    Args:
        X: feature matrix (n_matches, n_features)
        home_goals, away_goals: observed goals
        weights: temporal decay weights
        feature_names: list of feature column names
    """

    print("=" * 70)
    print("LINEAR REGRESSION TRAINING FOR DIXON-COLES MODEL")
    print("=" * 70)

    # Initialize and fit model
    model = LinearRegressionDixonColes(n_features=X.shape[1])
    model.feature_names = feature_names

    print(f"\nTraining with {X.shape[1]} features...")
    print("Optimizing likelihood with temporal decay weighting...")

    result = model.fit(X, home_goals, away_goals, weights=weights, max_iter=500)

    print("\n" + "=" * 70)
    print(f"Optimization result: {result.message}")
    print(f"Final NLL: {result.fun:.4f}")
    print("=" * 70)

    # Extract and display parameters
    params = model.get_params()

    print("\nHome Team Lambda Coefficients (beta_home):")
    for i, (name, coef) in enumerate(zip(['intercept'] + feature_names, params['beta_home'])):
        print(f"  {name:30s}: {coef:8.4f}")

    print("\nAway Team Lambda Coefficients (beta_away):")
    for i, (name, coef) in enumerate(zip(['intercept'] + feature_names, params['beta_away'])):
        print(f"  {name:30s}: {coef:8.4f}")

    print(f"\nRho (correlation parameter): {params['rho']:.4f}")

    return model


# ============================================================================
# MLP (Neural Network Model with Dixon-Coles Loss)
# ============================================================================

class MLPDixonColes(nn.Module):
    """
    Neural network that predicts lambda_home, lambda_away, and rho.
    
    Output:
    - lambda_home, lambda_away: expected goals (must be > 0)
    - rho: correlation parameter for Dixon-Coles model
    """
    def __init__(self, input_size, hidden_size):
        super(MLPDixonColes, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 3)  # Output: [lambda_home, lambda_away, rho]
        self.softplus = nn.Softplus()  # For lambda: ensures positive values
        self.tanh = nn.Tanh()  # For rho: constrains to [-1, 1]

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(self.relu(out))
        out = self.fc2(out)
        out = self.dropout(self.relu2(out))
        out = self.fc3(out)
        
        # Separate output
        lambda_home = self.softplus(out[:, 0]) + 0.1  # Add 0.1 for numerical stability
        lambda_away = self.softplus(out[:, 1]) + 0.1
        rho = self.tanh(out[:, 2])  # Constrained to [-1, 1]
        
        return lambda_home, lambda_away, rho


def dixon_coles_loss(lambda_home, lambda_away, rho, home_goals, away_goals, weights=None):
    """
    Compute negative log-likelihood using Poisson distribution with Dixon-Coles correction.

    Args:
        lambda_home, lambda_away: predicted expected goals (batches)
        rho: correlation parameter
        home_goals, away_goals: observed goals (tensors)
        weights: optional per-match temporal weights (tensor); if given, the loss is a
                 weighted mean so recent matches count more (matches the linear model)

    Returns:
        loss: scalar tensor (negative log-likelihood)
    """
    lambda_home = torch.clamp(lambda_home, min=0.01, max=100)
    lambda_away = torch.clamp(lambda_away, min=0.01, max=100)
    rho = torch.clamp(rho, min=-0.99, max=0.99)
    
    # Poisson log-likelihood: log(e^(-lambda) * lambda^k / k!)
    # Using log-space: -lambda + k*log(lambda) - log(k!)
    log_pmf_h = home_goals * torch.log(lambda_home) - lambda_home
    log_pmf_a = away_goals * torch.log(lambda_away) - lambda_away
    
    # Clamp to avoid -inf
    log_pmf_h = torch.clamp(log_pmf_h, min=-50)
    log_pmf_a = torch.clamp(log_pmf_a, min=-50)
    
    # Dixon-Coles low-score correction (tau factors)
    # Start with tau = 1 for all matches
    tau = torch.ones_like(rho)
    
    # For (0,0): tau = 1 - lambda_h * lambda_a * rho
    mask_0_0 = (home_goals == 0) & (away_goals == 0)
    tau[mask_0_0] = torch.clamp(1 - lambda_home[mask_0_0] * lambda_away[mask_0_0] * rho[mask_0_0], min=0.01, max=10)
    
    # For (0,1): tau = 1 + lambda_h * rho
    mask_0_1 = (home_goals == 0) & (away_goals == 1)
    tau[mask_0_1] = torch.clamp(1 + lambda_home[mask_0_1] * rho[mask_0_1], min=0.01, max=10)
    
    # For (1,0): tau = 1 + lambda_a * rho
    mask_1_0 = (home_goals == 1) & (away_goals == 0)
    tau[mask_1_0] = torch.clamp(1 + lambda_away[mask_1_0] * rho[mask_1_0], min=0.01, max=10)
    
    # For (1,1): tau = 1 - rho
    mask_1_1 = (home_goals == 1) & (away_goals == 1)
    tau[mask_1_1] = torch.clamp(1 - rho[mask_1_1], min=0.01, max=10)
    
    # Combined log-likelihood with Dixon-Coles correction
    log_likelihood = log_pmf_h + log_pmf_a + torch.log(tau)
    
    # Clamp to avoid extreme values
    log_likelihood = torch.clamp(log_likelihood, min=-50, max=10)

    # Negative (optionally weighted) average log-likelihood
    if weights is not None:
        loss = -(weights * log_likelihood).sum() / weights.sum()
    else:
        loss = -torch.mean(log_likelihood)

    return loss


def train_mlp_dixon_coles(X_train, home_goals, away_goals, weights=None, hidden_size=64, num_epochs=500, learning_rate=0.001, batch_size=256):
    """
    Train MLP model to predict Dixon-Coles parameters.

    Args:
        X_train: feature matrix (n_matches, n_features)
        home_goals, away_goals: observed goals
        weights: optional per-match temporal weights (recent matches weighted higher,
                 same weights the linear model uses)
        hidden_size: hidden layer size
        num_epochs: number of training epochs
        learning_rate: optimizer learning rate
        batch_size: mini-batch size for stochastic gradient descent

    Returns:
        model: trained MLP model
        loss_history: list of (epoch-average) loss values
    """
    # Standardize features (z-score) — raw ELO ~2000 would otherwise explode the loss to NaN
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8  # Avoid division by zero
    X_train = (X_train - mean) / std
    X_tensor = torch.FloatTensor(X_train)
    home_goals_tensor = torch.FloatTensor(home_goals)
    away_goals_tensor = torch.FloatTensor(away_goals)

    # Temporal weights travel with each match through the loader (default = uniform)
    if weights is None:
        weights = np.ones(len(X_train))
    weights_tensor = torch.FloatTensor(np.asarray(weights))

    # Mini-batch loader: shuffle so each epoch sees batches in a different order
    dataset = TensorDataset(X_tensor, home_goals_tensor, away_goals_tensor, weights_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_size = X_train.shape[1]
    model = MLPDixonColes(input_size, hidden_size)
    # Stash the scaler on the model so predictions can reuse the same transform
    model.feature_mean = mean
    model.feature_std = std

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    print(f"\nTraining MLP for {num_epochs} epochs...")
    print(f"  Input size: {input_size}, Hidden size: {hidden_size}, Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(loader)}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for xb, hg, ag, wb in loader:
            # Forward pass
            lambda_h, lambda_a, rho = model(xb)

            # Compute Dixon-Coles loss, weighting recent matches higher
            loss = dixon_coles_loss(lambda_h, lambda_a, rho, hg, ag, weights=wb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate a sample-weighted average (last batch may be smaller)
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model, loss_history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from evaluate import evaluate_models

    base_dir = Path(__file__).resolve().parent
    results_path = base_dir / 'results' / 'match_results_elo.csv'
    squad_features_path = base_dir / 'results' / 'squad_features.csv'

    # Temporal split: train on FIFA years <= 2023, validate on 2024+ (FC24/25/26).
    SPLIT_YEAR = 2024

    # Temporal decay: 0.0025 => ~0.76 yr half-life (2.5x more aggressive than before),
    # so recent form dominates both models' fits.
    WEIGHT_DECAY = 0.00013

    print("Preparing TRAIN data (fifa_year <= 2023)...")
    X_tr, hg_tr, ag_tr, w_tr, feature_names, _ = prepare_training_data(
        results_path, squad_features_path, weight_decay=WEIGHT_DECAY, max_year=SPLIT_YEAR - 1
    )
    print("\nPreparing VALIDATION data (fifa_year >= 2024)...")
    X_val, hg_val, ag_val, w_val, _, _ = prepare_training_data(
        results_path, squad_features_path, min_year=SPLIT_YEAR
    )

    # --- Train linear regression on the training split ---
    linear_model = train_linear_regression(X_tr, hg_tr, ag_tr, w_tr, feature_names)

    # --- Train MLP on the training split ---
    print("\n" + "=" * 70)
    print("NEURAL NETWORK (MLP) TRAINING FOR DIXON-COLES MODEL")
    print("=" * 70)

    mlp_model, loss_history = train_mlp_dixon_coles(
        X_tr, hg_tr, ag_tr, w_tr,
        hidden_size=128,
        num_epochs=1000,
        learning_rate=0.001,
        batch_size=256
    )

    # --- Evaluate both models (+ baseline) on the held-out validation years ---
    evaluate_models(linear_model, mlp_model, X_val, hg_val, ag_val, hg_tr, ag_tr)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Train: {len(X_tr)} matches (<= {SPLIT_YEAR - 1})   "
          f"Validation: {len(X_val)} matches (>= {SPLIT_YEAR})")
    print(f"MLP final training loss: {loss_history[-1]:.4f}")

