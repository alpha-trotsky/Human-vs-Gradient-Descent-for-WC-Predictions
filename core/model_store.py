"""
Save / load the trained linear + MLP Dixon-Coles models so callers (e.g.
canada_query.py) don't have to retrain on every run.

The "blend" (ensemble) is not a separate model: it's the mean of the linear and
MLP scoreline grids, computed at query time. So persisting the two base models
is all that's needed to reproduce linear / MLP / ensemble predictions exactly.

Artifacts (written to ./models):
    linear_dc.pkl   linear coefficients + feature scaler (pure numpy, pickled)
    mlp_dc.pt       MLP state_dict + architecture config + feature scaler

Public API:
    train_models()              -> (lin, mlp)   train both from the CSVs
    save_models(lin, mlp)       -> Path         persist both to ./models
    load_models()              -> (lin, mlp)   reconstruct both from ./models
    load_or_train(retrain=False)-> (lin, mlp)   load if present else train+save
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import torch

# Ensure the project root is in sys.path so wc_simulation can be imported
# whether this file is run directly or imported as core.model_store.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wc_simulation import (
    TRAIN_END, VAL_START, BEST_ALPHA, BEST_MLP, ELO_CSV, SQUAD_CSV,
)
from .training import prepare_training_data, LinearRegressionDixonColes
from .experiments import FULL_FEATURES, build_X, train_mlp, FlexibleMLP

MODEL_DIR = _root / 'models'
LINEAR_PATH = MODEL_DIR / 'linear_dc.pkl'
MLP_PATH = MODEL_DIR / 'mlp_dc.pt'


# ---------------------------------------------------------------------------
# Train (same pipeline + hyperparameters as wc_simulation / canada_query)
# ---------------------------------------------------------------------------
def train_models(verbose=True):
    """Train the linear + MLP Dixon-Coles models from the project CSVs."""
    if verbose:
        print("Training models (linear + MLP)...", flush=True)
    _, hgtr, agtr, wtr, _, dftr = prepare_training_data(
        ELO_CSV, SQUAD_CSV, end_date=TRAIN_END, verbose=False)
    _, hgval, agval, _, _, dfval = prepare_training_data(
        ELO_CSV, SQUAD_CSV, start_date=VAL_START, end_date=TRAIN_END, verbose=False)
    Xtr = build_X(dftr.reset_index(drop=True), FULL_FEATURES)
    Xval = build_X(dfval.reset_index(drop=True), FULL_FEATURES)

    lin = LinearRegressionDixonColes(Xtr.shape[1])
    lin.fit(Xtr, hgtr, agtr, weights=wtr, alpha=BEST_ALPHA, verbose=False, max_iter=500)

    mlp, _ = train_mlp(Xtr, hgtr, agtr, wtr, Xval, hgval, agval,
                       max_epochs=1500, patience=150, **BEST_MLP)
    return lin, mlp


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_models(lin, mlp, model_dir=MODEL_DIR):
    """Persist both models to model_dir (created if needed)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    # Linear model: pure numpy. Store coefficients + the train-set scaler so
    # predictions reproduce exactly without refitting.
    with open(model_dir / LINEAR_PATH.name, 'wb') as f:
        pickle.dump({
            'n_features': lin.n_features,
            'coefficients': lin.coefficients,
            'feature_mean': lin.feature_mean,
            'feature_std': lin.feature_std,
            'feature_names': lin.feature_names,
        }, f)

    # MLP: weights + the architecture needed to rebuild FlexibleMLP, + scaler.
    torch.save({
        'state_dict': mlp.state_dict(),
        'config': {
            'input_size': mlp.body[0].in_features,
            'width': BEST_MLP['width'],
            'depth': BEST_MLP['depth'],
            'dropout': BEST_MLP['dropout'],
        },
        'feature_mean': np.asarray(mlp.feature_mean),
        'feature_std': np.asarray(mlp.feature_std),
    }, model_dir / MLP_PATH.name)

    return model_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_models(model_dir=MODEL_DIR):
    """Reconstruct both models from model_dir. Raises if artifacts are missing."""
    model_dir = Path(model_dir)
    lin_path = model_dir / LINEAR_PATH.name
    mlp_path = model_dir / MLP_PATH.name
    if not lin_path.exists() or not mlp_path.exists():
        raise FileNotFoundError(
            f"Saved models not found in {model_dir}. Run: python core/model_store.py")

    with open(lin_path, 'rb') as f:
        d = pickle.load(f)
    lin = LinearRegressionDixonColes(d['n_features'])
    lin.coefficients = d['coefficients']
    lin.feature_mean = d['feature_mean']
    lin.feature_std = d['feature_std']
    lin.feature_names = d['feature_names']

    blob = torch.load(mlp_path, weights_only=False)
    cfg = blob['config']
    mlp = FlexibleMLP(cfg['input_size'], cfg['width'], cfg['depth'], cfg['dropout'])
    mlp.load_state_dict(blob['state_dict'])
    mlp.feature_mean = blob['feature_mean']
    mlp.feature_std = blob['feature_std']
    mlp.eval()

    return lin, mlp


def load_or_train(retrain=False, save=True, verbose=True):
    """Load saved models if present (and retrain not forced), else train + save."""
    if not retrain and LINEAR_PATH.exists() and MLP_PATH.exists():
        if verbose:
            print(f"Loading saved models from {MODEL_DIR}...", flush=True)
        return load_models()
    lin, mlp = train_models(verbose=verbose)
    if save:
        save_models(lin, mlp)
        if verbose:
            print(f"Saved models to {MODEL_DIR}", flush=True)
    return lin, mlp


if __name__ == "__main__":
    lin, mlp = train_models()
    out = save_models(lin, mlp)
    print(f"Saved linear + MLP models to {out}")
