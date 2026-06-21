"""
ELO sensitivity experiment.

Regenerate match features with different ELO settings and re-run the models that
won on validation, to see how the ELO logistic scale + K-factor affect accuracy.

Variants (all with K=50 fixed for every match):
    divisor = 200   (steep: rating gaps -> very extreme win probabilities)
    divisor = 1600  (flat: rating gaps -> mild win probabilities)
Baseline for reference: divisor = 800, tournament-based K (the existing pipeline).

Best models carried over from the main sweep (selected on validation):
    Linear : ridge alpha = 100, full feature set
    MLP    : lr=0.003, depth=3, width=64, dropout=0.0, grad_clip=5.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import pandas as pd

from pipeline.data_pipeline import build_match_features
from core.training import prepare_training_data
from core.experiments import (FULL_FEATURES, build_X, run_linear, run_mlp,
                               WEIGHT_DECAY, TRAIN_END, VAL_END)

base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / 'results' / 'experiment_results.csv'

BEST_ALPHA = 10.0
BEST_MLP = dict(lr=0.0005, depth=3, width=256, dropout=0.0, grad_clip=5.0)


def load_split_from(results_path, start=None, end=None):
    _, hg, ag, w, _, df = prepare_training_data(
        results_path, base_dir / 'results' / 'squad_features.csv',
        weight_decay=WEIGHT_DECAY, start_date=start, end_date=end, verbose=False)
    return df.reset_index(drop=True), hg, ag, w


def evaluate_variant(label, results_path):
    """Train best linear + best MLP on a given match_results file; return val rows."""
    dftr, hgtr, agtr, wtr = load_split_from(results_path, end=TRAIN_END)
    dfval, hgval, agval, _ = load_split_from(results_path, start=TRAIN_END, end=VAL_END)
    Xtr, Xval = build_X(dftr, FULL_FEATURES), build_X(dfval, FULL_FEATURES)

    _, lin_m = run_linear(Xtr, hgtr, agtr, wtr, Xval, hgval, agval, alpha=BEST_ALPHA)
    _, mlp_m, _ = run_mlp(Xtr, hgtr, agtr, wtr, Xval, hgval, agval,
                          max_epochs=1500, patience=150, **BEST_MLP)
    return [
        {'exp': '7_elo', 'model': f'Linear (alpha={BEST_ALPHA:g})', 'config': label, 'split': 'val', **lin_m},
        {'exp': '7_elo', 'model': 'MLP (best cfg)', 'config': label, 'split': 'val', **mlp_m},
    ]


def main():
    t0 = time.time()
    rows = []

    variants = [
        ('elo div=800 k=tourn (baseline)', dict(divisor=800, fixed_k=None),
         base_dir / 'results' / 'match_results_elo.csv'),
        ('elo div=200 k=50', dict(divisor=200, fixed_k=50),
         base_dir / 'results' / 'match_results_elo_div200_k50.csv'),
        ('elo div=1600 k=50', dict(divisor=1600, fixed_k=50),
         base_dir / 'results' / 'match_results_elo_div1600_k50.csv'),
    ]

    for label, params, out_path in variants:
        if not out_path.exists():
            print(f"[{time.time()-t0:6.1f}s] Generating {out_path.name} ({label})...", flush=True)
            df = build_match_features(**params)
            df.to_csv(out_path, index=False)
        else:
            print(f"[{time.time()-t0:6.1f}s] Using existing {out_path.name}", flush=True)
        rows.extend(evaluate_variant(label, out_path))
        print(f"[{time.time()-t0:6.1f}s] Done {label}: "
              f"lin RPS={rows[-2]['RPS']:.4f}, mlp RPS={rows[-1]['RPS']:.4f}", flush=True)

    # Append to the existing results CSV (drop any prior 7_elo rows first)
    existing = pd.read_csv(csv_path)
    existing = existing[existing['exp'] != '7_elo']
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"[{time.time()-t0:6.1f}s] Appended {len(rows)} ELO rows to {csv_path.name}", flush=True)


if __name__ == "__main__":
    main()
