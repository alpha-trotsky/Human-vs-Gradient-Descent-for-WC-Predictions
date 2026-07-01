"""
Scoreline calibration curves for the McHale–Copula model.

For every (match, scoreline) pair in the test set, collects the model's
predicted probability for that scoreline and whether it was actually observed.
Bins by predicted probability in 5% buckets and plots mean observed frequency
vs. predicted probability — a well-calibrated model lies on the 45° diagonal.

Three panels on one figure:
  A) All scorelines          (g1, g2)
  B) High-total              g1 + g2 > 6
  C) High-asymmetry          max(g1, g2) >= 3 and min(g1, g2) <= 1

Open circles / dashed CI = thin bucket (n < THIN_N).

Usage:
    python -m analysis.calibration_copula
    python -m analysis.calibration_copula --alpha 10 --k_mode log
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from core.experiments import load_split, build_X, FULL_FEATURES, TRAIN_END, VAL_END
from core.McHale_Copula import McHaleCopulaModel, score_matrix
from wc_simulation import BEST_COPULA_ALPHA, BEST_COPULA_K_MODE

MAX_GOALS = 10
THIN_N    = 20
Z95       = 1.96
BIN_EDGES = np.arange(0, 1.0001, 0.05)   # 20 bins of width 5%


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_pairs(model, X_te, hg_te, ag_te):
    """
    Return a DataFrame with one row per (match × scoreline cell).
    Columns: pred_prob, realized, g1, g2.
    Scorelines where actual score > MAX_GOALS are not marked realized in any cell.
    """
    mu_h, mu_a         = model.predict(X_te)
    r_h, r_a, k_params = model._unpack_extras(model.coefficients)
    k_arr              = model._compute_k(X_te, k_params)

    g              = np.arange(MAX_GOALS + 1)
    g1g, g2g       = np.meshgrid(g, g, indexing='ij')
    g1f, g2f       = g1g.ravel(), g2g.ravel()   # (n_grid,) flat index order
    n_grid, n_match = len(g1f), len(X_te)

    pred = np.empty(n_match * n_grid)
    real = np.zeros(n_match * n_grid)

    for i in range(n_match):
        k_i = float(k_arr) if np.ndim(k_arr) == 0 else float(k_arr[i])
        M   = score_matrix(float(mu_h[i]), float(mu_a[i]), r_h, r_a, k_i, MAX_GOALS)
        h_obs, a_obs = int(hg_te[i]), int(ag_te[i])
        pred[i * n_grid : (i + 1) * n_grid] = M.ravel()
        if h_obs <= MAX_GOALS and a_obs <= MAX_GOALS:
            # flat index within the (MAX_GOALS+1)×(MAX_GOALS+1) row-major grid
            real[i * n_grid + h_obs * (MAX_GOALS + 1) + a_obs] = 1.0

    return pd.DataFrame({
        'pred_prob': pred,
        'realized':  real,
        'g1':        np.tile(g1f, n_match).astype(int),
        'g2':        np.tile(g2f, n_match).astype(int),
    })


# ---------------------------------------------------------------------------
# Calibration binning (Wilson score CI)
# ---------------------------------------------------------------------------
def calibrate(pred, real):
    """Bin pred into 5% buckets; return DataFrame with obs frequency and Wilson CI."""
    mids = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    rows = []
    for lo, hi, mid in zip(BIN_EDGES[:-1], BIN_EDGES[1:], mids):
        mask = (pred >= lo) & (pred < hi)
        n    = int(mask.sum())
        if n == 0:
            rows.append({'mid': mid, 'freq': np.nan,
                         'ci_lo': np.nan, 'ci_hi': np.nan, 'n': 0})
            continue
        p     = real[mask].mean()
        denom = 1 + Z95**2 / n
        p_c   = (p + Z95**2 / (2 * n)) / denom
        marg  = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / denom
        rows.append({'mid': mid, 'freq': p,
                     'ci_lo': max(0.0, p_c - marg),
                     'ci_hi': min(1.0, p_c + marg), 'n': n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------
def draw_panel(ax, cal, title):
    valid = cal[cal['n'] > 0].copy()

    ax.axline((0, 0), slope=1, color='k', lw=1, ls='--', alpha=0.5)

    if valid.empty:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='grey')
        ax.set_title(title)
        return

    for _, row in valid.iterrows():
        x    = row['mid']  * 100
        y    = row['freq'] * 100
        lo   = row['ci_lo'] * 100
        hi   = row['ci_hi'] * 100
        thin = row['n'] < THIN_N
        col  = 'steelblue'
        ax.plot(x, y, 'o', color=col, mfc=('white' if thin else col), ms=6, zorder=3)
        ax.plot([x, x], [lo, hi], ('--' if thin else '-'), color=col, lw=1.5)

    handles = [
        Line2D([0], [0], color='k',         ls='--', label='Perfect calibration'),
        Line2D([0], [0], color='steelblue', ls='-',  marker='o',
               mfc='steelblue', label=f'n ≥ {THIN_N}'),
        Line2D([0], [0], color='steelblue', ls='--', marker='o',
               mfc='white',     label=f'n < {THIN_N}  (thin)'),
    ]
    ax.legend(handles=handles, fontsize=7)
    ax.set_xlabel('Predicted probability (%)')
    ax.set_ylabel('Observed frequency (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, lw=0.5)

    # console table
    print(f"\n  {title}")
    print(f"  {'bucket':>10}  {'n':>8}  {'obs%':>9}  95% CI")
    print("  " + "-" * 56)
    for _, row in valid.iterrows():
        flag = '*' if row['n'] < THIN_N else ' '
        print(f"  {row['mid']*100:>9.1f}%  {row['n']:>7}{flag}  "
              f"{row['freq']*100:>8.2f}%  "
              f"[{row['ci_lo']*100:.2f}%, {row['ci_hi']*100:.2f}%]")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',  type=float, default=BEST_COPULA_ALPHA,
                        help='Ridge alpha for the copula model')
    parser.add_argument('--k_mode', type=str,   default=BEST_COPULA_K_MODE,
                        help='k_mode: const | log | exp | quad')
    args = parser.parse_args()

    # ---- data ----
    print("Loading splits...")
    df_tr, hg_tr, ag_tr, w_tr = load_split(end=TRAIN_END)
    df_te, hg_te, ag_te, _    = load_split(start=VAL_END)
    X_tr = build_X(df_tr, FULL_FEATURES)
    X_te = build_X(df_te, FULL_FEATURES)
    print(f"  train={len(X_tr)}  test={len(X_te)}")

    # ---- model ----
    print(f"\nTraining McHale–Copula  (alpha={args.alpha}, k_mode={args.k_mode})...")
    model = McHaleCopulaModel(X_tr.shape[1], k_mode=args.k_mode)
    model.fit(X_tr, hg_tr, ag_tr, weights=w_tr,
              alpha=args.alpha, max_iter=1000, verbose=True)

    # ---- collect pairs ----
    print(f"\nCollecting scoreline pairs for {len(X_te)} test matches "
          f"({MAX_GOALS+1}×{MAX_GOALS+1} grid each)...")
    pairs = collect_pairs(model, X_te, hg_te, ag_te)
    n_real = int(pairs['realized'].sum())
    print(f"  total pairs: {len(pairs):,}   realized: {n_real}  "
          f"(expected ≈ {len(X_te)}, missing = out-of-grid actuals)")

    # ---- subsets (filter on predicted scoreline g1, g2) ----
    total = pairs['g1'] + pairs['g2']
    mx    = pairs[['g1', 'g2']].max(axis=1)
    mn    = pairs[['g1', 'g2']].min(axis=1)

    df_all  = pairs
    df_high = pairs[total > 6]
    df_asym = pairs[(mx >= 3) & (mn <= 1)]

    print(f"\n  All scorelines  : {len(df_all):,} pairs")
    print(f"  High-total      : {len(df_high):,} pairs  (g1+g2 > 6)")
    print(f"  High-asymmetry  : {len(df_asym):,} pairs  (max≥3, min≤1)")

    # ---- calibrate ----
    cal_all  = calibrate(df_all['pred_prob'].values,  df_all['realized'].values)
    cal_high = calibrate(df_high['pred_prob'].values, df_high['realized'].values)
    cal_asym = calibrate(df_asym['pred_prob'].values, df_asym['realized'].values)

    # shared axis limit: a little beyond the widest non-empty bin across all panels
    def _xmax(cal):
        v = cal[cal['n'] > 0]
        return float(v['mid'].max()) * 100 + 3 if not v.empty else 15.0
    lim = max(_xmax(cal_all), _xmax(cal_high), _xmax(cal_asym))

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    print("\n" + "=" * 60)
    draw_panel(axes[0], cal_all,  'All scorelines')
    draw_panel(axes[1], cal_high, 'High-total  (g1+g2 > 6)')
    draw_panel(axes[2], cal_asym, 'High-asymmetry  (max≥3, min≤1)')

    for ax in axes:
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    plt.suptitle(
        f'McHale–Copula scoreline calibration  '
        f'(α={args.alpha}, k_mode={args.k_mode})\n'
        f'Test set: {len(X_te)} matches  |  '
        f'open circle / dashed CI = n < {THIN_N}',
        fontsize=11,
    )
    plt.tight_layout()

    out = _root / 'figures' / 'calibration_copula.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")
