"""Generate report figures. Currently: the joint scoreline (two-Poisson) surface
for a hypothetical Spain vs France match, using the fitted linear Dixon-Coles model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import poisson

from core.training import prepare_training_data, LinearRegressionDixonColes
from core.experiments import FULL_FEATURES, build_X
import wc_simulation as W

base = Path(__file__).resolve().parent.parent


def main():
    # Train the best linear model on all pre-WC data
    _, hg, ag, wt, _, dftr = prepare_training_data(
        base / 'results' / 'match_results_elo.csv',
        base / 'results' / 'squad_features.csv', end_date='2026-06-01', verbose=False)
    X = build_X(dftr.reset_index(drop=True), FULL_FEATURES)
    m = LinearRegressionDixonColes(X.shape[1])
    m.fit(X, hg, ag, weights=wt, alpha=10.0, verbose=False, max_iter=500)

    # Spain vs France feature vector (Spain home, France away)
    groups, gm, ko = W.load_bracket()
    names = sorted({t for ts in groups.values() for t in ts})
    F = W.build_team_features(names)
    feat = W.make_feat_builder(F)
    idx = {t: i for i, t in enumerate(names)}
    Xrow = feat([idx['Spain']], [idx['France']])
    lh, la = m.predict(Xrow)
    lh, la = float(lh[0]), float(la[0])
    print(f"lambda Spain(home)={lh:.3f}  France(away)={la:.3f}")

    # Joint scoreline PMF (two independent Poissons)
    G = 7
    g = np.arange(G)
    P = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))  # P[i,j]=P(Spain=i,France=j)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    xp, yp = np.meshgrid(g, g, indexing='ij')
    dz = P.ravel()
    colors = plt.cm.viridis(dz / dz.max())
    ax.bar3d(xp.ravel() - 0.4, yp.ravel() - 0.4, np.zeros_like(dz),
             0.8, 0.8, dz, color=colors, shade=True)
    ax.set_xlabel('Spain goals'); ax.set_ylabel('France goals'); ax.set_zlabel('P(scoreline)')
    ax.set_xticks(g); ax.set_yticks(g)
    ax.set_title(f'Joint scoreline distribution: Spain vs France\n'
                 rf'$\lambda_{{Spain}}={lh:.2f}$, $\lambda_{{France}}={la:.2f}$')
    ax.view_init(elev=28, azim=-58)
    plt.tight_layout()
    out = base / 'figures' / 'image.png'
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
