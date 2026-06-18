"""
Inspect why the MLP rates Morocco ~ Haiti in Group C.
Prints per-team feature vectors and each model's neutral lambdas for every
Group C pairing, plus the MLP's input standardization stats.
"""
import numpy as np
np.set_printoptions(suppress=True, precision=2, linewidth=160)

from wc_simulation import (
    load_bracket, build_team_features, make_feat_builder, neutral_lambdas,
    TRAIN_END, VAL_START, BEST_ALPHA, BEST_MLP, ELO_CSV, SQUAD_CSV, SQUAD_COLS,
)
from training import prepare_training_data, LinearRegressionDixonColes
from experiments import FULL_FEATURES, build_X, train_mlp, predict_mlp

GROUP = 'C'

# ---- features ----
groups, group_matches, ko = load_bracket()
team_names = sorted({t for ts in groups.values() for t in ts})
team_idx = {t: i for i, t in enumerate(team_names)}
F = build_team_features(team_names)
feat = make_feat_builder(F)
gteams = groups[GROUP]

per_team_cols = ['elo', 's10', 'c10'] + SQUAD_COLS + ['gk']
print(f"\n=== Per-team feature vectors, Group {GROUP} ===")
print(f"{'team':12}" + "".join(f"{c:>11}" for c in per_team_cols))
for t in gteams:
    i = team_idx[t]
    print(f"{t:12}" + "".join(f"{F[c][i]:11.2f}" for c in per_team_cols))

# ---- train models ----
print("\nTraining models...", flush=True)
_, hgtr, agtr, wtr, _, dftr = prepare_training_data(ELO_CSV, SQUAD_CSV, end_date=TRAIN_END, verbose=False)
_, hgval, agval, _, _, dfval = prepare_training_data(ELO_CSV, SQUAD_CSV, start_date=VAL_START, end_date=TRAIN_END, verbose=False)
Xtr = build_X(dftr.reset_index(drop=True), FULL_FEATURES)
Xval = build_X(dfval.reset_index(drop=True), FULL_FEATURES)
lin = LinearRegressionDixonColes(Xtr.shape[1])
lin.fit(Xtr, hgtr, agtr, weights=wtr, alpha=BEST_ALPHA, verbose=False, max_iter=500)
mlp, _ = train_mlp(Xtr, hgtr, agtr, wtr, Xval, hgval, agval, max_epochs=1500, patience=150, **BEST_MLP)

lin_lam = lambda X: lin.predict(X)
def mlp_lam(X):
    lh, la, _ = predict_mlp(mlp, X)
    return lh, la

# ---- MLP input standardization: how much does each feature survive scaling? ----
print("\n=== MLP feature standardization (mean / std it divides by) ===")
print(f"{'feature':22}{'train_mean':>12}{'train_std':>12}")
for name, m, s in zip(FULL_FEATURES, mlp.feature_mean, mlp.feature_std):
    print(f"{name:22}{m:12.2f}{s:12.2f}")

# z-scores of each Group C team's elo/squad under the MLP scaler (home-slot positions)
print("\n=== Group C teams as MLP sees them (z-scored key inputs) ===")
key = ['home_elo', 'home_squad_avg', 'home_attack_avg', 'home_defence_avg']
idx = [FULL_FEATURES.index(k) for k in key]
print(f"{'team':12}" + "".join(f"{k.replace('home_',''):>14}" for k in key))
for t in gteams:
    i = team_idx[t]
    row = feat([i], [i])[0]          # home-slot features for this team
    z = (row - mlp.feature_mean) / mlp.feature_std
    print(f"{t:12}" + "".join(f"{z[j]:14.2f}" for j in idx))

# ---- neutral lambdas for every Group C pairing ----
print("\n=== Neutral expected goals per pairing (linear | MLP) ===")
print(f"{'matchup':28}{'lin xG':>16}{'mlp xG':>16}")
for a in gteams:
    for b in gteams:
        if a >= b:
            continue
        ia, ib = team_idx[a], team_idx[b]
        lA, lB = neutral_lambdas(lin_lam, feat, [ia], [ib])
        mA, mB = neutral_lambdas(mlp_lam, feat, [ia], [ib])
        print(f"{a[:12]+' v '+b[:12]:28}{f'{lA[0]:.2f}-{lB[0]:.2f}':>16}{f'{mA[0]:.2f}-{mB[0]:.2f}':>16}")

# ---- MLP output range overall: is it just compressed toward the mean? ----
lh, la, rho = predict_mlp(mlp, Xtr)
llh, lla = lin.predict(Xtr)
print("\n=== Predicted lambda spread across ALL training matches ===")
print(f"{'':10}{'min':>8}{'mean':>8}{'max':>8}{'std':>8}")
print(f"{'MLP':10}{lh.min():8.2f}{lh.mean():8.2f}{lh.max():8.2f}{lh.std():8.2f}")
print(f"{'linear':10}{llh.min():8.2f}{llh.mean():8.2f}{llh.max():8.2f}{llh.std():8.2f}")
print(f"MLP rho: mean={rho.mean():.3f} min={rho.min():.3f} max={rho.max():.3f}")
