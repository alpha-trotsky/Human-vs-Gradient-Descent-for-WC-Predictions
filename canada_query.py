"""
Match query tool: predictions for ANY single fixture, using the project's own
linear + MLP Dixon-Coles pipeline (same feature build, training and neutral-lambda
averaging as the headline WC odds in wc_simulation.py).

Usage:
    python canada_query.py "Senegal" "France"
    python canada_query.py "Brazil" "Morocco" --no-plot

Outputs, for linear / MLP / ensemble:
  * expected goals (xG) for each side
  * match result  W / D / L
  * winning-margin markets   (TeamA by 2+, by 3+ ; TeamB by 2+, by 3+)
  * total-goals over/under   (1.5, 2.5, 3.5)  and both-teams-to-score
  * P(team tops its group)   (Monte Carlo, same standings rules as headline sim)
and saves a 3D joint-scoreline surface to figures/<TeamA>_vs_<TeamB>.png .
"""
import sys
import numpy as np

from wc_simulation import (
    load_bracket, build_team_features, make_feat_builder, neutral_lambdas,
    TRAIN_END, VAL_START, BEST_ALPHA, BEST_MLP, ELO_CSV, SQUAD_CSV,
)
from core.training import prepare_training_data, LinearRegressionDixonColes
from core.experiments import FULL_FEATURES, build_X, train_mlp, predict_mlp
from core.McHale_Copula import score_matrix as copula_score_matrix

# ---------------------------------------------------------------------------
# Args:  python canada_query.py "Team A" "Team B" [--no-plot]
# ---------------------------------------------------------------------------
N = 10000
SEED = 0
MAX_GOALS = 10                       # scoreline grid for the Dixon-Coles matrix
PLOT_GOALS = 7                       # smaller grid for a readable 3D plot
args = [a for a in sys.argv[1:] if a not in ('--no-plot', '--retrain')]
MAKE_PLOT = '--no-plot' not in sys.argv
TEAM_A, TEAM_B = (args[0], args[1]) if len(args) >= 2 else ('Ghana', 'Croatia')


# ---------------------------------------------------------------------------
# Full Dixon-Coles joint-scoreline matrix  M[i,j] = P(A scores i, B scores j)
# (same low-score correction as dixon_coles.py, but returns the whole grid so
#  we can read off margins / totals as well as W/D/L)
# ---------------------------------------------------------------------------
from scipy.stats import poisson

def dc_matrix(lam_a, lam_b, rho, max_goals=MAX_GOALS):
    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, lam_a), poisson.pmf(g, lam_b))
    M[0, 0] *= 1 - lam_a * lam_b * rho
    M[0, 1] *= 1 + lam_a * rho
    M[1, 0] *= 1 + lam_b * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    M /= M.sum()
    return M


def markets_from_matrix(M):
    """Derive every market from the joint-scoreline matrix M[a, b]."""
    n = M.shape[0]
    i = np.arange(n)[:, None]        # team A goals (rows)
    j = np.arange(n)[None, :]        # team B goals (cols)
    diff = i - j                     # A margin (>0 means A winning)
    total = i + j
    m = {
        'A_win':  M[diff > 0].sum(),
        'draw':   M[diff == 0].sum(),
        'B_win':  M[diff < 0].sum(),
        'A_by2':  M[diff >= 2].sum(),   # A wins by over 1.5
        'A_by3':  M[diff >= 3].sum(),   # A wins by over 2.5
        'B_by2':  M[diff <= -2].sum(),  # B wins by over 1.5
        'B_by3':  M[diff <= -3].sum(),  # B wins by over 2.5
        'o15':    M[total >= 2].sum(),
        'o25':    M[total >= 3].sum(),
        'o35':    M[total >= 4].sum(),
        'btts':   M[(i >= 1) & (j >= 1)].sum(),
    }
    return m


def odds(p):
    return float('inf') if p <= 0 else 1.0 / p


# ---------------------------------------------------------------------------
# Load the same two models the WC simulation uses (train + cache on first run,
# reload from ./models afterwards). Pass --retrain to force a fresh fit.
# ---------------------------------------------------------------------------
from core.model_store import load_or_train
RETRAIN = '--retrain' in sys.argv
lin, mlp, copula = load_or_train(retrain=RETRAIN)

lin_lam    = lambda X: lin.predict(X)
def mlp_lam(X):
    lh, la, _ = predict_mlp(mlp, X)
    return lh, la
copula_lam = lambda X: copula.predict(X)

# ---------------------------------------------------------------------------
# Features (frozen pre-tournament, same as headline odds)
# ---------------------------------------------------------------------------
groups, group_matches, ko = load_bracket()
team_names = sorted({t for ts in groups.values() for t in ts})
team_idx = {t: i for i, t in enumerate(team_names)}
for t in (TEAM_A, TEAM_B):
    if t not in team_idx:
        sys.exit(f"Unknown team '{t}'. Known teams:\n  " + "\n  ".join(team_names))
F = build_team_features(team_names)
feat = make_feat_builder(F)
iA, iB = team_idx[TEAM_A], team_idx[TEAM_B]

print(f"\nFeatures: {TEAM_A} ELO={F['elo'][iA]:.0f}  squad_avg={F['squad_avg'][iA]:.1f}")
print(f"          {TEAM_B} ELO={F['elo'][iB]:.0f}  squad_avg={F['squad_avg'][iB]:.1f}\n")

# rho per model: linear = scalar coefficient; MLP = avg over the two orientations
rho_lin = float(lin.coefficients[-1])
_, _, rho_ab = predict_mlp(mlp, feat([iA], [iB]))
_, _, rho_ba = predict_mlp(mlp, feat([iB], [iA]))
rho_mlp = float((rho_ab[0] + rho_ba[0]) / 2)

# Build neutral lambdas + full scoreline matrix for each model
mats, lams = {}, {}
for label, pred, rho in [('linear', lin_lam, rho_lin), ('mlp', mlp_lam, rho_mlp)]:
    lamA, lamB = neutral_lambdas(pred, feat, [iA], [iB])
    lams[label] = (float(lamA[0]), float(lamB[0]))
    mats[label] = dc_matrix(lamA[0], lamB[0], rho)
r_h, r_a, k_params = copula._unpack_extras(copula.coefficients)
mu_A, mu_B = neutral_lambdas(copula_lam, feat, [iA], [iB])
# neutral k: average both orientations (cancels the linear term in quad mode)
k_ab = np.asarray(copula._compute_k(feat([iA], [iB]), k_params))
k_ba = np.asarray(copula._compute_k(feat([iB], [iA]), k_params))
k_neutral = float((k_ab.mean() + k_ba.mean()) / 2)
lams['copula'] = (float(mu_A[0]), float(mu_B[0]))
mats['copula'] = copula_score_matrix(float(mu_A[0]), float(mu_B[0]), r_h, r_a, k_neutral, max_goals=MAX_GOALS)
mats['ensemble'] = (mats['linear'] + mats['mlp'] + mats['copula']) / 3
mk = {lab: markets_from_matrix(M) for lab, M in mats.items()}

# ===========================================================================
# Result + goals markets
# ===========================================================================
bar = "=" * 78
print(bar)
print(f"  {TEAM_A}  vs  {TEAM_B}   (neutral one-off match)")
print(bar)
print(f"{'model':10}{'xG '+TEAM_A[:6]:>10}{'xG '+TEAM_B[:6]:>10}"
      f"{'A win':>9}{'draw':>8}{'B win':>9}")
for lab in ('linear', 'mlp', 'copula'):
    la, lb = lams[lab]; m = mk[lab]
    print(f"{lab:10}{la:10.2f}{lb:10.2f}{m['A_win']*100:8.1f}%{m['draw']*100:7.1f}%{m['B_win']*100:8.1f}%")
m = mk['ensemble']
print(f"{'ENSEMBLE':10}{'':10}{'':10}{m['A_win']*100:8.1f}%{m['draw']*100:7.1f}%{m['B_win']*100:8.1f}%")

print(f"\n  Winning margins / goals markets   (probability  |  fair decimal odds)")
print(f"  {'-'*72}")
rows = [
    (f"{TEAM_A} win by 2+  (over 1.5)", 'A_by2'),
    (f"{TEAM_A} win by 3+  (over 2.5)", 'A_by3'),
    (f"{TEAM_B} win by 2+  (over 1.5)", 'B_by2'),
    (f"{TEAM_B} win by 3+  (over 2.5)", 'B_by3'),
    ("Total goals over 1.5",           'o15'),
    ("Total goals over 2.5",           'o25'),
    ("Total goals over 3.5",           'o35'),
    ("Both teams to score",            'btts'),
]
print(f"  {'market':32}{'linear':>14}{'mlp':>14}{'copula':>14}{'ensemble':>14}")
for name, key in rows:
    cells = "".join(f"{mk[l][key]*100:7.1f}% {odds(mk[l][key]):5.2f}" for l in ('linear', 'mlp', 'copula', 'ensemble'))
    print(f"  {name:32}{cells}")

# ===========================================================================
# 3D joint-scoreline surface (linear)
# ===========================================================================
if MAKE_PLOT:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    P = mats['linear'][:PLOT_GOALS, :PLOT_GOALS]
    g = np.arange(PLOT_GOALS)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    xp, yp = np.meshgrid(g, g, indexing='ij')
    dz = P.ravel()
    colors = plt.cm.viridis(dz / dz.max())
    ax.bar3d(xp.ravel() - 0.4, yp.ravel() - 0.4, np.zeros_like(dz),
             0.8, 0.8, dz, color=colors, shade=True)
    ax.set_xlabel(f'{TEAM_A} goals'); ax.set_ylabel(f'{TEAM_B} goals'); ax.set_zlabel('P(scoreline)')
    ax.set_xticks(g); ax.set_yticks(g)
    la, lb = lams['linear']
    ax.set_title(f'Joint scoreline distribution (linear): {TEAM_A} vs {TEAM_B}\n'
                 rf'$\lambda_{{lin}}$=({la:.2f}, {lb:.2f})')
    ax.view_init(elev=28, azim=-58)
    plt.tight_layout()
    fname = f"{TEAM_A}_vs_{TEAM_B}.png".replace(' ', '_').replace('&', 'and')
    out = Path(__file__).resolve().parent / 'figures' / fname
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nsaved 3D scoreline plot -> {out}")

# ===========================================================================
# P(team tops its group)  -- Monte Carlo, same standings rules as headline sim
# ===========================================================================
def group_top_probs(predict_fn, GRP, seed=SEED):
    gteams = groups[GRP]
    teams4 = [team_idx[t] for t in gteams]
    b_matches = [(a, b) for (g, a, b) in group_matches if g == GRP]
    rng = np.random.default_rng(seed)
    local = {tid: k for k, tid in enumerate(teams4)}
    pts = np.zeros((N, 4)); gd = np.zeros((N, 4)); gf = np.zeros((N, 4))
    for (a, b) in b_matches:
        ia, ib = team_idx[a], team_idx[b]
        lamA, lamB = neutral_lambdas(predict_fn, feat, [ia], [ib])
        gA = rng.poisson(lamA[0], N); gB = rng.poisson(lamB[0], N)
        ka, kb = local[ia], local[ib]
        gf[:, ka] += gA; gf[:, kb] += gB
        gd[:, ka] += gA - gB; gd[:, kb] += gB - gA
        pts[:, ka] += np.where(gA > gB, 3, np.where(gA == gB, 1, 0))
        pts[:, kb] += np.where(gB > gA, 3, np.where(gA == gB, 1, 0))
    score = pts * 1e6 + gd * 1e3 + gf + rng.random((N, 4)) * 1e-3
    order = np.argsort(-score, axis=1)
    teams_arr = np.array(teams4)
    win = teams_arr[order[:, 0]]
    top2 = teams_arr[order[:, :2]]
    p1 = {t: (win == team_idx[t]).mean() for t in gteams}
    p2 = {t: (top2 == team_idx[t]).any(axis=1).mean() for t in gteams}
    return p1, p2

for team in (TEAM_A, TEAM_B):
    GRP = next((g for g, ts in groups.items() if team in ts), None)
    if GRP is None:
        continue
    gteams = groups[GRP]
    top1 = {}; top2 = {}
    for lab, pred in [('linear', lin_lam), ('mlp', mlp_lam), ('copula', copula_lam)]:
        top1[lab], top2[lab] = group_top_probs(pred, GRP)
    print(f"\n{bar}\n  GROUP {GRP}  -- P(win group) and P(top-2, auto-qualify)\n{bar}")
    print(f"  {'team':22}{'win (lin)':>12}{'win (mlp)':>12}{'win (cop)':>12}{'win (ens)':>12}{'top2 (ens)':>12}")
    for t in gteams:
        e1 = (top1['linear'][t] + top1['mlp'][t] + top1['copula'][t]) / 3
        e2 = (top2['linear'][t] + top2['mlp'][t] + top2['copula'][t]) / 3
        mark = "  <--" if t == team else ""
        print(f"  {t:22}{top1['linear'][t]*100:11.1f}%{top1['mlp'][t]*100:11.1f}%"
              f"{top1['copula'][t]*100:11.1f}%{e1*100:11.1f}%{e2*100:11.1f}%{mark}")
