"""
World Cup 2026 Monte Carlo bracket simulation.

Bracket from openfootball/worldcup.json (data/worldcup/worldcup2026.json):
48 teams, 12 groups of 4, then Round of 32 -> R16 -> QF -> SF -> Final.

For each of N simulations we:
  1. simulate all 72 group matches, build standings (pts, GD, GF), rank each group,
  2. take the 8 best third-placed teams and slot them into the R32 per the
     allowed-group constraints encoded in the bracket (e.g. '3A/B/C/D/F'),
  3. play the knockout tree, resolving tied matches (shootouts) by goalkeeper rating,
  4. record the champion.

Match goals are sampled as independent Poissons from model-predicted lambdas.
Each tie is played both home/away and the lambdas averaged, so matches are treated
as NEUTRAL (the World Cup is, bar the three hosts) -- this removes the model's
built-in home-field bias. Team features (ELO, last-10 form, FC26 squad) are frozen
at their pre-tournament (< 2026-06-11) values.

Models: best linear (ridge alpha=10, full features) and best MLP (lr=0.0005,
depth=3, width=256, dropout=0, grad-clip=5), both trained on all data < 2026-06-01.
Both configs were selected on validation in experiments.py.
"""
import json
import time
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

from core.training import prepare_training_data, LinearRegressionDixonColes
from core.experiments import FULL_FEATURES, build_X, train_mlp, predict_mlp
from core.McHale_Copula import McHaleCopulaModel

base_dir = Path(__file__).resolve().parent
WC_JSON = base_dir / 'data' / 'worldcup' / 'worldcup2026.json'
ELO_CSV = base_dir / 'results' / 'match_results_elo.csv'
SQUAD_CSV = base_dir / 'results' / 'squad_features.csv'
WC_CUTOFF = '2026-06-24'      # freeze features before the first WC match
TRAIN_END = '2026-06-24'      # train models on everything before this
VAL_START = '2024-01-01'

BEST_ALPHA = 10.0
BEST_MLP = dict(lr=0.0005, depth=3, width=256, dropout=0.0, grad_clip=5.0)
BEST_COPULA_ALPHA  = 50.0   # update after mode_comparison in core.McHale_Copula
BEST_COPULA_K_MODE = 'const'  # update after mode_comparison

# Per-team feature columns (order matters: must match how feat() stacks them)
SQUAD_COLS = ['squad_avg', 'squad_std', 'attack_avg', 'midfield_avg',
              'defence_avg', 'bench_avg', 'bench_std']


# ---------------------------------------------------------------------------
# Name normalization (openfootball names vs. our datasets)
# ---------------------------------------------------------------------------
def norm(name):
    s = unicodedata.normalize('NFKD', str(name))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace('&', ' ').replace('.', ' ').replace("'", '')
    s = ' '.join(w for w in s.split() if w != 'and')
    return s


ALIAS = {  # normalized openfootball -> normalized dataset name
    'cape verde': 'cabo verde', 'czech republic': 'czechia', 'dr congo': 'congo dr',
    'south korea': 'korea republic', 'usa': 'united states', 'ivory coast': 'cote divoire',
    'turkey': 'turkiye',
}


def resolve(name, index_norm):
    """Map an openfootball team name to the matching dataset name (or None).

    Try the raw normalized name first (match-results data uses common English names
    like 'Czech Republic'); only fall back to the alias (FC squad nationality names
    like 'Czechia') if the raw name is absent.
    """
    key = norm(name)
    if key in index_norm:
        return index_norm[key]
    ak = ALIAS.get(key)
    if ak is not None and ak in index_norm:
        return index_norm[ak]
    return None


# ---------------------------------------------------------------------------
# Bracket parsing
# ---------------------------------------------------------------------------
def load_bracket():
    d = json.load(open(WC_JSON, encoding='utf-8'))
    matches = d['matches']
    groups = {}            # 'A' -> [team names]
    group_matches = []     # (group_letter, teamA, teamB)
    for m in matches:
        if m.get('group'):
            g = m['group'].split()[-1]   # 'Group A' -> 'A'
            groups.setdefault(g, [])
            for t in (m['team1'], m['team2']):
                if t not in groups[g]:
                    groups[g].append(t)
            group_matches.append((g, m['team1'], m['team2']))

    # Knockout matches, in play order. Assign 103 (3rd place) and 104 (final).
    ko = []
    for m in matches:
        if m['round'] in ('Round of 32', 'Round of 16', 'Quarter-final', 'Semi-final'):
            ko.append((m['num'], m['round'], m['team1'], m['team2']))
        elif m['round'] == 'Match for third place':
            ko.append((103, m['round'], m['team1'], m['team2']))
        elif m['round'] == 'Final':
            ko.append((104, m['round'], m['team1'], m['team2']))
    ko.sort(key=lambda x: x[0])
    return groups, group_matches, ko


def parse_ref(ref):
    """('winner', n) | ('loser', n) | ('pos', pos, group) | ('third', frozenset)."""
    if ref.startswith('W'):
        return ('winner', int(ref[1:]))
    if ref.startswith('L'):
        return ('loser', int(ref[1:]))
    if ref[0] == '3' and '/' in ref:
        return ('third', frozenset(ref[1:].split('/')))
    if ref[0] in '123' and len(ref) == 2:
        return ('pos', int(ref[0]), ref[1])
    raise ValueError(f"unparsable ref {ref!r}")


# ---------------------------------------------------------------------------
# Per-team feature table
# ---------------------------------------------------------------------------
def build_team_features(team_names):
    """Return a dict of per-team numpy arrays (indexed 0..47) for all features."""
    mr = pd.read_csv(ELO_CSV)
    mr['date'] = pd.to_datetime(mr['date'])
    mr = mr[mr['date'] < WC_CUTOFF]
    home = mr[['date', 'home_team', 'home_elo', 'home_scored_last10', 'home_conceded_last10']].rename(
        columns={'home_team': 'team', 'home_elo': 'elo', 'home_scored_last10': 's10', 'home_conceded_last10': 'c10'})
    away = mr[['date', 'away_team', 'away_elo', 'away_scored_last10', 'away_conceded_last10']].rename(
        columns={'away_team': 'team', 'away_elo': 'elo', 'away_scored_last10': 's10', 'away_conceded_last10': 'c10'})
    appall = pd.concat([home, away]).sort_values('date')
    last = appall.groupby('team').last()
    elo_norm = {norm(t): t for t in last.index}

    sf = pd.read_csv(SQUAD_CSV)
    sf26 = sf[sf['year'] == 2026].set_index('team')
    squad_norm = {norm(t): t for t in sf26.index}

    n = len(team_names)
    feats = {k: np.zeros(n) for k in ['elo', 's10', 'c10', 'gk'] + SQUAD_COLS}
    # sensible fallbacks
    elo_default = last['elo'].quantile(0.10)
    squad_means = {c: sf26[c].mean() for c in SQUAD_COLS}

    unresolved = []
    for i, name in enumerate(team_names):
        en = resolve(name, elo_norm)
        if en is not None:
            feats['elo'][i] = last.loc[en, 'elo']
            feats['s10'][i] = np.nan_to_num(last.loc[en, 's10'])
            feats['c10'][i] = np.nan_to_num(last.loc[en, 'c10'])
        else:
            feats['elo'][i] = elo_default
            unresolved.append(('elo', name))
        sn = resolve(name, squad_norm)
        if sn is not None:
            for c in SQUAD_COLS:
                feats[c][i] = sf26.loc[sn, c]
            feats['gk'][i] = sf26.loc[sn, 'gk_rating']
        else:
            for c in SQUAD_COLS:
                feats[c][i] = squad_means[c]
            feats['gk'][i] = np.nan
            unresolved.append(('squad', name))

    # GK rating for shootouts: fall back to defence_avg, then squad_avg, if absent.
    gkv = feats['gk']
    gkv = np.where(np.isnan(gkv), feats['defence_avg'], gkv)
    gkv = np.where(np.isnan(gkv), feats['squad_avg'], gkv)
    feats['gk'] = gkv

    # Match training, which fills missing squad stats with 0; an unhandled NaN
    # here would poison the predicted lambdas. (Thin squads are already repaired
    # in squad_features.py, so this is just a final safety net.)
    for c in SQUAD_COLS:
        feats[c] = np.nan_to_num(feats[c], nan=0.0)
    if unresolved:
        print("  WARNING unresolved (using fallback):", unresolved)
    return feats


# ---------------------------------------------------------------------------
# Model prediction (neutral: average both home/away orientations)
# ---------------------------------------------------------------------------
def make_feat_builder(F):
    def feat(home_idx, away_idx):
        return np.column_stack([
            F['elo'][home_idx], F['elo'][away_idx],
            F['s10'][home_idx], F['c10'][home_idx], F['s10'][away_idx], F['c10'][away_idx],
            F['squad_avg'][home_idx], F['squad_std'][home_idx], F['attack_avg'][home_idx],
            F['midfield_avg'][home_idx], F['defence_avg'][home_idx], F['bench_avg'][home_idx],
            F['bench_std'][home_idx],
            F['squad_avg'][away_idx], F['squad_std'][away_idx], F['attack_avg'][away_idx],
            F['midfield_avg'][away_idx], F['defence_avg'][away_idx], F['bench_avg'][away_idx],
            F['bench_std'][away_idx],
        ])
    return feat


def neutral_lambdas(predict_fn, feat, idxA, idxB):
    idxA = np.asarray(idxA); idxB = np.asarray(idxB)
    lhA, laB = predict_fn(feat(idxA, idxB))   # A home, B away
    lhB, laA = predict_fn(feat(idxB, idxA))   # B home, A away
    return (lhA + laA) / 2.0, (laB + lhB) / 2.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(predict_fn, feat, groups, group_matches, ko, team_idx, N, seed=0,
             draw_method='gk', gk=None):
    """Run N tournament simulations.

    draw_method controls how a tied knockout match (penalty shootout) is decided:
      'gk'     -> P(team A wins) = gk_A / (gk_A + gk_B)  (goalkeeper-controlled)
      'lambda' -> P(team A wins) = lam_A / (lam_A + lam_B)  (overall strength)
    """
    rng = np.random.default_rng(seed)
    gl = list(groups.keys())

    # ---- group stage ----
    group_pos = {}     # g -> {1: idx_array, 2:..., 3:...}
    third_score = {}   # g -> (N,) ranking score of its 3rd-place team
    third_team = {}    # g -> (N,) team idx of its 3rd-place team
    for g in gl:
        teams4 = [team_idx[t] for t in groups[g]]
        pts = np.zeros((N, 4)); gd = np.zeros((N, 4)); gf = np.zeros((N, 4))
        local = {tid: k for k, tid in enumerate(teams4)}
        for (gg, a, b) in group_matches:
            if gg != g:
                continue
            ia, ib = team_idx[a], team_idx[b]
            lamA, lamB = neutral_lambdas(predict_fn, feat, [ia], [ib])
            gA = rng.poisson(lamA[0], N); gB = rng.poisson(lamB[0], N)
            ka, kb = local[ia], local[ib]
            gf[:, ka] += gA; gf[:, kb] += gB
            gd[:, ka] += gA - gB; gd[:, kb] += gB - gA
            pts[:, ka] += np.where(gA > gB, 3, np.where(gA == gB, 1, 0))
            pts[:, kb] += np.where(gB > gA, 3, np.where(gA == gB, 1, 0))
        score = pts * 1e6 + gd * 1e3 + gf + rng.random((N, 4)) * 1e-3  # random tiebreak
        order = np.argsort(-score, axis=1)
        teams4 = np.array(teams4)
        group_pos[g] = {p: teams4[order[:, p - 1]] for p in (1, 2, 3)}
        third_score[g] = np.take_along_axis(score, order[:, 2:3], axis=1)[:, 0]
        third_team[g] = teams4[order[:, 2]]

    # ---- best 8 third-placed teams ----
    TS = np.stack([third_score[g] for g in gl], axis=1)   # (N,12)
    qual_cols = np.argsort(-TS, axis=1)[:, :8]            # (N,8) cols into gl

    # third-place R32 slots and their allowed groups
    third_slots = []
    for (num, rnd, r1, r2) in ko:
        for r in (r1, r2):
            p = parse_ref(r)
            if p[0] == 'third':
                third_slots.append((num, p[1]))

    assign_cache = {}

    def assign(qual_groups):
        key = qual_groups
        if key in assign_cache:
            return assign_cache[key]
        order = sorted(third_slots, key=lambda s: len(s[1] & qual_groups))
        res, used = {}, set()

        def bt(i):
            if i == len(order):
                return True
            num, allowed = order[i]
            for gg in sorted(allowed & qual_groups):
                if gg not in used:
                    used.add(gg); res[num] = gg
                    if bt(i + 1):
                        return True
                    used.discard(gg); del res[num]
            return False
        bt(0)
        assign_cache[key] = dict(res)
        return assign_cache[key]

    third_assigned = {num: np.empty(N, dtype=int) for num, _ in third_slots}
    for s in range(N):
        Q = frozenset(gl[c] for c in qual_cols[s])
        amap = assign(Q)
        for num, _ in third_slots:
            g = amap.get(num)
            if g is None:                       # fallback (matching failed: rare)
                g = gl[qual_cols[s][0]]
            third_assigned[num][s] = third_team[g][s]

    # ---- knockout ----
    winners, losers = {}, {}

    def resolve_ref(ref, num):
        p = parse_ref(ref)
        if p[0] == 'winner':
            return winners[p[1]]
        if p[0] == 'loser':
            return losers[p[1]]
        if p[0] == 'pos':
            return group_pos[p[2]][p[1]]
        if p[0] == 'third':
            return third_assigned[num]
        raise ValueError(ref)

    for (num, rnd, r1, r2) in ko:
        idxA = resolve_ref(r1, num); idxB = resolve_ref(r2, num)
        lamA, lamB = neutral_lambdas(predict_fn, feat, idxA, idxB)
        gA = rng.poisson(lamA); gB = rng.poisson(lamB)
        Awin = gA > gB
        draw = gA == gB
        # Penalty shootout: a coin flip the goalkeepers control. Weight by GK rating
        # (or, as a fallback, by overall strength via the predicted lambdas).
        if draw_method == 'gk':
            gkA = gk[idxA]; gkB = gk[idxB]
            p_adv = gkA / (gkA + gkB)
        else:
            p_adv = lamA / (lamA + lamB)
        coin = rng.random(N) < p_adv
        A_adv = Awin | (draw & coin)
        winners[num] = np.where(A_adv, idxA, idxB)
        losers[num] = np.where(A_adv, idxB, idxA)

    champion = winners[104]
    finalists = np.concatenate([winners[101], winners[102]])  # who reached the final
    return champion, finalists


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_models():
    print("  preparing train (<2025-10-01) + val (2024..2025-10) splits...", flush=True)
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

    copula = McHaleCopulaModel(Xtr.shape[1], k_mode=BEST_COPULA_K_MODE)
    copula.fit(Xtr, hgtr, agtr, weights=wtr, alpha=BEST_COPULA_ALPHA,
               max_iter=500, verbose=True)

    lin_pred = lambda X: lin.predict(X)
    def mlp_pred(X):
        lh, la, _ = predict_mlp(mlp, X)
        return lh, la
    copula_pred = lambda X: copula.predict(X)
    return lin_pred, mlp_pred, copula_pred, copula


def main(N=10000):
    t0 = time.time()
    groups, group_matches, ko = load_bracket()
    team_names = sorted({t for ts in groups.values() for t in ts})
    team_idx = {t: i for i, t in enumerate(team_names)}
    print(f"[{time.time()-t0:.1f}s] Bracket: {len(team_names)} teams, "
          f"{len(group_matches)} group matches, {len(ko)} knockout matches", flush=True)

    F = build_team_features(team_names)
    feat = make_feat_builder(F)

    lin_pred, mlp_pred, copula_pred, copula = train_models()
    print(f"[{time.time()-t0:.1f}s] Models trained. Running {N} sims each...", flush=True)

    n_teams = len(team_names)
    cols = {'team': team_names}
    for label, pred in [('linear', lin_pred), ('mlp', mlp_pred), ('copula', copula_pred)]:
        for method in ('gk', 'lambda'):
            champ, finalists = simulate(pred, feat, groups, group_matches, ko, team_idx, N,
                                        draw_method=method, gk=F['gk'])
            cols[f'champion_{label}_{method}'] = np.bincount(champ, minlength=n_teams) / N
            if method == 'gk':
                cols[f'final_{label}'] = np.bincount(finalists, minlength=n_teams) / N
            print(f"[{time.time()-t0:.1f}s] {label}/{method}: done", flush=True)

    df = pd.DataFrame(cols).sort_values('champion_linear_gk', ascending=False).reset_index(drop=True)
    # Ensemble: average all three models' probabilities
    df['champion_ensemble_gk'] = (df['champion_linear_gk'] + df['champion_mlp_gk'] + df['champion_copula_gk']) / 3
    df['final_ensemble'] = (df['final_linear'] + df['final_mlp'] + df['final_copula']) / 3
    out_csv = base_dir / 'results' / 'wc_odds.csv'
    df.to_csv(out_csv, index=False)
    print(f"[{time.time()-t0:.1f}s] Saved {out_csv}", flush=True)

    # Draw-resolution comparison: how much do title odds move (GK vs lambda)?
    for label in ('linear', 'mlp', 'copula'):
        d = (df[f'champion_{label}_gk'] - df[f'champion_{label}_lambda']).abs()
        print(f"\n{label}: mean |GK - lambda| champ-prob shift = {d.mean()*100:.2f}pp, "
              f"max = {d.max()*100:.2f}pp")
        moved = df.assign(delta=df[f'champion_{label}_gk'] - df[f'champion_{label}_lambda'])
        moved = moved.reindex(moved['delta'].abs().sort_values(ascending=False).index)
        print(f"  biggest movers ({label}, GK minus lambda):")
        for _, r in moved.head(5).iterrows():
            print(f"    {r['team']:14} {r['delta']*100:+.2f}pp  "
                  f"(gk {r[f'champion_{label}_gk']*100:.1f}% vs lam {r[f'champion_{label}_lambda']*100:.1f}%)")

    print("\nTop 16 by linear championship probability (GK draw resolution):")
    show = ['team', 'champion_linear_gk', 'champion_mlp_gk', 'champion_copula_gk',
            'final_linear', 'final_mlp', 'final_copula']
    print(df[show].head(16).to_string(index=False,
          formatters={c: '{:.3f}'.format for c in show if c != 'team'}))


if __name__ == "__main__":
    main()
