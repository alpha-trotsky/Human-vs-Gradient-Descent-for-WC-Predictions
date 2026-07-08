"""
Five goal-scoring models for the matches in odds.csv (WC 2026 finals + international
friendlies), all sharing one team-strength engine (classic Dixon-Coles attack/defence
parameterisation, log(mu) = home_adv + attack_home - defence_away, fit by weighted MLE
with an exponential time-decay weight exp(-alpha*days_elapsed)):

  M1  Poisson marginals              + Dixon-Coles low-score rho correction
  M2  Negative-Binomial marginals    + Dixon-Coles low-score rho correction
  M3  Negative-Binomial marginals    + Frank copula (full joint dependence)
  M4  M3's r/theta, mu_H/mu_A MLE-calibrated per match to devigged Kalshi 1X2 odds
  M5  M3's r/theta, mu_H/mu_A MLE-calibrated per match to Kalshi 1X2 + O/U odds

Historical corpus: results/results.csv (49k+ international matches, 1872-2026).
Test matches: odds.csv (419 matches, WC 2026 + friendlies, Kalshi prices ~2h pre-kickoff).

Gradient trick: analytic per-team gradients would require differentiating through
Negative-Binomial CDFs and the Frank copula -- doable but error-prone. Instead we take
central-difference derivatives of the (vectorised, O(n_matches)) log-likelihood ONLY
with respect to the two scalar per-match quantities (mu_home, mu_away) and the handful
of shared scalars (r_home, r_away, rho/theta) -- never with respect to the O(n_teams)
attack/defence parameters directly. Because mu_home/mu_away are simple exponentials of
(home_adv + attack - defence), the chain rule back to every attack/defence parameter is
then exact and analytic (a cheap scatter-add). This keeps one gradient evaluation at a
constant ~10 vectorised likelihood calls, independent of the number of teams, which is
what makes refitting 41 times (once per unique match date) tractable.
"""
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson, nbinom

from core.McHale_Copula import frank_copula, joint_pmf, score_matrix as _frank_score_matrix
from core.evaluate import (
    ranked_probability_score, log_loss as nll_score, accuracy, actual_outcomes,
)

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / 'results' / 'results.csv'
ODDS_CSV = ROOT / 'odds.csv'
SCORE_CSV = ROOT / 'kalshi_wc_exact_score_odds.csv'

MAX_GOALS = 10   # truncation grid for full joint distributions (EV sim, normalisation)
CAL_GOALS = 5    # truncation grid for the calibration heatmaps

LOOKBACK_YEARS = 5     # trailing window of historical data used at each refit
RIDGE = 1e-6           # tiny L2 penalty on attack/defence -- purely for identifiability
EPS_MU = 1e-4          # relative bump for central differences on mu_home/mu_away
EPS_EXTRA = 1e-4       # additive bump for central differences on rho/theta/log_r
ALPHA_DECAY = 0.0025   # per-day match weight exp(-alpha*days); ~0.76yr half-life,
                       # matches core/training.py's most recent default

KINDS = ('poisson_dc', 'nbinom_dc', 'nbinom_frank')          # M1, M2, M3
MODEL_KEYS = KINDS + ('m4', 'm5')                             # + calibrated M4, M5
MODEL_NAMES = {
    'poisson_dc':   'M1 Poisson+DC',
    'nbinom_dc':    'M2 NegBin+DC',
    'nbinom_frank': 'M3 NegBin+Frank',
    'm4':           'M4 NegBin+Frank (calib 1X2)',
    'm5':           'M5 NegBin+Frank (calib 1X2+O/U)',
}
THRESHOLD = 0.02   # BUY/SELL edge required vs devigged market price ($1 contracts)


# ---------------------------------------------------------------------------
# Name normalisation: Kalshi (odds.csv) name -> results.csv name
# ---------------------------------------------------------------------------
def norm(name):
    s = unicodedata.normalize('NFKD', str(name))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace('&', ' ').replace('.', ' ').replace("'", '')
    s = ' '.join(w for w in s.split() if w != 'and')
    return s


# Extends backtest/backtest_odds.py's _ODDS_TO_RES with a few more aliases needed
# for the friendlies in odds.csv (that file only covers the 2026 WC team list).
ODDS_TO_RES = {
    'czechia': 'czech republic',
    'congo dr': 'dr congo',
    'korea republic': 'south korea',
    'republic of korea': 'south korea',
    'ksa': 'saudi arabia',
    'turkiye': 'turkey',
    'ir iran': 'iran',
    'usa': 'united states',
    'cabo verde': 'cabo verde',
    'br virgin islands': 'british virgin islands',
    'virgin islands british': 'british virgin islands',
    'china pr': 'china',
    'curacao': 'curaçao',
    'ireland': 'republic of ireland',
}


def to_res(name):
    n = norm(name)
    return ODDS_TO_RES.get(n, n)


# ---------------------------------------------------------------------------
# Historical corpus
# ---------------------------------------------------------------------------
def load_historical():
    df = pd.read_csv(RESULTS_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df['home_norm'] = df['home_team'].map(to_res)
    df['away_norm'] = df['away_team'].map(to_res)
    df['neutral_f'] = (df['neutral'].astype(str).str.upper() == 'TRUE').astype(float)
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    df = df.dropna(subset=['home_score', 'away_score']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# odds.csv -> one row per real-world match
# ---------------------------------------------------------------------------
def _event_suffix(event_ticker):
    return event_ticker.split('-', 1)[1] if '-' in event_ticker else event_ticker


def _split_match_teams(match_title):
    base = match_title.split(':')[0]
    a, b = base.split(' vs ')
    return a.strip(), b.strip()


def _devig_1x2(grp):
    """grp: rows for one moneyline event (KXWCGAME/KXINTLFRIENDLYGAME), 3 outcomes.
    Returns (p_home, p_draw, p_away) devigged (normalised to sum 1), or None."""
    # De-duplicate: if a market got re-listed (rare), keep the highest-volume row
    # per outcome bucket.
    grp = grp.sort_values('volume_at_snapshot').drop_duplicates('outcome', keep='last')
    home_raw, away_raw = _split_match_teams(grp['match'].iloc[0])
    home_n, away_n = norm(home_raw), norm(away_raw)

    p_home = p_draw = p_away = None
    for _, row in grp.iterrows():
        out = str(row['outcome'])
        out = out[len('Reg Time: '):] if out.startswith('Reg Time: ') else out
        out_n = norm(out)
        price = row['price_2h_before']
        if pd.isna(price):
            continue
        if out_n == 'tie':
            p_draw = float(price)
        elif out_n == home_n:
            p_home = float(price)
        elif out_n == away_n:
            p_away = float(price)
    if p_home is None or p_draw is None or p_away is None:
        return None
    total = p_home + p_draw + p_away
    if total <= 0:
        return None
    return (p_home / total, p_draw / total, p_away / total)


def _parse_ou(grp):
    """grp: rows for one totals event. Returns list of (threshold, p_over) sorted."""
    out = []
    for _, row in grp.iterrows():
        price = row['price_2h_before']
        if pd.isna(price):
            continue
        s = str(row['outcome'])
        s = s[len('Reg Time: '):] if s.startswith('Reg Time: ') else s
        # "Over 2.5 goals scored" -> 2.5
        try:
            g = float(s.split('Over', 1)[1].split('goals')[0].strip())
        except (IndexError, ValueError):
            continue
        out.append((g, float(price)))
    out.sort(key=lambda t: t[0])
    return out


def _parse_score_outcome(outcome, home_raw, away_raw):
    """'Reg Time: Draw 0-0' / 'Reg Time: Belgium wins 4-1' -> (home_goals, away_goals)."""
    s = str(outcome)
    s = s[len('Reg Time: '):] if s.startswith('Reg Time: ') else s
    s = s.strip()
    if s.lower().startswith('draw'):
        i = int(s.split()[-1].split('-')[0])
        j = int(s.split()[-1].split('-')[1])
        return (i, j)
    if ' wins ' not in s:
        return None
    team, score = s.rsplit(' wins ', 1)
    team = team.strip()
    try:
        a, b = (int(x) for x in score.strip().split('-'))
    except ValueError:
        return None
    if norm(team) == norm(home_raw):
        return (a, b)
    if norm(team) == norm(away_raw):
        return (b, a)
    return None


def _devig_scores(grp, home_raw, away_raw):
    """Returns {(i,j): (p_devig, decimal_odds)} normalised to sum to 1 across all
    listed scorelines for the match (same "normalise to 1" convention as the 1X2 devig)."""
    raw = {}
    for _, row in grp.iterrows():
        price = row['price_2h_before']
        if pd.isna(price) or price <= 0:
            continue
        ij = _parse_score_outcome(row['outcome'], home_raw, away_raw)
        if ij is None:
            continue
        raw[ij] = float(price)
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {ij: (p / total, total / p if p > 0 else np.nan) for ij, p in raw.items()}
    # decimal_odds here is against the DEVIGGED probability's implied fair odds is NOT
    # what we bet at -- the actual tradable decimal odds are 1/raw_price (see build below).


def load_match_table():
    """One row per real-world match in odds.csv, with devigged 1X2 / O-U market
    probabilities, actual final score (joined from results.csv), and -- for World Cup
    matches only -- devigged Kalshi correct-score odds (joined from
    kalshi_wc_exact_score_odds.csv)."""
    odds = pd.read_csv(ODDS_CSV)
    odds['event_suffix'] = odds['event_ticker'].map(_event_suffix)
    odds['kickoff_utc'] = pd.to_datetime(odds['kickoff_utc'], format='ISO8601').dt.tz_localize(None)

    hist = load_historical()
    # Only need a (home_norm, away_norm, date) -> (hg, ag) lookup for scoring odds.csv
    # matches; try exact date, then +-1 day (Kalshi kickoff_utc can spill past local
    # midnight relative to the date recorded in results.csv).
    score_lut = {}
    for _, r in hist.iterrows():
        score_lut.setdefault((r['home_norm'], r['away_norm'], r['date'].normalize()),
                             (r['home_score'], r['away_score'], r['neutral_f']))

    def find_score(home_n, away_n, kickoff):
        for delta in (0, -1, 1, -2, 2):
            day = (kickoff + pd.Timedelta(days=delta)).normalize()
            key = (home_n, away_n, day)
            if key in score_lut:
                return score_lut[key]
            swapped = (away_n, home_n, day)
            if swapped in score_lut:
                # results.csv recorded the opposite home/away orientation (common
                # on neutral-site tournament fixtures) -- swap the scores back.
                ag_, hg_, neutral_ = score_lut[swapped]
                return hg_, ag_, neutral_
        return (None, None, None)

    score_odds = pd.read_csv(SCORE_CSV) if SCORE_CSV.exists() else pd.DataFrame()
    if not score_odds.empty:
        score_odds['event_suffix'] = score_odds['event_ticker'].map(_event_suffix)

    rows = []
    money = odds[odds['series'].isin(['KXWCGAME', 'KXINTLFRIENDLYGAME'])]
    for suffix, grp in money.groupby('event_suffix'):
        is_wc = (grp['series'] == 'KXWCGAME').any()
        home_raw, away_raw = _split_match_teams(grp['match'].iloc[0])
        home_n, away_n = to_res(home_raw), to_res(away_raw)
        kickoff = grp['kickoff_utc'].iloc[0]

        mkt_1x2 = _devig_1x2(grp)

        tot_series = 'KXWCTOTAL' if is_wc else 'KXINTLFRIENDLYTOTAL'
        tot_grp = odds[(odds['series'] == tot_series) & (odds['event_suffix'] == suffix)]
        ou = _parse_ou(tot_grp) if not tot_grp.empty else []

        hg, ag, neutral = find_score(home_n, away_n, kickoff)

        sc_odds = {}
        if is_wc and not score_odds.empty:
            sc_grp = score_odds[score_odds['event_suffix'] == suffix]
            if not sc_grp.empty:
                sc_odds = _devig_scores(sc_grp, home_raw, away_raw)

        rows.append(dict(
            match_id=suffix, home_raw=home_raw, away_raw=away_raw,
            home_norm=home_n, away_norm=away_n, kickoff=kickoff, is_wc=is_wc,
            neutral=neutral if neutral is not None else (1.0 if is_wc else 0.0),
            actual_hg=hg, actual_ag=ag,
            mkt_1x2=mkt_1x2, ou=ou, score_odds=sc_odds,
        ))

    out = pd.DataFrame(rows).sort_values('kickoff').reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Dixon-Coles low-score tau correction (shared by M1 Poisson and M2 NB)
# ---------------------------------------------------------------------------
def dc_tau(mu_h, mu_a, hg, ag, rho):
    tau = np.ones_like(mu_h)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1 - mu_h[m00] * mu_a[m00] * rho
    tau[m01] = 1 + mu_h[m01] * rho
    tau[m10] = 1 + mu_a[m10] * rho
    tau[m11] = 1 - rho
    return np.clip(tau, 1e-10, None)


def dc_tau_grid(mu_h, mu_a, rho, max_goals):
    """Tau applied to a full (max_goals+1, max_goals+1) grid (single match)."""
    tau = np.ones((max_goals + 1, max_goals + 1))
    tau[0, 0] = 1 - mu_h * mu_a * rho
    tau[0, 1] = 1 + mu_h * rho
    tau[1, 0] = 1 + mu_a * rho
    tau[1, 1] = 1 - rho
    return np.clip(tau, 1e-10, None)


# ---------------------------------------------------------------------------
# Per-match log-likelihood kernels (vectorised over matches)
# ---------------------------------------------------------------------------
def poisson_dc_loglik(mu_h, mu_a, hg, ag, rho):
    p = poisson.pmf(hg, mu_h) * poisson.pmf(ag, mu_a) * dc_tau(mu_h, mu_a, hg, ag, rho)
    return np.log(np.clip(p, 1e-300, None))


def nbinom_dc_loglik(mu_h, mu_a, hg, ag, r_h, r_a, rho):
    p = (nbinom.pmf(hg, r_h, r_h / (r_h + mu_h))
       * nbinom.pmf(ag, r_a, r_a / (r_a + mu_a))
       * dc_tau(mu_h, mu_a, hg, ag, rho))
    return np.log(np.clip(p, 1e-300, None))


def nbinom_frank_loglik(mu_h, mu_a, hg, ag, r_h, r_a, theta):
    return np.log(joint_pmf(hg, ag, mu_h, mu_a, r_h, r_a, theta))


# ---------------------------------------------------------------------------
# Full-grid scoreline matrices (single match; used for 1X2/O-U aggregation,
# EV simulation, calibration heatmaps)
# ---------------------------------------------------------------------------
def poisson_dc_grid(mu_h, mu_a, rho, max_goals=MAX_GOALS):
    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, mu_h), poisson.pmf(g, mu_a))
    M = M * dc_tau_grid(mu_h, mu_a, rho, max_goals)
    M = np.clip(M, 0, None)
    return M / M.sum()


def nbinom_dc_grid(mu_h, mu_a, r_h, r_a, rho, max_goals=MAX_GOALS):
    g = np.arange(max_goals + 1)
    ph, pa = r_h / (r_h + mu_h), r_a / (r_a + mu_a)
    M = np.outer(nbinom.pmf(g, r_h, ph), nbinom.pmf(g, r_a, pa))
    M = M * dc_tau_grid(mu_h, mu_a, rho, max_goals)
    M = np.clip(M, 0, None)
    return M / M.sum()


def nbinom_frank_grid(mu_h, mu_a, r_h, r_a, theta, max_goals=MAX_GOALS):
    return _frank_score_matrix(mu_h, mu_a, r_h, r_a, theta, max_goals)


def grid_to_1x2(M):
    home = np.tril(M, -1).sum()
    draw = np.trace(M)
    away = np.triu(M, 1).sum()
    return home, draw, away


def grid_to_over(M, threshold):
    """P(total goals > threshold), threshold like 2.5."""
    n = M.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return M[(i + j) > threshold].sum()


# ---------------------------------------------------------------------------
# Central-difference gradient w.r.t. (mu_home, mu_away, *extra) -- generic,
# reused by all three MLE-fit model kernels. See module docstring for why.
# ---------------------------------------------------------------------------
def _central_diff_grad(loglik_fn, mu_h, mu_a, extra, hg, ag):
    n_extra = len(extra)
    base_args = (hg, ag) + tuple(extra)

    lp = loglik_fn(mu_h * (1 + EPS_MU), mu_a, *base_args)
    lm = loglik_fn(mu_h * (1 - EPS_MU), mu_a, *base_args)
    d_mu_h = (lp - lm) / (2 * EPS_MU * mu_h)

    lp = loglik_fn(mu_h, mu_a * (1 + EPS_MU), *base_args)
    lm = loglik_fn(mu_h, mu_a * (1 - EPS_MU), *base_args)
    d_mu_a = (lp - lm) / (2 * EPS_MU * mu_a)

    d_extra = []
    for k in range(n_extra):
        e_p = list(extra); e_p[k] = e_p[k] + EPS_EXTRA
        e_m = list(extra); e_m[k] = e_m[k] - EPS_EXTRA
        lp = loglik_fn(mu_h, mu_a, hg, ag, *e_p)
        lm = loglik_fn(mu_h, mu_a, hg, ag, *e_m)
        d_extra.append((lp - lm) / (2 * EPS_EXTRA))

    ll = loglik_fn(mu_h, mu_a, *base_args)
    return ll, d_mu_h, d_mu_a, d_extra


# ---------------------------------------------------------------------------
# Generic team-strength MLE fit (shared by M1 / M2 / M3)
# ---------------------------------------------------------------------------
class StrengthFit:
    """Holds a fitted attack/defence/home_adv (+ shared r/rho/theta) parameter set
    for one team universe, plus lookup helpers to get (mu_home, mu_away) for any
    (home, away) pair seen in that universe."""

    def __init__(self, team_index, home_adv, attack, defence, extra):
        self.team_index = team_index
        self.home_adv = home_adv
        self.attack = attack
        self.defence = defence
        self.extra = extra   # dict, model-specific (rho) or (r_h, r_a, rho) or (r_h, r_a, theta)

    def _idx(self, name_norm):
        return self.team_index.get(name_norm)  # None -> unseen team, treated as average

    def mu(self, home_norm, away_norm, neutral):
        hi, ai = self._idx(home_norm), self._idx(away_norm)
        a_h = self.attack[hi] if hi is not None else 0.0
        d_h = self.defence[hi] if hi is not None else 0.0
        a_a = self.attack[ai] if ai is not None else 0.0
        d_a = self.defence[ai] if ai is not None else 0.0
        mu_h = np.exp(np.clip(self.home_adv * (1 - neutral) + a_h - d_a, -10, 10))
        mu_a = np.exp(np.clip(a_a - d_h, -10, 10))
        return float(mu_h), float(mu_a)


def _kind_config(kind):
    """(loglik_fn, n_extra, extra_names, x0_extra, bounds_extra)."""
    if kind == 'poisson_dc':
        return poisson_dc_loglik, 1, ['raw_rho'], [0.0], [(-5, 5)]
    if kind == 'nbinom_dc':
        return nbinom_dc_loglik, 3, ['log_r_h', 'log_r_a', 'raw_rho'], [1.0, 1.0, 0.0], \
               [(-5, 5), (-5, 5), (-5, 5)]
    if kind == 'nbinom_frank':
        return nbinom_frank_loglik, 3, ['log_r_h', 'log_r_a', 'theta'], [1.0, 1.0, 0.0], \
               [(-5, 5), (-5, 5), (-20, 20)]
    raise ValueError(kind)


def _unpack_extra(kind, raw_extra):
    """raw optimiser values -> transformed (real-domain) values passed to loglik_fn."""
    if kind == 'poisson_dc':
        rho = 0.9 * np.tanh(raw_extra[0])
        return [rho], [0.9 * (1 - np.tanh(raw_extra[0]) ** 2)]   # (values, d(value)/d(raw))
    if kind == 'nbinom_dc':
        log_r_h, log_r_a, raw_rho = raw_extra
        r_h, r_a = np.exp(np.clip(log_r_h, -5, 5)), np.exp(np.clip(log_r_a, -5, 5))
        rho = 0.9 * np.tanh(raw_rho)
        return [r_h, r_a, rho], [r_h, r_a, 0.9 * (1 - np.tanh(raw_rho) ** 2)]
    if kind == 'nbinom_frank':
        log_r_h, log_r_a, theta_raw = raw_extra
        r_h, r_a = np.exp(np.clip(log_r_h, -5, 5)), np.exp(np.clip(log_r_a, -5, 5))
        theta = np.clip(theta_raw, -20, 20)
        return [r_h, r_a, theta], [r_h, r_a, 1.0]
    raise ValueError(kind)


def fit_strength_model(matches, kind, ridge=RIDGE, max_iter=300, verbose=False):
    """matches: DataFrame with columns home_norm, away_norm, home_score, away_score,
    weight, neutral_f. Fits home_adv + attack[T] + defence[T] + extra via weighted MLE.
    kind in {'poisson_dc', 'nbinom_dc', 'nbinom_frank'}."""
    teams = sorted(set(matches['home_norm']) | set(matches['away_norm']))
    team_index = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    home_idx = matches['home_norm'].map(team_index).to_numpy()
    away_idx = matches['away_norm'].map(team_index).to_numpy()
    hg = matches['home_score'].to_numpy(dtype=float)
    ag = matches['away_score'].to_numpy(dtype=float)
    w = matches['weight'].to_numpy(dtype=float)
    neutral = matches['neutral_f'].to_numpy(dtype=float)

    loglik_fn, n_extra, extra_names, x0_extra, bounds_extra = _kind_config(kind)

    def unpack(params):
        home_adv = params[0]
        attack = params[1:1 + T]
        defence = params[1 + T:1 + 2 * T]
        raw_extra = params[1 + 2 * T:]
        return home_adv, attack, defence, raw_extra

    def nll_and_grad(params):
        home_adv, attack, defence, raw_extra = unpack(params)
        mu_h = np.exp(np.clip(home_adv * (1 - neutral) + attack[home_idx] - defence[away_idx], -10, 10))
        mu_a = np.exp(np.clip(attack[away_idx] - defence[home_idx], -10, 10))

        extra_vals, extra_jac = _unpack_extra(kind, raw_extra)
        ll, d_mu_h, d_mu_a, d_extra = _central_diff_grad(loglik_fn, mu_h, mu_a, extra_vals, hg, ag)

        nll = -(w * ll).sum() + ridge * (np.dot(attack, attack) + np.dot(defence, defence))

        dnll_dmu_h = -w * d_mu_h
        dnll_dmu_a = -w * d_mu_a

        grad = np.zeros_like(params)
        # home_adv
        grad[0] = np.sum(dnll_dmu_h * mu_h * (1 - neutral))
        # attack / defence via scatter-add (exact chain rule through mu = exp(...))
        g_attack = np.zeros(T)
        g_defence = np.zeros(T)
        np.add.at(g_attack, home_idx, dnll_dmu_h * mu_h)
        np.add.at(g_attack, away_idx, dnll_dmu_a * mu_a)
        np.add.at(g_defence, away_idx, -dnll_dmu_h * mu_h)
        np.add.at(g_defence, home_idx, -dnll_dmu_a * mu_a)
        g_attack += 2 * ridge * attack
        g_defence += 2 * ridge * defence
        grad[1:1 + T] = g_attack
        grad[1 + T:1 + 2 * T] = g_defence
        # extra (raw optimiser scale, via chain rule through the transform)
        for k in range(n_extra):
            grad[1 + 2 * T + k] = np.sum(-w * d_extra[k]) * extra_jac[k]

        return nll, grad

    x0 = np.zeros(1 + 2 * T + n_extra)
    x0[1 + 2 * T:] = x0_extra
    bounds = [(-3, 3)] + [(-8, 8)] * (2 * T) + bounds_extra

    result = minimize(nll_and_grad, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6})

    home_adv, attack, defence, raw_extra = unpack(result.x)
    extra_vals, _ = _unpack_extra(kind, raw_extra)
    extra_dict = dict(zip(extra_names[:0] or [], []))  # placeholder, replaced below
    if kind == 'poisson_dc':
        extra_dict = {'rho': extra_vals[0]}
    elif kind == 'nbinom_dc':
        extra_dict = {'r_h': extra_vals[0], 'r_a': extra_vals[1], 'rho': extra_vals[2]}
    elif kind == 'nbinom_frank':
        extra_dict = {'r_h': extra_vals[0], 'r_a': extra_vals[1], 'theta': extra_vals[2]}

    if verbose:
        print(f"    [{kind}] n={len(matches):,} teams={T} NLL={result.fun:.1f} "
              f"extra={extra_dict} ({result.message})")

    return StrengthFit(team_index, home_adv, attack, defence, extra_dict)


# ---------------------------------------------------------------------------
# Historical match slice + time-decay weights for one refit
# ---------------------------------------------------------------------------
def historical_slice(hist, cutoff, alpha_decay, lookback_years=LOOKBACK_YEARS):
    start = cutoff - pd.DateOffset(years=lookback_years)
    sub = hist[(hist['date'] >= start) & (hist['date'] < cutoff)].copy()
    days = (cutoff - sub['date']).dt.days.to_numpy(dtype=float)
    sub['weight'] = np.exp(-alpha_decay * days)
    return sub


# ---------------------------------------------------------------------------
# Grid dispatch helpers (kind -> grid function with the right signature)
# ---------------------------------------------------------------------------
def grid_for(kind, fit: StrengthFit, home_norm, away_norm, neutral, max_goals=MAX_GOALS):
    mu_h, mu_a = fit.mu(home_norm, away_norm, neutral)
    if kind == 'poisson_dc':
        return poisson_dc_grid(mu_h, mu_a, fit.extra['rho'], max_goals)
    if kind == 'nbinom_dc':
        return nbinom_dc_grid(mu_h, mu_a, fit.extra['r_h'], fit.extra['r_a'], fit.extra['rho'], max_goals)
    if kind == 'nbinom_frank':
        return nbinom_frank_grid(mu_h, mu_a, fit.extra['r_h'], fit.extra['r_a'], fit.extra['theta'], max_goals)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# M4 / M5: per-match calibration of (mu_home, mu_away) to devigged Kalshi odds,
# holding M3's shared (r_h, r_a, theta) fixed. "MLE-calibrated" here means:
# treat the market's devigged probabilities as soft targets and find the
# (mu_home, mu_away) that minimise cross-entropy between the model's own
# 1X2 (M4) / 1X2+O-U (M5) probabilities and those targets -- i.e. the market
# odds themselves are the "data" this per-match fit is estimated on, in place
# of the single realised scoreline a classic MLE would use.
# ---------------------------------------------------------------------------
def _calib_loss(log_mu, r_h, r_a, theta, mkt_1x2, ou, max_goals):
    mu_h, mu_a = np.exp(np.clip(log_mu, -6, 6))
    M = nbinom_frank_grid(mu_h, mu_a, r_h, r_a, theta, max_goals)
    p_h, p_d, p_a = grid_to_1x2(M)
    p_home_mkt, p_draw_mkt, p_away_mkt = mkt_1x2
    loss = -(p_home_mkt * np.log(max(p_h, 1e-12))
            + p_draw_mkt * np.log(max(p_d, 1e-12))
            + p_away_mkt * np.log(max(p_a, 1e-12)))
    if ou:
        for threshold, p_over_mkt in ou:
            p_over = np.clip(grid_to_over(M, threshold), 1e-12, 1 - 1e-12)
            loss += -(p_over_mkt * np.log(p_over) + (1 - p_over_mkt) * np.log(1 - p_over))
    return loss


def calibrate_mu_to_market(mu_h0, mu_a0, r_h, r_a, theta, mkt_1x2, ou=None, max_goals=CAL_GOALS):
    """Returns (mu_home, mu_away) minimising cross-entropy vs devigged market
    probabilities. Falls back to (mu_h0, mu_a0) -- the M3 model-implied rates --
    if no market prices are available or the optimiser fails to improve on them."""
    if mkt_1x2 is None:
        return mu_h0, mu_a0

    x0 = np.log([max(mu_h0, 1e-3), max(mu_a0, 1e-3)])
    args = (r_h, r_a, theta, mkt_1x2, ou, max_goals)
    result = minimize(_calib_loss, x0, args=args, method='L-BFGS-B',
                       bounds=[(-6, 6), (-6, 6)],
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    if not result.success or result.fun > _calib_loss(x0, *args):
        return mu_h0, mu_a0
    mu_h, mu_a = np.exp(np.clip(result.x, -6, 6))
    return float(mu_h), float(mu_a)


# ---------------------------------------------------------------------------
# Walk-forward loop: refit M1/M2/M3 once per unique kickoff date on all
# historical data strictly before that date (no leakage), then predict every
# match kicking off that day. M4/M5 reuse M3's (r_h, r_a, theta) per date and
# re-solve only (mu_home, mu_away) per match against that match's own odds.
# ---------------------------------------------------------------------------
def walk_forward_predict(hist, match_table, alpha_decay=ALPHA_DECAY,
                          lookback_years=LOOKBACK_YEARS, verbose=True):
    mt = match_table.copy()
    kickoff = mt['kickoff']
    if kickoff.dt.tz is not None:
        kickoff = kickoff.dt.tz_localize(None)
    mt['date'] = kickoff.dt.normalize()
    dates = sorted(mt['date'].unique())

    records = []
    for i, cutoff in enumerate(dates):
        cutoff = pd.Timestamp(cutoff)
        day = mt[mt['date'] == cutoff]
        sub = historical_slice(hist, cutoff, alpha_decay, lookback_years)

        fits = {kind: fit_strength_model(sub, kind, verbose=False) for kind in KINDS}
        if verbose:
            print(f"  [{i + 1:2d}/{len(dates)}] {cutoff.date()}  "
                  f"train_n={len(sub):5,d}  matches={len(day)}")

        f3 = fits['nbinom_frank']
        r_h, r_a, theta = f3.extra['r_h'], f3.extra['r_a'], f3.extra['theta']

        for _, row in day.iterrows():
            grids = {kind: grid_for(kind, fits[kind], row['home_norm'], row['away_norm'],
                                     row['neutral']) for kind in KINDS}

            mu_h0, mu_a0 = f3.mu(row['home_norm'], row['away_norm'], row['neutral'])

            mu_h4, mu_a4 = calibrate_mu_to_market(mu_h0, mu_a0, r_h, r_a, theta, row['mkt_1x2'])
            grids['m4'] = nbinom_frank_grid(mu_h4, mu_a4, r_h, r_a, theta, MAX_GOALS)

            mu_h5, mu_a5 = calibrate_mu_to_market(mu_h0, mu_a0, r_h, r_a, theta,
                                                   row['mkt_1x2'], ou=row['ou'])
            grids['m5'] = nbinom_frank_grid(mu_h5, mu_a5, r_h, r_a, theta, MAX_GOALS)

            rec = row.to_dict()
            rec['grids'] = grids
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Accuracy: RPS / log-loss / accuracy vs actual results, for every model plus
# the market's own devigged probabilities as a baseline.
# ---------------------------------------------------------------------------
def evaluate_predictions(records):
    scored = [r for r in records if r['actual_hg'] is not None and not pd.isna(r['actual_hg'])]
    if not scored:
        print("No matches with actual scores yet -- skipping accuracy metrics.")
        return None

    outcomes = actual_outcomes([r['actual_hg'] for r in scored], [r['actual_ag'] for r in scored])

    rows = []
    for key in MODEL_KEYS:
        probs = np.array([grid_to_1x2(r['grids'][key]) for r in scored])
        rps, _ = ranked_probability_score(probs, outcomes)
        rows.append(dict(model=MODEL_NAMES[key], n=len(scored), RPS=rps,
                          log_loss=nll_score(probs, outcomes),
                          accuracy=accuracy(probs, outcomes)))

    has_mkt = [i for i, r in enumerate(scored) if r['mkt_1x2'] is not None]
    if has_mkt:
        mkt_probs = np.array([scored[i]['mkt_1x2'] for i in has_mkt])
        mkt_outcomes = outcomes[has_mkt]
        rps, _ = ranked_probability_score(mkt_probs, mkt_outcomes)
        rows.append(dict(model='Kalshi market (devigged)', n=len(has_mkt), RPS=rps,
                          log_loss=nll_score(mkt_probs, mkt_outcomes),
                          accuracy=accuracy(mkt_probs, mkt_outcomes)))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PnL backtest: BUY/SELL vs devigged Kalshi 1X2 prices, $1/contract, for every
# model. BUY when model_p > implied_p + THRESHOLD, SELL when the reverse.
# ---------------------------------------------------------------------------
def backtest_1x2(records, threshold=THRESHOLD):
    bet_rows = []
    for r in records:
        if r['mkt_1x2'] is None or r['actual_hg'] is None or pd.isna(r['actual_hg']):
            continue
        hg, ag = r['actual_hg'], r['actual_ag']
        outcome_idx = 0 if hg > ag else (1 if hg == ag else 2)

        for key in MODEL_KEYS:
            model_p = grid_to_1x2(r['grids'][key])
            for oc_i, oc_name in enumerate(('home', 'draw', 'away')):
                imp = r['mkt_1x2'][oc_i]
                mdl = model_p[oc_i]
                diff = mdl - imp
                if abs(diff) <= threshold:
                    continue
                hit = float(oc_i == outcome_idx)
                action = 'BUY' if diff > 0 else 'SELL'
                pnl = (hit - imp) if action == 'BUY' else (imp - hit)
                bet_rows.append(dict(
                    model=MODEL_NAMES[key], match_id=r['match_id'],
                    match=f"{r['home_raw']} vs {r['away_raw']}", is_wc=r['is_wc'],
                    outcome=oc_name, action=action, model_p=round(mdl, 4),
                    implied=round(imp, 4), diff=round(diff, 4), hit=hit, pnl=round(pnl, 4),
                ))
    return pd.DataFrame(bet_rows)


def pnl_summary(bets, order=MODEL_KEYS):
    if bets.empty:
        print("  No bets generated.")
        return
    g = bets.groupby('model').agg(n_bets=('pnl', 'size'), pnl=('pnl', 'sum'),
                                   hit_rate=('hit', 'mean'))
    g['roi_pct'] = g['pnl'] / g['n_bets'] * 100
    g = g.reindex([MODEL_NAMES[k] for k in order if MODEL_NAMES[k] in g.index])
    print(g.to_string(float_format=lambda x: f"{x:,.3f}"))


# ---------------------------------------------------------------------------
# Exact-scoreline PnL backtest: BUY/SELL vs devigged Kalshi correct-score
# prices ($1/contract). WC-only -- kalshi_wc_exact_score_odds.csv has no
# friendlies coverage, so `records` here should be filtered to matches with
# a non-empty `score_odds` dict (checked below anyway).
# ---------------------------------------------------------------------------
def backtest_scorelines(records, model_keys=('nbinom_frank', 'm4', 'm5'), threshold=THRESHOLD):
    bet_rows = []
    for r in records:
        if not r['score_odds'] or r['actual_hg'] is None or pd.isna(r['actual_hg']):
            continue
        actual = (int(r['actual_hg']), int(r['actual_ag']))

        for key in model_keys:
            grid = r['grids'][key]
            n = grid.shape[0]
            for (i, j), (p_mkt, _decimal) in r['score_odds'].items():
                if i >= n or j >= n:
                    continue   # scoreline falls outside the model's truncation grid
                mdl = float(grid[i, j])
                diff = mdl - p_mkt
                if abs(diff) <= threshold:
                    continue
                hit = float((i, j) == actual)
                action = 'BUY' if diff > 0 else 'SELL'
                pnl = (hit - p_mkt) if action == 'BUY' else (p_mkt - hit)
                bet_rows.append(dict(
                    model=MODEL_NAMES[key], match_id=r['match_id'],
                    match=f"{r['home_raw']} vs {r['away_raw']}",
                    scoreline=f"{i}-{j}", actual=f"{actual[0]}-{actual[1]}",
                    action=action, model_p=round(mdl, 4), implied=round(p_mkt, 4),
                    diff=round(diff, 4), hit=hit, pnl=round(pnl, 4),
                ))
    return pd.DataFrame(bet_rows)


# ---------------------------------------------------------------------------
# Entry point -- python -m models  (or: python models.py)
# ---------------------------------------------------------------------------
def main(use_cache=True):
    import pickle
    SEP = '=' * 70
    print(SEP)
    print("  WC 2026 -- M1-M5 walk-forward backtest vs Kalshi")
    print(SEP)

    cache_path = ROOT / 'results' / 'models_records.pkl'
    if use_cache and cache_path.exists():
        print(f"\n[1-2] Loading cached walk-forward records from {cache_path.name} ...")
        with open(cache_path, 'rb') as f:
            records = pickle.load(f)
        print(f"    {len(records)} match records")
    else:
        print("\n[1] Loading historical corpus + odds.csv match table ...")
        hist = load_historical()
        match_table = load_match_table()
        n_dates = match_table['kickoff'].dt.normalize().nunique()
        print(f"    {len(match_table)} matches across {n_dates} unique kickoff dates")

        print(f"\n[2] Walk-forward fit  (lookback={LOOKBACK_YEARS}y, "
              f"decay={ALPHA_DECAY}/day) ...")
        records = walk_forward_predict(hist, match_table, verbose=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(records, f)
        print(f"    cached -> {cache_path}")

    print(f"\n{SEP}\n  ACCURACY  (matches with a known final score)\n{SEP}")
    acc = evaluate_predictions(records)
    if acc is not None:
        print(acc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n{SEP}\n  PNL BACKTEST  (threshold={THRESHOLD}, $1/contract, 1X2 market)\n{SEP}")
    bets = backtest_1x2(records)
    pnl_summary(bets)
    out = ROOT / 'results' / 'models_backtest_bets.csv'
    bets.to_csv(out, index=False)
    print(f"\n  Full bet log -> {out}")

    print(f"\n{SEP}\n  PNL BACKTEST  (threshold={THRESHOLD}, $1/contract, "
          f"exact-scoreline market, WC only, M3/M4/M5)\n{SEP}")
    sc_bets = backtest_scorelines(records)
    pnl_summary(sc_bets, order=('nbinom_frank', 'm4', 'm5'))
    sc_out = ROOT / 'results' / 'models_backtest_scoreline_bets.csv'
    sc_bets.to_csv(sc_out, index=False)
    print(f"\n  Full scoreline bet log -> {sc_out}")
    print(SEP)


if __name__ == '__main__':
    main()
