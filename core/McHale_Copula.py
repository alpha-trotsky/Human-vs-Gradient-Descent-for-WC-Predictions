"""
Negative Binomial marginals + Frank's copula football match model.

  log(μ_home) = X β_home      (same 20-dim features as Linear/MLP models)
  log(μ_away) = X β_away
  Home goals ~ NegBin(μ_home, r_home)   Away goals ~ NegBin(μ_away, r_away)
  Joint PMF:  P(H=h, A=a) = ΔΔ C( F_home(h), F_away(a) ; k )
              C = Frank's copula

k may be constant or a function of the standardised quality gap x_i:
  x_i = (home_elo − away_elo) + (home_squad_avg − away_squad_avg)  [std-scaled]

  k_mode='const' : k = a                          (1 extra param)
  k_mode='log'   : k = a + b·log(1+|x|)           (2 extra params)
  k_mode='exp'   : k = a + b·(exp(|x|)−1)         (2 extra params)
  k_mode='quad'  : k = a + b·x + c·x²             (3 extra params)

Parameters: β_home (n+1), β_away (n+1), log_r_home, log_r_away, k_params...
Fitted by L-BFGS-B MLE + ridge α on slope coefficients.

Run: python -m core.McHale_Copula
"""
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import nbinom

from .evaluate import (
    ranked_probability_score, log_loss, brier_score, accuracy, actual_outcomes,
)

_N_K_PARAMS = {'const': 1, 'log': 2, 'exp': 2, 'quad': 3}

# Column indices in FULL_FEATURES (20-dim vector) for the quality-gap feature.
_IDX_HOME_ELO, _IDX_AWAY_ELO     = 0, 1
_IDX_HOME_SQUAD, _IDX_AWAY_SQUAD = 6, 13


# ---------------------------------------------------------------------------
# Frank's copula — scalar or array k
# ---------------------------------------------------------------------------
def frank_copula(u, v, k):
    """
    Frank copula C(u, v; k). Broadcasts over u, v; k may be scalar or array.
    k > 0: positive dependence; k < 0: negative; k = 0: independence (u·v).
    """
    u = np.clip(u, 1e-12, 1 - 1e-12)
    v = np.clip(v, 1e-12, 1 - 1e-12)
    k = np.asarray(k, dtype=float)
    small  = np.abs(k) < 1e-8
    k_safe = np.where(small, 1.0, k)          # avoid /0; result masked below
    num    = np.expm1(-k_safe * u) * np.expm1(-k_safe * v)
    den    = np.expm1(-k_safe)
    ratio  = np.clip(num / den, -1 + 1e-12, None)
    return np.where(small, u * v, -np.log1p(ratio) / k_safe)


# ---------------------------------------------------------------------------
# Joint PMF and scoreline matrix
# ---------------------------------------------------------------------------
def joint_pmf(h, a, mu_h, mu_a, r_h, r_a, k):
    """
    P(H=h_i, A=a_i) for a batch of matches. Vectorised over matches.
    h, a: int arrays; mu_h, mu_a: float arrays; r_h, r_a: scalars; k: scalar or (n,) array.
    """
    ph = r_h / (r_h + mu_h)
    pa = r_a / (r_a + mu_a)

    F_h  = nbinom.cdf(h,     r_h, ph)
    F_hm = np.where(h > 0, nbinom.cdf(h - 1, r_h, ph), 0.0)
    F_a  = nbinom.cdf(a,     r_a, pa)
    F_am = np.where(a > 0, nbinom.cdf(a - 1, r_a, pa), 0.0)

    p = (frank_copula(F_h,  F_a,  k)
       - frank_copula(F_hm, F_a,  k)
       - frank_copula(F_h,  F_am, k)
       + frank_copula(F_hm, F_am, k))
    return np.clip(p, 1e-10, None)


def score_matrix(mu_h, mu_a, r_h, r_a, k, max_goals=10):
    """
    Full joint PMF grid M[i,j] = P(H=i, A=j) for a single match. k must be scalar.
    """
    g  = np.arange(max_goals + 1)
    ph = r_h / (r_h + mu_h)
    pa = r_a / (r_a + mu_a)

    F_h  = nbinom.cdf(g, r_h, ph)
    F_hm = np.concatenate([[0.0], F_h[:-1]])
    F_a  = nbinom.cdf(g, r_a, pa)
    F_am = np.concatenate([[0.0], F_a[:-1]])

    M = (frank_copula(F_h[:, None],  F_a[None, :],  k)
       - frank_copula(F_hm[:, None], F_a[None, :],  k)
       - frank_copula(F_h[:, None],  F_am[None, :], k)
       + frank_copula(F_hm[:, None], F_am[None, :], k))
    M = np.clip(M, 0.0, None)
    M /= M.sum()
    return M


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class McHaleCopulaModel:
    """
    NB marginals + Frank's copula, fitted by ridge-penalised MLE (L-BFGS-B).

    k_mode controls how the copula parameter varies per match:
      'const' : k = a
      'log'   : k = a + b·log(1+|x|)
      'exp'   : k = a + b·(exp(|x|)−1)
      'quad'  : k = a + b·x + c·x²
    where x = standardised (home_elo−away_elo) + (home_squad_avg−away_squad_avg).

    Parameter vector layout:
      [ β_home (n+1) | β_away (n+1) | log_r_home | log_r_away | k_params... ]
    """

    def __init__(self, n_features, k_mode='const'):
        assert k_mode in _N_K_PARAMS, f"k_mode must be one of {list(_N_K_PARAMS)}"
        self.n_features  = n_features
        self.k_mode      = k_mode
        self.n_k_params  = _N_K_PARAMS[k_mode]
        self.coefficients = None
        self.feature_mean = None
        self.feature_std  = None

    # ------------------------------------------------------------------
    def _standardize(self, X):
        if self.feature_mean is not None:
            return (X - self.feature_mean) / self.feature_std
        return X

    def _quality_gap(self, X):
        """Standardised (home − away) quality: elo_diff + squad_avg_diff."""
        Xs  = self._standardize(X)
        gap = Xs[:, _IDX_HOME_ELO] - Xs[:, _IDX_AWAY_ELO]
        if self.n_features > _IDX_AWAY_SQUAD:
            gap = gap + Xs[:, _IDX_HOME_SQUAD] - Xs[:, _IDX_AWAY_SQUAD]
        return gap

    def _compute_k(self, X, k_params):
        """Per-match copula parameter from quality-gap and k_params."""
        p = k_params
        if self.k_mode == 'const':
            return float(np.clip(p[0], -15, 15))
        x = self._quality_gap(X)
        if self.k_mode == 'log':
            k = p[0] + p[1] * np.log1p(np.abs(x))
        elif self.k_mode == 'exp':
            k = p[0] + p[1] * np.expm1(np.clip(np.abs(x), 0, 5))
        elif self.k_mode == 'quad':
            k = p[0] + p[1] * x + p[2] * x ** 2
        return np.clip(k, -15, 15)

    def _predict_mu(self, X, params):
        n      = self.n_features
        beta_h = params[:n + 1]
        beta_a = params[n + 1 : 2 * (n + 1)]
        Xs     = self._standardize(X)
        X1     = np.column_stack([np.ones(len(Xs)), Xs])
        mu_h   = np.exp(np.clip(X1 @ beta_h, -10, 10))
        mu_a   = np.exp(np.clip(X1 @ beta_a, -10, 10))
        return mu_h, mu_a

    def _unpack_extras(self, params):
        base     = 2 * (self.n_features + 1)
        r_h      = np.exp(np.clip(params[base],     -5, 5))
        r_a      = np.exp(np.clip(params[base + 1], -5, 5))
        k_params = params[base + 2:]
        return r_h, r_a, k_params

    # ------------------------------------------------------------------
    def _nll(self, params, X, hg, ag, weights, alpha):
        mu_h, mu_a         = self._predict_mu(X, params)
        r_h, r_a, k_params = self._unpack_extras(params)
        k   = self._compute_k(X, k_params)
        lp  = np.log(joint_pmf(hg, ag, mu_h, mu_a, r_h, r_a, k))
        nll = -(weights * lp).sum() if weights is not None else -lp.sum()
        if alpha > 0:
            n      = self.n_features
            slopes = np.concatenate([params[1 : n + 1], params[n + 2 : 2 * (n + 1)]])
            nll   += alpha * np.dot(slopes, slopes)
        return nll

    # ------------------------------------------------------------------
    def fit(self, X, hg, ag, weights=None, alpha=0.0, max_iter=500, verbose=False):
        self.feature_mean = X.mean(axis=0)
        self.feature_std  = X.std(axis=0) + 1e-8

        n       = self.n_features
        base    = 2 * (n + 1)
        n_total = base + 2 + self.n_k_params

        x0          = np.zeros(n_total)
        x0[0]       = np.log(max(float(hg.mean()), 0.1))   # home intercept
        x0[n + 1]   = np.log(max(float(ag.mean()), 0.1))   # away intercept
        x0[base]    = 1.0    # log r_home → r ≈ 2.7
        x0[base+1]  = 1.0    # log r_away
        # k_params initialised to 0 (k=0 everywhere, independence)

        bounds = (
            [(-10, 10)] * (2 * (n + 1))
            + [(-5, 5), (-5, 5)]
            + [(-20, 20)] * self.n_k_params
        )

        result = minimize(
            self._nll, x0,
            args=(X, hg, ag, weights, alpha),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6},
        )
        self.coefficients = result.x
        if verbose:
            r_h, r_a, kp = self._unpack_extras(result.x)
            kp_str = '  '.join(f'{v:+.4f}' for v in kp)
            print(f"  α={alpha:5.1f}  NLL={result.fun:.1f}  "
                  f"r_h={r_h:.3f}  r_a={r_a:.3f}  k=[{kp_str}]  [{result.message}]")
        return result

    # ------------------------------------------------------------------
    def predict(self, X):
        """Return (mu_home, mu_away). Mirrors LinearRegressionDixonColes.predict()."""
        if self.coefficients is None:
            raise ValueError("Model not fitted")
        return self._predict_mu(X, self.coefficients)

    def predict_wdl(self, X, max_goals=10):
        """Return (n, 3) array of [P(home win), P(draw), P(away win)] per match."""
        mu_h, mu_a         = self.predict(X)
        r_h, r_a, k_params = self._unpack_extras(self.coefficients)
        k_arr = self._compute_k(X, k_params)
        probs = np.zeros((len(mu_h), 3))
        for i in range(len(mu_h)):
            k_i = float(k_arr) if np.ndim(k_arr) == 0 else float(k_arr[i])
            M = score_matrix(float(mu_h[i]), float(mu_a[i]), r_h, r_a, k_i, max_goals)
            probs[i, 0] = np.tril(M, -1).sum()
            probs[i, 1] = np.trace(M)
            probs[i, 2] = np.triu(M,  1).sum()
        return probs


# ---------------------------------------------------------------------------
# Alpha sweep
# ---------------------------------------------------------------------------
def alpha_sweep(X_tr, hg_tr, ag_tr, w_tr,
                X_val, hg_val, ag_val,
                alphas=(0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0),
                k_mode='const', max_iter=500):
    """Fit for each alpha; report val RPS. Returns DataFrame sorted by val RPS."""
    import pandas as pd
    out_val = actual_outcomes(hg_val, ag_val)

    print(f"\n  [k_mode={k_mode}]")
    print(f"  {'alpha':>6}  {'val_RPS':>9}  {'val_acc':>8}  {'r_home':>7}  {'r_away':>7}  k_params")
    print("  " + "-" * 68)

    rows = []
    for alpha in alphas:
        m   = McHaleCopulaModel(X_tr.shape[1], k_mode=k_mode)
        res = m.fit(X_tr, hg_tr, ag_tr, weights=w_tr,
                    alpha=float(alpha), max_iter=max_iter, verbose=False)
        probs  = m.predict_wdl(X_val)
        rps, _ = ranked_probability_score(probs, out_val)
        acc    = accuracy(probs, out_val)
        r_h, r_a, kp = m._unpack_extras(m.coefficients)
        kp_str = '  '.join(f'{v:+.4f}' for v in kp)
        print(f"  {alpha:6.1f}  {rps:9.4f}  {acc:8.3f}  {r_h:7.3f}  {r_a:7.3f}  [{kp_str}]")
        rows.append({'alpha': alpha, 'k_mode': k_mode, 'val_rps': rps, 'val_acc': acc,
                     'r_home': r_h, 'r_away': r_a, 'converged': res.success,
                     **{f'k{i}': v for i, v in enumerate(kp)}})

    df   = pd.DataFrame(rows)
    best = df.loc[df['val_rps'].idxmin()]
    print(f"\n  Best alpha = {best['alpha']:.1f}  (val RPS = {best['val_rps']:.4f})")
    return df


# ---------------------------------------------------------------------------
# Mode comparison (log / exp / quad vs const baseline)
# ---------------------------------------------------------------------------
def mode_comparison(X_tr, hg_tr, ag_tr, w_tr,
                    X_val, hg_val, ag_val,
                    X_te,  hg_te,  ag_te,
                    alphas=(0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0),
                    max_iter=500):
    """
    For each k_mode: find best alpha on val, then evaluate on test.
    Prints a summary table. Returns DataFrame.
    """
    import pandas as pd
    out_te  = actual_outcomes(hg_te, ag_te)
    summary = []

    for mode in ('const', 'log', 'exp', 'quad'):
        print(f"\n{'='*62}\n  k_mode = {mode}\n{'='*62}")
        sweep      = alpha_sweep(X_tr, hg_tr, ag_tr, w_tr,
                                 X_val, hg_val, ag_val,
                                 alphas=alphas, k_mode=mode, max_iter=max_iter)
        best_alpha = float(sweep.loc[sweep['val_rps'].idxmin(), 'alpha'])
        best_val   = float(sweep['val_rps'].min())

        m = McHaleCopulaModel(X_tr.shape[1], k_mode=mode)
        m.fit(X_tr, hg_tr, ag_tr, weights=w_tr,
              alpha=best_alpha, max_iter=max_iter, verbose=True)

        probs_te  = m.predict_wdl(X_te)
        rps_te, _ = ranked_probability_score(probs_te, out_te)
        acc_te    = accuracy(probs_te, out_te)
        r_h, r_a, kp = m._unpack_extras(m.coefficients)
        summary.append({'k_mode': mode, 'best_alpha': best_alpha,
                        'val_rps': best_val, 'test_rps': rps_te, 'test_acc': acc_te,
                        'r_home': r_h, 'r_away': r_a, 'k_params': list(kp)})

    df = pd.DataFrame(summary).sort_values('test_rps')
    print(f"\n{'='*62}\n  MODE COMPARISON SUMMARY\n{'='*62}")
    print(f"  {'mode':8}  {'best_α':>7}  {'val_RPS':>9}  {'test_RPS':>9}  {'acc':>6}  k_params")
    print("  " + "-" * 68)
    for _, r in df.iterrows():
        kp_str = '  '.join(f'{v:+.4f}' for v in r['k_params'])
        print(f"  {r['k_mode']:8}  {r['best_alpha']:7.1f}  {r['val_rps']:9.4f}  "
              f"{r['test_rps']:9.4f}  {r['test_acc']:6.3f}  [{kp_str}]")
    print("\n  (compare: linear RPS=0.1762, MLP RPS=0.1756 on the same test set)")
    return df


# ---------------------------------------------------------------------------
# Entry point — run as: python -m core.McHale_Copula
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pandas as pd
    from core.experiments import load_split, build_X, FULL_FEATURES, TRAIN_END, VAL_END

    print("Loading train / val / test splits...")
    df_tr,  hg_tr,  ag_tr,  w_tr = load_split(end=TRAIN_END)
    df_val, hg_val, ag_val, _    = load_split(start=TRAIN_END, end=VAL_END)
    df_te,  hg_te,  ag_te,  _    = load_split(start=VAL_END)

    X_tr  = build_X(df_tr,  FULL_FEATURES)
    X_val = build_X(df_val, FULL_FEATURES)
    X_te  = build_X(df_te,  FULL_FEATURES)
    print(f"  train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    mode_comparison(
        X_tr,  hg_tr,  ag_tr,  w_tr,
        X_val, hg_val, ag_val,
        X_te,  hg_te,  ag_te,
        alphas=[0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0],
    )
