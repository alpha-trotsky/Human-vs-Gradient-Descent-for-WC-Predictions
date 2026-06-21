"""
Experiment harness: model selection on a leak-safe TRAIN / VALIDATION / TEST split.

Split (by match date):
    TRAIN       : date <  2024-01-01
    VALIDATION  : 2024-01-01 <= date < 2025-10-01   (model + hyperparameter selection)
    TEST        : date >= 2025-10-01                 (end of 2025 + 2026; scored ONCE)

All hyperparameter choices are made on VALIDATION only. The TEST set is touched
exactly once at the very end, for the single best linear and best MLP, so the
reported test numbers are an honest estimate of out-of-sample performance.

Runs:
  1. Isolation / convergence  (decay 0.001, MLP gets temporal weights, long training)
  2. MLP hyperparameter sweep (lr, depth, width, dropout) + gradient-clip ablation
  3. Linear L2 (ridge) sweep over 5 alphas
  4. Reduced feature set (best linear + best MLP)
  5. Friendlies stripped from train+val (best linear + best MLP)
  6. Final TEST evaluation of the overall best models

Outputs: experiments_report.tex (LaTeX) and results/experiment_results.csv
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from .training import prepare_training_data, LinearRegressionDixonColes, dixon_coles_loss
from .evaluate import (outcome_probs_from_lambdas, actual_outcomes, predict_linear, predict_mlp,
                       ranked_probability_score, log_loss, brier_score, accuracy, base_rate_probs)

base_dir = Path(__file__).resolve().parent.parent
results_path = base_dir / 'results' / 'match_results_elo.csv'
squad_path = base_dir / 'results' / 'squad_features.csv'

WEIGHT_DECAY = 0.001
TRAIN_END = '2024-01-01'
VAL_END = '2025-10-01'

FULL_FEATURES = [
    'home_elo', 'away_elo',
    'home_scored_last10', 'home_conceded_last10', 'away_scored_last10', 'away_conceded_last10',
    'home_squad_avg', 'home_squad_std', 'home_attack_avg', 'home_midfield_avg', 'home_defence_avg',
    'home_bench_avg', 'home_bench_std',
    'away_squad_avg', 'away_squad_std', 'away_attack_avg', 'away_midfield_avg', 'away_defence_avg',
    'away_bench_avg', 'away_bench_std'
]
# Reduced set requested: elo, bench avg, per-match goal difference (last 10),
# and attack/defence squad averages — home & away each.
REDUCED_FEATURES = [
    'home_elo', 'away_elo',
    'home_bench_avg', 'away_bench_avg',
    'home_goaldiff_pm', 'away_goaldiff_pm',
    'home_attack_avg', 'home_defence_avg', 'away_attack_avg', 'away_defence_avg',
]


# ---------------------------------------------------------------------------
# Data loading / feature building
# ---------------------------------------------------------------------------
def add_goaldiff(df):
    """Per-match goal difference over the last 10 games (scored - conceded) / 10."""
    df = df.copy()
    df['home_goaldiff_pm'] = (df['home_scored_last10'] - df['home_conceded_last10']) / 10.0
    df['away_goaldiff_pm'] = (df['away_scored_last10'] - df['away_conceded_last10']) / 10.0
    return df


def build_X(df, feature_cols):
    df = add_goaldiff(df)
    for c in feature_cols:
        df[c] = df[c].fillna(0)
    return df[feature_cols].values.astype(float)


def load_split(start=None, end=None, drop_friendlies=False):
    """Return (df, home_goals, away_goals, weights) for one date window."""
    _, hg, ag, w, _, df = prepare_training_data(
        results_path, squad_path, weight_decay=WEIGHT_DECAY,
        start_date=start, end_date=end, drop_friendlies=drop_friendlies, verbose=False)
    return df.reset_index(drop=True), hg, ag, w


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def eval_probs(probs, hg, ag):
    out = actual_outcomes(hg, ag)
    rps, _ = ranked_probability_score(probs, out)
    return {'RPS': rps, 'log_loss': log_loss(probs, out),
            'Brier': brier_score(probs, out), 'acc': accuracy(probs, out)}


# ---------------------------------------------------------------------------
# Linear model
# ---------------------------------------------------------------------------
def run_linear(Xtr, hgtr, agtr, wtr, Xev, hgev, agev, alpha):
    m = LinearRegressionDixonColes(Xtr.shape[1])
    m.fit(Xtr, hgtr, agtr, weights=wtr, alpha=alpha, verbose=False, max_iter=500)
    probs = outcome_probs_from_lambdas(*predict_linear(m, Xev))
    return m, eval_probs(probs, hgev, agev)


# ---------------------------------------------------------------------------
# Flexible MLP
# ---------------------------------------------------------------------------
class FlexibleMLP(nn.Module):
    def __init__(self, input_size, width, depth, dropout):
        super().__init__()
        layers, d = [], input_size
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 3)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.body(x)
        out = self.head(h)
        lam_h = self.softplus(out[:, 0]) + 0.1
        lam_a = self.softplus(out[:, 1]) + 0.1
        rho = self.tanh(out[:, 2])
        return lam_h, lam_a, rho


def train_mlp(Xtr, hgtr, agtr, wtr, Xev, hgev, agev, *, width=128, depth=2, dropout=0.2,
              lr=0.001, grad_clip=None, max_epochs=1500, patience=150, batch_size=256, seed=0):
    """Train one MLP; early-stop on validation NLL. Returns (model, info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-8
    Xtr_s = (Xtr - mean) / std
    Xev_s = (Xev - mean) / std

    ds = TensorDataset(torch.FloatTensor(Xtr_s), torch.FloatTensor(hgtr),
                       torch.FloatTensor(agtr), torch.FloatTensor(np.asarray(wtr)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    Xev_t = torch.FloatTensor(Xev_s)
    hgev_t = torch.FloatTensor(hgev)
    agev_t = torch.FloatTensor(agev)

    model = FlexibleMLP(Xtr.shape[1], width, depth, dropout)
    model.feature_mean = mean
    model.feature_std = std
    opt = optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, best_epoch, wait = float('inf'), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, hg, ag, wb in loader:
            lam_h, lam_a, rho = model(xb)
            loss = dixon_coles_loss(lam_h, lam_a, rho, hg, ag, weights=wb)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        model.eval()
        with torch.no_grad():
            lam_h, lam_a, rho = model(Xev_t)
            vloss = dixon_coles_loss(lam_h, lam_a, rho, hgev_t, agev_t).item()
        if vloss < best_val - 1e-5:
            best_val, best_epoch, wait = vloss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {'best_val_nll': best_val, 'best_epoch': best_epoch + 1, 'epochs_ran': epoch + 1}
    return model, info


def run_mlp(Xtr, hgtr, agtr, wtr, Xev, hgev, agev, **kw):
    model, info = train_mlp(Xtr, hgtr, agtr, wtr, Xev, hgev, agev, **kw)
    probs = outcome_probs_from_lambdas(*predict_mlp(model, Xev))
    metrics = eval_probs(probs, hgev, agev)
    return model, metrics, info


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t0 = time.time()
    rows = []   # flat list of result dicts for CSV + LaTeX

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    # ---- Load splits (full features) ----
    log("Loading splits...")
    dftr, hgtr, agtr, wtr = load_split(end=TRAIN_END)
    dfval, hgval, agval, wval = load_split(start=TRAIN_END, end=VAL_END)
    dfte, hgte, agte, wte = load_split(start=VAL_END)

    Xtr = build_X(dftr, FULL_FEATURES)
    Xval = build_X(dfval, FULL_FEATURES)
    Xte = build_X(dfte, FULL_FEATURES)

    split_info = {
        'train': (len(dftr), str(dftr['date'].min().date()), str(dftr['date'].max().date())),
        'val':   (len(dfval), str(dfval['date'].min().date()), str(dfval['date'].max().date())),
        'test':  (len(dfte), str(dfte['date'].min().date()), str(dfte['date'].max().date())),
    }
    log(f"Split sizes: train={len(dftr)}, val={len(dfval)}, test={len(dfte)}")

    # Base rate baseline (validation)
    base_probs = base_rate_probs(hgtr, agtr, len(hgval))
    base_metrics = eval_probs(base_probs, hgval, agval)
    rows.append({'exp': '1_isolation', 'model': 'Base rate', 'config': '-', 'split': 'val', **base_metrics})

    # ---- Exp 1: isolation / convergence ----
    log("Exp 1: linear baseline (alpha=0) + MLP with weights, long training...")
    lin0, lin0_m = run_linear(Xtr, hgtr, agtr, wtr, Xval, hgval, agval, alpha=0.0)
    rows.append({'exp': '1_isolation', 'model': 'Linear (alpha=0)', 'config': 'full feats', 'split': 'val', **lin0_m})

    mlp_iso, mlp_iso_m, mlp_iso_info = run_mlp(
        Xtr, hgtr, agtr, wtr, Xval, hgval, agval,
        width=128, depth=2, dropout=0.2, lr=0.001, grad_clip=None,
        max_epochs=4000, patience=400)
    rows.append({'exp': '1_isolation', 'model': 'MLP (weights, long)',
                 'config': f"w128 d2 do0.2 lr1e-3 | best@{mlp_iso_info['best_epoch']}/{mlp_iso_info['epochs_ran']}",
                 'split': 'val', **mlp_iso_m})
    log(f"  Linear RPS={lin0_m['RPS']:.4f} | MLP RPS={mlp_iso_m['RPS']:.4f} "
        f"(converged best@{mlp_iso_info['best_epoch']}, ran {mlp_iso_info['epochs_ran']})")

    # ---- Exp 2: MLP hyperparameter sweep ----
    log("Exp 2: MLP sweep (lr x depth x width x dropout, grad_clip=5)...")
    sweep_rows = []
    best_mlp_cfg, best_mlp_rps = None, float('inf')
    for lr in (0.0005, 0.001, 0.003):
        for depth in (1, 2, 3):
            for width in (64, 128, 256):
                for dropout in (0.0, 0.2):
                    cfg = dict(lr=lr, depth=depth, width=width, dropout=dropout, grad_clip=5.0)
                    _, m, info = run_mlp(Xtr, hgtr, agtr, wtr, Xval, hgval, agval,
                                         max_epochs=1500, patience=150, **cfg)
                    label = f"lr{lr} d{depth} w{width} do{dropout}"
                    sweep_rows.append({'exp': '2_mlp_sweep', 'model': 'MLP', 'config': label,
                                       'split': 'val', **m})
                    if m['RPS'] < best_mlp_rps:
                        best_mlp_rps, best_mlp_cfg = m['RPS'], cfg
    rows.extend(sweep_rows)
    log(f"  Best MLP config: {best_mlp_cfg} (val RPS={best_mlp_rps:.4f})")

    # Gradient-clip ablation: best config with clip OFF
    cfg_noclip = dict(best_mlp_cfg)
    cfg_noclip['grad_clip'] = None
    _, m_noclip, _ = run_mlp(Xtr, hgtr, agtr, wtr, Xval, hgval, agval,
                             max_epochs=1500, patience=150, **cfg_noclip)
    rows.append({'exp': '2_mlp_sweep', 'model': 'MLP (best, clip OFF)',
                 'config': f"lr{cfg_noclip['lr']} d{cfg_noclip['depth']} w{cfg_noclip['width']} do{cfg_noclip['dropout']}",
                 'split': 'val', **m_noclip})

    # ---- Exp 3: Linear L2 (ridge) sweep ----
    log("Exp 3: Linear ridge sweep over 5 alphas...")
    best_alpha, best_alpha_rps = 0.0, float('inf')
    for alpha in (0.0, 10.0, 100.0, 1000.0, 10000.0):
        _, m = run_linear(Xtr, hgtr, agtr, wtr, Xval, hgval, agval, alpha=alpha)
        rows.append({'exp': '3_linear_ridge', 'model': 'Linear', 'config': f"alpha={alpha:g}",
                     'split': 'val', **m})
        if m['RPS'] < best_alpha_rps:
            best_alpha_rps, best_alpha = m['RPS'], alpha
        log(f"  alpha={alpha:<8g} RPS={m['RPS']:.4f}")
    log(f"  Best alpha={best_alpha} (val RPS={best_alpha_rps:.4f})")

    # ---- Exp 4: Reduced feature set ----
    log("Exp 4: reduced feature set (best alpha + best MLP)...")
    Xtr_r = build_X(dftr, REDUCED_FEATURES)
    Xval_r = build_X(dfval, REDUCED_FEATURES)
    Xte_r = build_X(dfte, REDUCED_FEATURES)

    _, lin_red_m = run_linear(Xtr_r, hgtr, agtr, wtr, Xval_r, hgval, agval, alpha=best_alpha)
    rows.append({'exp': '4_reduced', 'model': f'Linear (alpha={best_alpha:g})',
                 'config': f'{len(REDUCED_FEATURES)} feats', 'split': 'val', **lin_red_m})

    _, mlp_red_m, _ = run_mlp(Xtr_r, hgtr, agtr, wtr, Xval_r, hgval, agval,
                              max_epochs=1500, patience=150, **best_mlp_cfg)
    rows.append({'exp': '4_reduced', 'model': 'MLP (best cfg)',
                 'config': f'{len(REDUCED_FEATURES)} feats', 'split': 'val', **mlp_red_m})

    # ---- Exp 5: Strip friendlies from train + val ----
    log("Exp 5: friendlies stripped from train + val...")
    dftr_nf, hgtr_nf, agtr_nf, wtr_nf = load_split(end=TRAIN_END, drop_friendlies=True)
    dfval_nf, hgval_nf, agval_nf, _ = load_split(start=TRAIN_END, end=VAL_END, drop_friendlies=True)
    Xtr_nf = build_X(dftr_nf, FULL_FEATURES)
    Xval_nf = build_X(dfval_nf, FULL_FEATURES)

    _, lin_nf_m = run_linear(Xtr_nf, hgtr_nf, agtr_nf, wtr_nf, Xval_nf, hgval_nf, agval_nf, alpha=best_alpha)
    rows.append({'exp': '5_no_friendlies', 'model': f'Linear (alpha={best_alpha:g})',
                 'config': f'no friendlies (n_tr={len(dftr_nf)})', 'split': 'val', **lin_nf_m})

    _, mlp_nf_m, _ = run_mlp(Xtr_nf, hgtr_nf, agtr_nf, wtr_nf, Xval_nf, hgval_nf, agval_nf,
                             max_epochs=1500, patience=150, **best_mlp_cfg)
    rows.append({'exp': '5_no_friendlies', 'model': 'MLP (best cfg)',
                 'config': f'no friendlies (n_tr={len(dftr_nf)})', 'split': 'val', **mlp_nf_m})

    # ---- Final: TEST evaluation of overall best models ----
    # Best linear = best alpha on full features (refit on train), best MLP = best sweep cfg.
    log("FINAL: evaluating best linear + best MLP on TEST set (scored once)...")
    base_te = eval_probs(base_rate_probs(hgtr, agtr, len(hgte)), hgte, agte)
    rows.append({'exp': '6_test', 'model': 'Base rate', 'config': '-', 'split': 'test', **base_te})

    lin_best, _ = run_linear(Xtr, hgtr, agtr, wtr, Xval, hgval, agval, alpha=best_alpha)
    lin_te_probs = outcome_probs_from_lambdas(*predict_linear(lin_best, Xte))
    lin_te = eval_probs(lin_te_probs, hgte, agte)
    rows.append({'exp': '6_test', 'model': f'Linear (alpha={best_alpha:g})', 'config': 'full feats',
                 'split': 'test', **lin_te})

    mlp_best, _ = train_mlp(Xtr, hgtr, agtr, wtr, Xval, hgval, agval,
                            max_epochs=1500, patience=150, **best_mlp_cfg)
    mlp_te_probs = outcome_probs_from_lambdas(*predict_mlp(mlp_best, Xte))
    mlp_te = eval_probs(mlp_te_probs, hgte, agte)
    rows.append({'exp': '6_test', 'model': 'MLP (best cfg)',
                 'config': str(best_mlp_cfg), 'split': 'test', **mlp_te})
    log(f"  TEST: base RPS={base_te['RPS']:.4f} | linear RPS={lin_te['RPS']:.4f} | MLP RPS={mlp_te['RPS']:.4f}")

    # ---- Save CSV + LaTeX ----
    res = pd.DataFrame(rows)
    csv_path = base_dir / 'results' / 'experiment_results.csv'
    res.to_csv(csv_path, index=False)
    log(f"Saved {csv_path}")

    write_latex(base_dir / 'experiments_report.tex', res, split_info,
                best_mlp_cfg, best_mlp_rps, best_alpha, mlp_iso_info)
    log("Saved experiments_report.tex")
    log(f"DONE in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# LaTeX report
# ---------------------------------------------------------------------------
def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v).replace('_', r'\_').replace('&', r'\&')


def _table(df, caption):
    cols = ['model', 'config', 'RPS', 'log_loss', 'Brier', 'acc']
    out = [r"\begin{table}[h]", r"\centering", r"\small",
           r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Model & Config & RPS & LogLoss & Brier & Acc \\", r"\midrule"]
    # best (lowest) RPS in this block highlighted
    best_idx = df['RPS'].astype(float).idxmin()
    for i, r in df.iterrows():
        line = f"{_fmt(r['model'])} & {_fmt(r['config'])} & {_fmt(r['RPS'])} & " \
               f"{_fmt(r['log_loss'])} & {_fmt(r['Brier'])} & {_fmt(r['acc'])} \\\\"
        if i == best_idx:
            line = r"\textbf{" + line.replace(' & ', r'} & \textbf{').rstrip(r'\\') + r"} \\"
        out.append(line)
    out += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{{caption}}}", r"\end{table}", ""]
    return "\n".join(out)


def write_latex(path, res, split_info, best_mlp_cfg, best_mlp_rps, best_alpha, mlp_iso_info):
    L = []
    L.append(r"\documentclass{article}")
    L.append(r"\usepackage{booktabs}")
    L.append(r"\usepackage[margin=1in]{geometry}")
    L.append(r"\title{World Cup Match Model: Linear vs.\ MLP Dixon-Coles}")
    L.append(r"\author{Experiment harness}")
    L.append(r"\date{\today}")
    L.append(r"\begin{document}")
    L.append(r"\maketitle")

    L.append(r"\section{Setup}")
    L.append("Leak-safe split by match date. Hyperparameters selected on VALIDATION; "
             "TEST scored once. Metrics: RPS (Ranked Probability Score, ordinal 1X2, lower better), "
             "multiclass log-loss, multiclass Brier, and argmax accuracy.")
    L.append("")
    L.append(r"\begin{itemize}")
    for name, (n, a, b) in split_info.items():
        L.append(rf"\item \textbf{{{name}}}: {n} matches, {a} to {b}")
    L.append(r"\end{itemize}")
    L.append(rf"Temporal decay $\lambda={WEIGHT_DECAY}$ (applied to both models). "
             rf"Best MLP config: \texttt{{{_fmt(str(best_mlp_cfg))}}} (val RPS {best_mlp_rps:.4f}). "
             rf"Best ridge $\alpha={best_alpha:g}$. "
             rf"Isolation MLP converged at epoch {mlp_iso_info['best_epoch']} "
             rf"(ran {mlp_iso_info['epochs_ran']}).")

    sections = [
        ('1_isolation', 'Experiment 1: Isolation / convergence (validation)'),
        ('3_linear_ridge', 'Experiment 3: Linear ridge $\\alpha$ sweep (validation)'),
        ('4_reduced', 'Experiment 4: Reduced 10-feature set (validation)'),
        ('5_no_friendlies', 'Experiment 5: Friendlies removed from train+val (validation)'),
        ('6_test', 'Final: TEST set (end 2025 + 2026), scored once'),
    ]
    for exp, title in sections:
        sub = res[res['exp'] == exp].reset_index(drop=True)
        if len(sub):
            L.append(rf"\section*{{{title}}}")
            L.append(_table(sub, title))

    # MLP sweep: show top 10 by RPS
    sweep = res[res['exp'] == '2_mlp_sweep'].reset_index(drop=True)
    if len(sweep):
        top = sweep.sort_values('RPS').head(12).reset_index(drop=True)
        L.append(r"\section*{Experiment 2: MLP hyperparameter sweep (validation, top 12 of "
                 + str(len(sweep)) + r" configs)}")
        L.append(_table(top, "MLP sweep --- best 12 configurations by RPS"))

    L.append(r"\end{document}")
    Path(path).write_text("\n".join(L), encoding='utf-8')


if __name__ == "__main__":
    main()
