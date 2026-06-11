"""
Regenerate the LaTeX report from results/experiment_results.csv (+ results/wc_odds.csv).

Decoupled from the heavy sweeps so the report can be rebuilt cheaply. Tables are
wrapped in \\resizebox so nothing overflows the margin, numbers are fixed to 4 dp,
and the best (lowest-RPS) row in each block is bolded.
"""
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent
CSV = base_dir / 'results' / 'experiment_results.csv'
WC_CSV = base_dir / 'results' / 'wc_odds.csv'
OUT = base_dir / 'experiments_report.tex'


def esc(s):
    return (str(s).replace('\\', r'\textbackslash ').replace('_', r'\_')
            .replace('&', r'\&').replace('%', r'\%').replace('{', r'\{').replace('}', r'\}'))


def metric_table(df, caption):
    """One results block -> a resizebox'd booktabs table, best RPS row bolded."""
    df = df.reset_index(drop=True)
    best = df['RPS'].astype(float).idxmin()
    lines = [r"\begin{table}[h!]", r"\centering",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Model & Config & RPS & LogLoss & Brier & Acc \\", r"\midrule"]
    for i, r in df.iterrows():
        cells = [esc(r['model']), esc(r['config']),
                 f"{r['RPS']:.4f}", f"{r['log_loss']:.4f}",
                 f"{r['Brier']:.4f}", f"{r['acc']:.3f}"]
        if i == best:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              rf"\caption{{{caption}}}", r"\end{table}", ""]
    return "\n".join(lines)


def wc_table(df, n=24):
    """World Cup championship / final odds (GK draw resolution), top-n by ensemble."""
    ens = 'champion_ensemble_gk' if 'champion_ensemble_gk' in df else 'champion_linear_gk'
    df = df.sort_values(ens, ascending=False).reset_index(drop=True).head(n)
    lines = [r"\begin{table}[h!]", r"\centering",
             r"\resizebox{0.8\textwidth}{!}{%",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Team & Champ (Lin) & Champ (MLP) & Champ (Ens) & Final (Lin) & Final (MLP) \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{esc(r['team'])} & {100*r['champion_linear_gk']:.1f}\\% & "
                     f"{100*r['champion_mlp_gk']:.1f}\\% & {100*r[ens]:.1f}\\% & "
                     f"{100*r['final_linear']:.1f}\\% & {100*r['final_mlp']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{World Cup 2026 Monte Carlo (10{,}000 sims): championship "
              r"probabilities for the linear model, the MLP, and their ensemble (average), "
              r"plus reach-final probabilities; top 24 by ensemble. Shootouts resolved by "
              r"goalkeeper rating.}", r"\end{table}", ""]
    return "\n".join(lines)


def draw_compare_table(df, n=12):
    """Side-by-side championship odds under GK vs lambda shootout resolution."""
    df = df.sort_values('champion_linear_gk', ascending=False).reset_index(drop=True).head(n)
    lines = [r"\begin{table}[h!]", r"\centering",
             r"\resizebox{0.85\textwidth}{!}{%",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"& \multicolumn{3}{c}{Linear} & \multicolumn{3}{c}{MLP} \\",
             r"Team & GK & $\lambda$ & $\Delta$ & GK & $\lambda$ & $\Delta$ \\", r"\midrule"]
    for _, r in df.iterrows():
        dl = 100 * (r['champion_linear_gk'] - r['champion_linear_lambda'])
        dm = 100 * (r['champion_mlp_gk'] - r['champion_mlp_lambda'])
        lines.append(
            f"{esc(r['team'])} & {100*r['champion_linear_gk']:.1f}\\% & "
            f"{100*r['champion_linear_lambda']:.1f}\\% & {dl:+.1f} & "
            f"{100*r['champion_mlp_gk']:.1f}\\% & {100*r['champion_mlp_lambda']:.1f}\\% & {dm:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Shootout-resolution sensitivity: championship probability when "
              r"knockout draws are decided by goalkeeper rating (GK) vs.\ overall strength "
              r"($\lambda$). $\Delta$ in percentage points.}", r"\end{table}", ""]
    return "\n".join(lines)


def changelog_section():
    """Methodology log: every change introduced, with how it was implemented."""
    items = [
        (r"FIFA-year alignment fix",
         r"Matches were keyed to squad ratings a year early. Corrected the rule to "
         r"\texttt{month $\geq$ 10 $\Rightarrow$ year+1} in \texttt{prepare\_training\_data}, "
         r"so each match merges with the squad of the game that was live at the time."),
        (r"Linear feature standardization fixed",
         r"Standardization had been recomputed per call and applied to the coefficients "
         r"(nullifying the intercept). Now the train-set mean/std are fit once in "
         r"\texttt{fit()} and reused at predict time; the intercept column is appended "
         r"\emph{after} scaling. Validation log-loss fell from $3.54$ to $0.94$."),
        (r"Dixon--Coles $\tau$ in the linear likelihood",
         r"$\rho$ was inert because the linear NLL multiplied two independent Poissons. "
         r"Added the low-score $\tau$ correction for the (0,0)/(1,0)/(0,1)/(1,1) cells; "
         r"$\rho$ now fits to $\approx -0.05$."),
        (r"MLP: mini-batches, temporal weights, early stopping",
         r"\texttt{dixon\_coles\_loss} takes per-match weights and returns a weighted mean; "
         r"the \texttt{DataLoader} carries each match's temporal-decay weight. Training "
         r"early-stops on validation NLL, which removed the over-fitting that had pinned "
         r"the MLP near the base rate."),
        (r"Leak-safe TRAIN/VAL/TEST split",
         r"Split by match date: TRAIN $<$2024-01-01, VAL 2024-01-01..2025-10-01 (all "
         r"selection), TEST $\geq$2025-10-01 (end 2025 + the World Cup, scored once)."),
        (r"Linear ridge regularization",
         r"Added an $\alpha\sum\beta^2$ penalty on the slope coefficients (not the "
         r"intercepts or $\rho$) to \texttt{\_negative\_log\_likelihood}; swept five $\alpha$."),
        (r"ELO scale / K-factor sensitivity",
         r"\texttt{elo\_calculator} parameterized by logistic divisor and a fixed $K$; "
         r"regenerated match features for divisor $\in\{200,800,1600\}$ with $K=50$."),
        (r"Reduced feature set \& friendly stripping",
         r"Evaluated a 10-feature subset (elo, bench avg, per-match goal difference, "
         r"attack/defence squad avgs) and a variant with friendlies removed from train+val."),
        (r"World Cup Monte Carlo",
         r"10{,}000-simulation bracket from openfootball/worldcup.json: group stage, "
         r"best-eight third-place teams slotted into the R32 by a memoized backtracking "
         r"matching that respects each slot's allowed groups, then the full knockout tree. "
         r"Matches are neutralized by averaging both home/away orientations."),
        (r"Goalkeeper-weighted shootouts",
         r"Knockout draws are now decided by $P(A)=\mathrm{gk}_A/(\mathrm{gk}_A+\mathrm{gk}_B)$ "
         r"using a per-squad \texttt{gk\_rating} (best goalkeeper overall, falling back to "
         r"defence average). Previously draws were resolved by overall strength $\lambda$."),
        (r"Thin-squad repair (FC26 data gaps)",
         r"Some nations have very small FC26 squads (Egypt 10 players, Iran 6, Uzbekistan 5), "
         r"giving a $0$ bench average and an inflated dispersion that the MLP extrapolated "
         r"to absurd odds (Egypt champion $65\%$). Fixes in \texttt{squad\_features.py}: "
         r"(i) std features converted to standard error ($\mathrm{std}/\sqrt{n}$) and capped "
         r"at the 85th percentile; (ii) bench average imputed for squads $\leq$20 as "
         r"squad\_avg minus the mean squad-to-bench gap of full squads; (iii) "
         r"\texttt{gk\_rating} extracted, with a defence-average fallback."),
    ]
    out = [r"\section*{Methodology changelog}",
           "Every change introduced during development, with its implementation:",
           r"\begin{enumerate}"]
    for title, body in items:
        out.append(rf"\item \textbf{{{title}.}} {body}")
    out += [r"\end{enumerate}", ""]
    return "\n".join(out)


def main():
    res = pd.read_csv(CSV)
    L = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",      # for \resizebox
        r"\usepackage[margin=1in]{geometry}",
        r"\title{World Cup Prediction: Human (Linear Dixon--Coles) vs.\ Gradient Descent (MLP)}",
        r"\author{Experiment harness}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Setup}",
        "Leak-safe split by match date: TRAIN $<$ 2024-01-01, VALIDATION 2024-01-01 to "
        "2025-10-01 (all model and hyperparameter selection), TEST $\\geq$ 2025-10-01 "
        "(end of 2025 + the 2026 World Cup; scored once). Metric of record is the Ranked "
        "Probability Score (RPS) over the ordinal home/draw/away outcome (lower is better); "
        "we also report multiclass log-loss, Brier score, and argmax accuracy. A temporal "
        "exponential decay ($\\lambda=0.001$) weights recent matches more and is applied to "
        "both models.",
        "",
    ]

    sections = [
        ('1_isolation', 'Experiment 1: Isolation and convergence (validation)'),
        ('3_linear_ridge', r'Experiment 3: Linear ridge $\alpha$ sweep (validation)'),
        ('4_reduced', 'Experiment 4: Reduced 10-feature set (validation)'),
        ('5_no_friendlies', 'Experiment 5: Friendlies removed from train + val (validation)'),
        ('7_elo', 'Experiment 6: ELO scale / K-factor sensitivity (validation)'),
        ('6_test', 'Final held-out TEST set (end 2025 + 2026), scored once'),
    ]
    for exp, title in sections:
        sub = res[res['exp'] == exp]
        if len(sub):
            L.append(rf"\section*{{{title}}}")
            L.append(metric_table(sub, title))

    # MLP sweep: top 12 by RPS
    sweep = res[res['exp'] == '2_mlp_sweep']
    if len(sweep):
        top = sweep.sort_values('RPS').head(12)
        L.append(r"\section*{Experiment 2: MLP hyperparameter sweep (validation)}")
        L.append(f"Best 12 of {len(sweep)} configurations by validation RPS. "
                 "Across the whole grid RPS spanned only "
                 f"{sweep['RPS'].min():.4f}--{sweep['RPS'].max():.4f}, "
                 "i.e.\\ architecture had little effect.")
        L.append(metric_table(top, "MLP sweep: best 12 configurations by RPS"))

    # World Cup odds
    if WC_CSV.exists():
        wc = pd.read_csv(WC_CSV)
        L.append(r"\section*{World Cup 2026 Monte Carlo simulation}")
        L.append("Each model's predicted goal expectations drive 10{,}000 simulated "
                 "tournaments (group stage, best-eight third places, full knockout bracket "
                 "from openfootball/worldcup.json). Matches are treated as neutral by "
                 "averaging both home/away orientations. Knockout ties (penalty shootouts) "
                 "are resolved by a coin flip weighted by the teams' goalkeeper ratings, "
                 "$P(A) = \\mathrm{gk}_A/(\\mathrm{gk}_A+\\mathrm{gk}_B)$.")
        L.append(wc_table(wc))
        L.append(r"\subsection*{Shootout-resolution sensitivity (GK vs.\ strength)}")
        L.append("We compare resolving knockout draws by goalkeeper rating against the "
                 "simpler overall-strength ($\\lambda$) rule. The effect on championship "
                 "odds is small, since shootouts decide only a minority of knockout games.")
        L.append(draw_compare_table(wc))

    L.append(changelog_section())

    L.append(r"\end{document}")
    OUT.write_text("\n".join(L), encoding='utf-8')
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
