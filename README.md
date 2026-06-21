# Human vs Gradient Descent — WC 2026 Predictions

A numerical comparison of **linear regression** and a **multilayer perceptron (MLP)** as parameter estimators for the Dixon-Coles football match model, applied to predicting the 2026 FIFA World Cup.

---

## What this is

The Dixon-Coles model treats the goals scored by each side as two correlated Poisson random variables parameterised by (λ_home, λ_away, ρ). The question this project answers is: **does a neural network estimate those parameters better than a structured linear fit?**

The short answer is: barely, and at a significant cost in robustness.

On a held-out test set the MLP achieves RPS = 0.1756 vs. the linear model's RPS = 0.1762 — statistically within noise. But when those same models are used to simulate the World Cup 10,000 times, the MLP is wildly unstable: changing the network width from 64 to 256 hidden units shifts the predicted tournament favourite from Spain to Germany. The linear model's championship probabilities (Spain 14%, France 10.8%, Argentina 8.6%) are stable across all hyperparameter settings and close to bookmaker expectations.

The thesis: equal single-match accuracy hides a large gap in robustness, in favour of the simpler, interpretable "human" model.

Full write-up: [`reports/project_report.pdf`](reports/project_report.pdf)

---

## Features

**Match results** (`pipeline/data_pipeline.py`):
- Walk-forward **Elo ratings** (divisor 800, tournament-scaled K) computed from all international results since 2000
- **Goals scored and conceded** over each team's last 10 matches as separate features (not collapsed to goal difference)

**Squad quality** (`pipeline/squad_features.py`):
- Sourced from EA Sports FC / FIFA player ratings (FC15–FC26)
- Per team: `squad_avg`, `squad_std`, `attack_avg`, `midfield_avg`, `defence_avg`, `bench_avg`, `bench_std`, goalkeeper rating
- Matched by FIFA year (FC game release maps to the calendar year of the tournament)

The full feature vector is 20-dimensional (home and away versions of all the above).

---

## Models

Both models are fitted by maximising the Dixon-Coles log-likelihood on temporally weighted match data (exponential decay, recent matches count more).

| | Linear | MLP |
|---|---|---|
| Architecture | Two linear heads (λ_home, λ_away) + scalar ρ | 3-layer ReLU MLP, softplus output for λ, tanh for ρ |
| Optimiser | L-BFGS-B (scipy) | Adam with gradient clipping |
| Regularisation | Ridge penalty α = 10 on slope coefficients | Dropout + early stopping on val NLL |
| Test RPS | 0.1762 | 0.1756 |

Knockout ties are resolved by goalkeeper rating (gk_A / (gk_A + gk_B)) rather than overall λ, reflecting that penalty shootouts are dominated by the keeper.

---

## Key findings

- Both models beat a base-rate baseline (RPS 0.2234) by a wide margin; neither beats the other significantly.
- The MLP's tournament forecasts are brittle to architecture choices; the linear model's are not.
- The linear model is interpretable: opposition Elo suppresses goal expectation ~3× more than own Elo raises it; goals conceded is ~2× more informative than goals scored; a strong defence increases expected goals (likely via an aggressive defensive line enabling higher press).
- Dropping friendly matches from training hurts both models — friendlies carry real squad-quality signal despite lower stakes.
- Feature reduction to 10 features costs almost nothing in accuracy, suggesting the positional sub-features are largely redundant.

---

## Repo structure

```
wc_simulation.py        Run the full WC 2026 Monte Carlo simulation (10 000 draws).
                        Prints championship probabilities for all 48 teams, linear
                        and MLP side-by-side.

canada_query.py         Query tool for any single fixture. Given two team names,
                        prints expected goals, W/D/L odds, margin/total/BTTS
                        markets, and group qualification probabilities for all
                        three models. Saves a 3D scoreline surface to figures/.

core/
  dixon_coles.py        Dixon-Coles joint scoreline matrix and W/D/L probabilities
                        from two Poisson rates + the τ low-score correction.
  training.py           LinearRegressionDixonColes class (L-BFGS-B MLE fit) and
                        prepare_training_data() — loads CSVs, merges squad
                        features by FIFA year, builds the feature matrix with
                        temporal decay weights.
  evaluate.py           Scoring rules (RPS, log-loss, Brier, accuracy) and
                        model-to-probability adapters (predict_linear, predict_mlp).
  experiments.py        Full experiment harness: train/val/test split, MLP
                        hyperparameter sweep (55 configs), ridge α sweep, feature
                        reduction, friendlies ablation, and final test evaluation.
                        Writes results/experiment_results.csv and
                        reports/experiments_report.tex.
  model_store.py        Save/load trained models to models/ so canada_query.py
                        doesn't retrain from scratch on every run.

pipeline/
  data_pipeline.py      Builds match_results_elo.csv from the raw results CSV:
                        walk-forward Elo ratings + last-10 goals scored/conceded.
  squad_features.py     Builds squad_features.csv from FC15–FC26 player CSVs:
                        per-team squad/positional averages and goalkeeper rating.

analysis/
  elo_experiment.py     Sensitivity study: reruns best models with Elo divisor
                        D ∈ {200, 800, 1600} and fixed K=50. Appends results to
                        experiment_results.csv.
  figures.py            Generates the joint scoreline surface figure (Spain vs
                        France) used in the report.
  inspect_features.py   Diagnostic script: prints per-team feature vectors and
                        neutral λ estimates for every Group C pairing, and checks
                        how the MLP's input standardization affects each feature.

data/
  worldcup/             worldcup2026.json — official bracket (groups, R32, KO tree)
  fc24/                 FC15–FC24 player ratings (male_players.csv)
  fc25/                 FC25 player ratings
  fc26/                 FC26 player ratings
  results/              results.csv — raw international match results

results/                Generated CSVs: match_results_elo.csv, squad_features.csv,
                        experiment_results.csv

models/                 Cached trained models (linear_dc.pkl, mlp_dc.pt).
                        Auto-created on first run of canada_query.py.

figures/                Output plots (scoreline surfaces).
reports/                project_report.pdf — full write-up
                        experiments_report.pdf — experiment tables (auto-generated)
```

---

## Running

**Simulate the World Cup:**
```bash
python wc_simulation.py
```

**Query a match (e.g. Canada vs Switzerland):**
```bash
python canada_query.py "Canada" "Switzerland"
python canada_query.py "Brazil" "Argentina" --no-plot
```

**Re-run experiments** (takes ~30–60 min):
```bash
python -m core.experiments
```

**Rebuild data pipeline** (requires raw CSVs in `data/`):
```bash
python -m pipeline.data_pipeline    # builds match_results_elo.csv
python -m pipeline.squad_features   # builds squad_features.csv
```

---

## References

- Dixon, M. J. & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265–280.
- Arntzen, H. & Hvattum, L. M. (2021). Predicting match outcomes in association football using team ratings and player ratings. *Statistical Modelling*, 21(5).
