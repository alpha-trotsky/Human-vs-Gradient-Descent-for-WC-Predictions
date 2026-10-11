# Copula + Market-Calibrated Scoreline Models — Architecture & Approach

This document covers the part of the project that the main write-up (`reports/project_report.pdf`) does **not**: everything built after the Linear-vs-MLP Dixon-Coles study. The report only mentions copulas and Negative-Binomial marginals as "Future Steps"; this extension implements them, connects them to live Kalshi prediction-market prices, and backtests them on the 2026 World Cup and international friendlies.

The core idea in one sentence: **let the betting market decide how good each team is, and let the historical data decide what the scoreline distribution looks like.**

---

## 1. Why this extension exists

The Dixon-Coles model in the report has two structural weaknesses:

1. **Poisson marginals.** Poisson forces variance = mean. International goals are mildly over-dispersed (more blowouts and more 0-0s than Poisson predicts).
2. **The τ correction is a patch.** Dixon-Coles only adjusts the four scorelines 0-0, 1-0, 0-1, 1-1. Dependence between home and away goals anywhere else in the table is ignored.

It also has a practical weakness: a model fitted only on historical results is a *worse* estimator of team strength than a liquid betting market 2 hours before kickoff, since the market already prices in injuries, line-ups, motivation, and so on. But the market doesn't give you a full scoreline distribution. Kalshi quotes Win/Draw/Loss and a few Over/Under lines, and exact-score prices only for WC games (and those are thin).

The extension fixes both:

- Replace Poisson + τ with **Negative-Binomial marginals joined by a Frank copula** (full-table dependence).
- Take the **location** (expected goals μ_home, μ_away) from the market, and the **shape** (dispersion r and dependence θ) from history.

---

## 2. Timeline

| Commit | Date | What landed |
|---|---|---|
| `f31d274` "copula formulation" | 2026-07-01 | `core/McHale_Copula.py`: NB marginals + Frank copula, feature-driven (Elo + FC ratings). Calibration plots (`analysis/calibration_copula.py`). Plugged into `wc_simulation.py` / `canada_query.py`. |
| `3a84034` "some updates" | 2026-07-03 | Kalshi scraper (`kalshi_national_team_odds.py`), `odds.csv`, first two backtests (`backtest/backtest_odds.py`, `backtest/backtest_totals.py`), `backtest_report.pdf`. |
| `f7673ae` | 2026-07-06 | Scrapes and test files for correct-score / O-U / team-total markets. |
| `ae262ce` "5 new models" | 2026-07-08 | `models.py`: the M1–M5 walk-forward engine, market calibration (M4/M5), exact-scoreline backtest against `kalshi_wc_exact_score_odds.csv`. |

---

## 3. System overview

```
                   results/results.csv  (49k+ internationals, 1872–2026)
                                  │
                   5-year window, exp(-0.0025·days) weights
                                  │
            ┌─────────────────────┼──────────────────────┐
            ▼                     ▼                      ▼
     M1 Poisson + DC        M2 NegBin + DC        M3 NegBin + Frank copula
                                                         │
                                          shared shape: r_home, r_away, θ
                                          prior location: μ_home⁰, μ_away⁰
                                                         │
   Kalshi odds.csv (price 2h pre-KO) ──devig──►  per-match calibration of (μ_home, μ_away)
                                                    │                     │
                                              target: 1X2           target: 1X2 + O/U
                                                    ▼                     ▼
                                                   M4                    M5
                                                         │
                     every model → 11×11 scoreline grid P(H=i, A=j)
                                                         │
          ┌──────────────────────────────┬───────────────┴──────────────────┐
          ▼                              ▼                                  ▼
   Accuracy vs results           1X2 PnL vs Kalshi             Exact-score PnL vs Kalshi
   (RPS, log-loss, acc)          (all matches)                 (WC only, M3/M4/M5)
```

Everything downstream consumes one object: the **scoreline grid** `M[i, j] = P(home = i, away = j)`, truncated at 10 goals. 1X2, Over/Under and exact-score probabilities are all sums over cells of that grid (`grid_to_1x2`, `grid_to_over`), so every market is priced from one consistent joint distribution.

---

## 4. The statistical model

### 4.1 Negative-Binomial marginals

Each side's goals follow a Negative-Binomial with mean μ and dispersion r:

$$H \sim \text{NB}(\mu_h, r_h), \quad A \sim \text{NB}(\mu_a, r_a), \quad \text{Var} = \mu + \mu^2 / r$$

As r → ∞ this collapses to Poisson, so M2 and M3 can only do at least as well as Poisson on fit. r_home and r_away are **global** (one value each for the whole team universe), not per-team.

### 4.2 Frank copula (joint dependence)

The marginal CDFs `F_h`, `F_a` are coupled with Frank's copula:

$$C(u, v; \theta) = -\frac{1}{\theta}\log\left(1 + \frac{(e^{-\theta u}-1)(e^{-\theta v}-1)}{e^{-\theta}-1}\right)$$

θ > 0 means positive dependence (goals come together, so open games), θ < 0 means negative dependence, and θ = 0 means independence. Because goals are discrete, the joint PMF is the **rectangle difference** of the copula (`core/McHale_Copula.py:joint_pmf`, `score_matrix`):

$$P(H{=}h, A{=}a) = C(F_h(h), F_a(a)) - C(F_h(h{-}1), F_a(a)) - C(F_h(h), F_a(a{-}1)) + C(F_h(h{-}1), F_a(a{-}1))$$

Unlike Dixon-Coles' τ, this shapes **every** cell of the table, not just the four low-score ones. This is the "McHale" formulation (NB marginals + Frank copula, after McHale & Scarf).

### 4.3 Two parameterisations of μ

The copula machinery is used in two places, and the two versions differ in how μ is produced:

| | `core/McHale_Copula.py` (Jul 1) | `models.py` M3 (Jul 8) |
|---|---|---|
| μ from | 20-dim feature regression (Elo, FC squad ratings), like the report's linear model | Classic Dixon-Coles team parameters: `log μ_h = home_adv·(1−neutral) + attack_home − defence_away` |
| θ | Constant, or a function of the quality gap (`k_mode` = const / log / exp / quad) | Constant |
| Fit | Once, static train/val/test split, ridge α sweep (best: α=50, `const`) | **Refit walk-forward**, once per match date, strictly on prior data |
| Used by | `wc_simulation.py`, `canada_query.py`, `backtest/*.py` | M1–M5 backtest |

The `models.py` version drops the feature regression on purpose: team attack/defence ratings fitted on recent results don't depend on FC video-game ratings. That makes it easier to refit over time and easier to compare like-for-like against M1/M2.

---

## 5. The market-calibration idea (M4 / M5)

This is the novel piece.

A historically fitted model has two jobs, and it does them with very different quality:

- **Location** (how many goals each team is expected to score in *this* match) is where historical fits are weakest. They can't see line-ups, injuries, rotation in friendlies, or motivation.
- **Shape** (how goals are dispersed around that mean, how often games are 0-0 versus 3-3, and how home and away goals co-move) is a stable property of international football. Tens of thousands of matches estimate it well.

A liquid prediction market 2h before kickoff is the opposite. It's excellent at location but only gives you 3 numbers (W/D/L) and maybe a few O/U lines, which is far too little to recover a full scoreline distribution on its own.

**M4/M5 combine the two:**

1. Fit M3 on history → get the shape parameters `(r_h, r_a, θ)` and a prior location `(μ_h⁰, μ_a⁰)`.
2. Devig the Kalshi prices 2h before kickoff (proportional normalisation to sum to 1).
3. **Freeze** `(r_h, r_a, θ)` and re-solve only `(μ_h, μ_a)` for this match, minimising cross-entropy between the model's implied probabilities and the market's (`models.py:_calib_loss`, `calibrate_mu_to_market`):

$$\min_{\mu_h,\mu_a}\; -\sum_{o\in\{H,D,A\}} p^{\text{mkt}}_o \log p^{\text{model}}_o(\mu_h,\mu_a) \;\;\underbrace{-\sum_{\ell} \Big[p^{\text{mkt}}_{>\ell}\log p^{\text{model}}_{>\ell} + (1-p^{\text{mkt}}_{>\ell})\log(1-p^{\text{model}}_{>\ell})\Big]}_{\text{M5 only: O/U lines } \ell = 0.5, 1.5, \dots}$$

   This is a maximum-likelihood fit where the "data" are the market's soft probabilities instead of one realised scoreline. It's optimised in log-μ space with L-BFGS-B, starting from M3's μ⁰, and falls back to μ⁰ if there are no prices or the optimiser doesn't improve.
4. Build the full 11×11 NB-Frank grid from the calibrated μ and the historical shape. **That grid is the exact-scoreline prediction.**

In short:

- **M4** calibrates to 1X2 only. 3 probabilities (2 degrees of freedom) pin down 2 unknowns, so the market fixes the goal *difference* balance, while the historical shape determines total goals and exact scores.
- **M5** calibrates to 1X2 plus every available O/U line. Now the market also informs the total goals level, so the problem is over-determined and the optimiser finds a compromise. If a match has no O/U prices (26 of 178 scored matches), M5 = M4.

---

## 6. Fitting engine (M1–M3)

`models.py:fit_strength_model` is one shared weighted-MLE fitter for all three historical models.

- **Parameters:** `home_adv`, `attack[T]`, `defence[T]`, plus model-specific extras (`ρ` for M1; `log r_h, log r_a, ρ` for M2; `log r_h, log r_a, θ` for M3). ρ is squashed via `0.9·tanh(·)` to keep τ positive.
- **Data window:** 5-year lookback, match weights `exp(−0.0025·days)` (half-life ≈ 0.76 yr), and a tiny ridge (1e-6) purely for identifiability.
- **Gradient trick.** Analytic gradients through NB CDFs and the Frank copula are painful. Instead:
  - central differences are taken **only** with respect to the two per-match scalars (μ_h, μ_a) and the handful of shared extras, costing about 10 vectorised likelihood calls;
  - the chain rule back to every attack/defence parameter is then exact, because μ = exp(linear), and is applied with a scatter-add (`np.add.at`).

  So one gradient evaluation costs a constant amount regardless of how many teams there are, and refitting three models on every one of the ~41 match dates stays tractable.
- **Walk-forward** (`walk_forward_predict`): for each unique kickoff date, refit on data strictly before that date, then predict that day's matches. No leakage. Teams not seen in the window get average strength (attack = defence = 0).
- **Cache:** results are pickled to `results/models_records.pkl`, so `python models.py` re-scores without refitting.

---

## 7. Data

| Source | File | Notes |
|---|---|---|
| International results | `results/results.csv` | Training corpus + ground-truth scores for test matches (±2-day fuzzy date join, handles swapped home/away orientation on neutral sites). |
| Kalshi 1X2 + totals | `odds.csv` via `kalshi_national_team_odds.py` | Public API, last hourly candle ≤ kickoff − 2h. Series `KXWCGAME`/`KXWCTOTAL` (World Cup), `KXINTLFRIENDLYGAME`/`KXINTLFRIENDLYTOTAL` (friendlies). |
| Kalshi exact score | `kalshi_wc_exact_score_odds.csv` | WC only. Outcomes like "Belgium wins 4-1" / "Draw 0-0", parsed to (home, away) orientation. |
| Name mapping | `models.py:norm`, `ODDS_TO_RES` | Unicode-stripped normalisation + alias table (e.g. "Korea Republic" → "south korea"). |

Test set: 221 matches with odds, of which **178 have a final score** (73 World Cup, 105 friendlies). 152 of the 178 have O/U prices, and all 73 WC matches have exact-score prices.

---

## 8. Results

All numbers are from the cached walk-forward run (`results/models_records.pkl`).

### 8.1 Probability accuracy (1X2, 178 scored matches)

| Model | RPS ↓ | Log-loss ↓ | Accuracy |
|---|---|---|---|
| M1 Poisson+DC | 0.1440 | 0.7926 | 66.3% |
| M2 NegBin+DC | 0.1440 | 0.7925 | 66.3% |
| M3 NegBin+Frank | 0.1423 | 0.7868 | 67.4% |
| **M4 calib 1X2** | **0.1311** | 0.7283 | **70.2%** |
| M5 calib 1X2+O/U | 0.1337 | 0.7355 | 66.3% |
| Kalshi (devigged) | 0.1314 | **0.7243** | 70.1% |

**World Cup only (73):**

| Model | RPS | Log-loss | Acc |
|---|---|---|---|
| M3 | 0.1391 | 0.7894 | 72.6% |
| M4 | 0.1385 | 0.7700 | 71.2% |
| **M5** | **0.1302** | **0.7089** | 68.5% |
| Kalshi | 0.1385 | 0.7664 | 71.2% |

**Friendlies only (105):**

| Model | RPS | Log-loss | Acc |
|---|---|---|---|
| M3 | 0.1445 | 0.7849 | 63.8% |
| **M4** | **0.1260** | 0.6993 | 69.5% |
| M5 | 0.1362 | 0.7540 | 64.8% |
| Kalshi | 0.1264 | 0.6947 | 69.2% |

### 8.2 1X2 PnL vs Kalshi ($1 contracts, 2¢ edge threshold)

| Model | Bets | PnL | Hit | ROI |
|---|---|---|---|---|
| M1 | 443 | +14.71 | 35.9% | +3.3% |
| M2 | 444 | +14.55 | 35.6% | +3.3% |
| **M3** | 445 | **+18.77** | 35.3% | **+4.2%** |
| M4 | 30 | −1.34 | 53.3% | −4.5% |
| M5 | 343 | +5.02 | 35.3% | +1.5% |

### 8.3 Exact-scoreline PnL vs Kalshi (WC only)

| Model | Bets | PnL | Hit | ROI |
|---|---|---|---|---|
| M3 | 691 | +2.93 | 8.0% | +0.4% |
| M4 | 654 | −5.56 | 7.8% | −0.9% |
| **M5** | 708 | **+20.09** | 8.2% | **+2.8%** |

---

## 9. What the results say

1. **Copula > τ, NB ≈ Poisson.** M2 is indistinguishable from M1, so over-dispersion barely matters at the 1X2 level. M3's Frank copula gives a small, consistent improvement in every metric. That's the "Future Steps" hypothesis from the report, confirmed, although the effect is modest.
2. **Market location + historical shape matches the market on 1X2.** Overall, M4 ties Kalshi on RPS and slightly beats it on accuracy. That's expected, because M4 is fitted to reproduce Kalshi's 1X2. The real point is that M4 does this *while* producing a full scoreline grid the market doesn't publish.
3. **O/U information is what makes scorelines good.** M5 is the best-calibrated model on World Cup matches, where it beats Kalshi's own 1X2 RPS (0.1302 vs 0.1385), and it's the only model that makes meaningful money on exact-score contracts (+2.8% ROI). The O/U lines pin down the *total goals* level, and that's the missing piece for exact scores. On friendlies (noisier O/U markets, rotated squads) the extra constraint hurts, and M4 is best there.
4. **Accuracy ≠ edge.** M4 barely disagrees with the market, so it only places 30 1X2 bets. Those bets are essentially calibration residuals (5-goal truncation in the calibration loss, optimiser fallbacks), not signal, and they lose. M3 is the *least* market-like of the copula models and has the best 1X2 ROI. Market-calibrated models are the tool for **derivative markets** (exact score, totals); independent historical models are the tool for **disagreeing with the headline market**.
5. **Contrast with `backtest_report.pdf`.** The earlier feature-based McHale-Copula backtest lost 2.3% ROI on WC moneylines. The walk-forward attack/defence M3 in `models.py` is positive. The two runs differ in data (friendlies included), fitting (walk-forward vs a static pre-2025-10 fit), and PnL accounting (see caveats), so they aren't directly comparable.

---

## 10. Caveats (read before quoting any ROI)

- **PnL is measured against devigged prices, not tradable prices.** In `backtest_1x2` / `backtest_scorelines`, entry price = normalised implied probability. Real fills pay the spread and vig (≈2–4% on Kalshi WC markets), which is roughly the size of every ROI above. All PnL figures are upper bounds.
- **Exact-score devig normalises over *listed* scorelines only.** Kalshi doesn't list every scoreline, so normalising the listed ones to 1 inflates their probabilities. This biases the scoreline backtest toward profitable SELLs.
- **Small samples.** There are 178 scored matches and 73 WC. No confidence intervals or significance tests have been computed, and a ±$20 PnL on ~700 correlated contracts (many contracts per match) is well within noise.
- **Proportional devig** ignores favourite-longshot bias. Shin or power devig methods may change M4/M5.
- **Global shape.** r_h, r_a and θ are shared by every team and every match. The `k_mode` options in `core/McHale_Copula.py` (θ depending on the quality gap) are not used in M3–M5.
- **Calibration grid** is truncated at 5 goals (`CAL_GOALS`), but predictions use 10 (`MAX_GOALS`). The mismatch is small but non-zero, and it's the main source of M4's residual disagreement with the market.
- **Unseen teams** default to average strength. That affects some minor-nation friendlies.

---

## 11. Running it

```bash
python models.py                      # uses cached walk-forward records
python -c "import models; models.main(use_cache=False)"   # full refit (~41 dates × 3 models)

python -m core.McHale_Copula          # feature-based copula: alpha / k_mode sweep
python -m analysis.calibration_copula # scoreline calibration curves
python backtest/backtest_odds.py      # original moneyline backtest
python backtest/backtest_totals.py    # asymmetric-result / totals backtest
python kalshi_national_team_odds.py   # re-scrape Kalshi odds (public API)
```

Outputs: `results/models_backtest_bets.csv` (1X2 bet log), `results/models_backtest_scoreline_bets.csv` (exact-score bet log), `results/models_records.pkl` (per-match grids for every model).

---

## 12. File map

```
models.py                        M1–M5 engine: data joins, devig, fitters, walk-forward,
                                 market calibration, evaluation, both PnL backtests
core/McHale_Copula.py            Frank copula, NB-Frank joint PMF / score matrix,
                                 feature-driven McHaleCopulaModel + alpha/k_mode sweeps
analysis/calibration_copula.py   Reliability curves for scoreline probabilities
backtest/backtest_odds.py        Feature-copula vs Kalshi moneyline (WC)
backtest/backtest_totals.py      Heavy-favourite → over-goals hypothesis (WC)
kalshi_national_team_odds.py     Kalshi public-API scraper (price 2h pre-kickoff)
odds.csv                         Kalshi 1X2 + totals snapshots
kalshi_wc_exact_score_odds.csv   Kalshi WC correct-score snapshots
results/backtest_report.tex/.pdf Write-up of the two feature-copula backtests
```

## References

- McHale, I. & Scarf, P. (2007). Modelling soccer matches using bivariate discrete distributions with general dependence structure. *Statistica Neerlandica*, 61(4).
- McHale, I. & Scarf, P. (2011). Modelling the dependence of goals scored by opposing teams in international soccer matches. *Statistical Modelling*, 11(3).
- Dixon, M. J. & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *JRSS C*, 46(2).
- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
