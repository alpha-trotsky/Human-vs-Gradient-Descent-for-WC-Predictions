# Copula Execution Layer

The execution layer for **Copula**: connecting structural probability models (economics continuous distributions, sports Negative-Binomial + Frank copulas) to live order books, automated inconsistency scanning, risk management, and order routing.

---

## 1. System Architecture

```
                          ┌──────────────────────────────┐
                          │   Kalshi / Prediction Mkts   │
                          │   (REST v2 & WebSockets)     │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │     copula_execution         │
                          │      MarketIngestion         │
                          └──────────────┬───────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
     ┌────────────────────────┐                    ┌────────────────────────┐
     │    CoherenceScanner    │                    │  Structural Model Fit  │
     │  - Sum-to-1 Brackets   │                    │  - Economics (Normal)  │
     │  - Monotonicity P(>X)  │                    │  - Sports (NB + Frank) │
     └────────────┬───────────┘                    └────────────┬───────────┘
                  │                                             │
                  │              DisagreementFlag               │
                  └──────────────────────┬──────────────────────┘
                                         ▼
                          ┌──────────────────────────────┐
                          │      Risk Gate & Sizing      │
                          │  - Min net edge after fees   │
                          │  - Event & portfolio caps    │
                          │  - Daily circuit breaker     │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │     Execution Gateways       │
                          │  - PaperExecutionGateway     │
                          │  - KalshiExecutionGateway    │
                          └──────────────┬───────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
     ┌────────────────────────┐                    ┌────────────────────────┐
     │   Convergence Tracker  │                    │  Long-Tail Quoter      │
     │  Track price migration │                    │  Two-sided bids/asks   │
     │  toward fair value     │                    │  with inventory skew   │
     └────────────────────────┘                    └────────────────────────┘
```

---

## 2. Implemented Modules

| Module | File | Purpose |
|---|---|---|
| **Types & Contracts** | [`copula_execution/types.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/types.py) | `OrderBook`, `OrderRequest`, `OrderResult`, `DisagreementFlag`, `Position`. |
| **Ingestion** | [`copula_execution/ingestion.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/ingestion.py) | Connects to Kalshi public v2 API; fetches open events, bracket families, and reconstructs full L2 order books (inverting NO bids to YES asks). |
| **Coherence Scanner** | [`copula_execution/coherence_scanner.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/coherence_scanner.py) | Mathematically audits mutually exclusive bracket sums, monotonic strike orderings, and model-to-market copula divergences net of fees. |
| **Event Classification** | [`copula_execution/classify.py`](copula_execution/classify.py) | Decides how an event's contracts relate (`EventKind`: partition / cumulative threshold / nested by date / unknown) and *proves* exhaustiveness (bracket tiling on the print grid; Polymarket negRisk without placeholders). Everything downstream is gated on this. |
| **Pricing & Quoting** | [`copula_execution/pricer.py`](copula_execution/pricer.py) | `EventPricer` prices every contract of an event from its own live books and returns a `FairValue` (value, uncertainty, provenance); `MarketMakerQuoter` quotes around it. See section 4. Sports grids come from [`models.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/models.py) & [`core/McHale_Copula.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/core/McHale_Copula.py). |
| **Risk Management** | [`copula_execution/risk_manager.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/risk_manager.py) | Exchange fee schedules, rate-limiting, event and portfolio exposure limits, and daily circuit breakers. |
| **Execution Gateway** | [`copula_execution/gateway.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/gateway.py) | `PaperExecutionGateway` with realistic book walking & fill models, and `KalshiExecutionGateway` for live authenticated trading. |
| **Engine Orchestrator** | [`copula_execution/engine.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/engine.py) | Wires scanner, risk gates, gateway, and logs convergence rate metrics over time. |

---

## 3. Usage & Testing

### Running Tests
```bash
python -m unittest copula_execution.test_execution copula_execution.test_stress copula_execution.test_pricing
```

### Running the Backtest
```bash
python copula_execution/run_backtest.py
```

### Running the Live / Paper Execution Demo
```bash
python copula_execution/run_demo.py
```

---

## 4. Pricing Engine

### How an event is priced
| `EventKind` | Valid coherence checks | Pricing model |
|---|---|---|
| `PARTITION` with numeric edges (Kalshi brackets, ordered counts like "N Fed cuts") | sell-basket always; buy-basket only if `exhaustive` | Normal *or* Student-t (better fit wins) fitted to the liquid brackets; every bracket and tail priced from it |
| `PARTITION` without numeric structure (elections) | same | Market mids rescaled to sum to 1 - **flagged as not a model view** |
| `CUMULATIVE_THRESHOLD` ("above k" ladders) | monotonicity | CDF fitted through liquid strikes (probability noise propagated, isotonic repair) |
| `NESTED_BY_DATE` ("by <date>") | monotonicity | Monotone survival curve; constant-hazard interpolation between liquid deadlines |
| `UNKNOWN` | none | Unpriced (`fv=None`, shown as `-`, never a placeholder) |

Each `FairValue` carries `uncertainty`, `source`, `is_anchor` (its own book calibrated the model) and
`has_model_view`. A contract does **not** claim a view when the fit is poor (rmse above
`max_fit_rmse`) or when it is extrapolated beyond the last liquid deadline. Edges against the book
should only be taken from contracts with `has_model_view=True`, and must clear fees plus the
fair-value uncertainty (`CoherenceScanner.check_model_disagreement(..., uncertainty=)`).

### Why the guards exist
- *"By date" markets are nested, not exclusive.* Their prices do not sum to 1, so a basket check would report false arbitrage.
- *Bracket ("between") contracts are not threshold contracts.* A higher floor may legitimately cost more, so they are never monotonicity-checked.
- *Buying every leg is only risk-free if the legs are exhaustive.* Augmented negRisk events (placeholders such as "Party A") and truncated or gapped bracket listings are not.
- *Kalshi `mutually_exclusive` lives on the event* (`GET /events/{ticker}`), not on market objects.

### Fees
`FeeCalculator.fee_for_contract` uses the market's own schedule: Kalshi
`ceil_to_cent(0.07 * C * p * (1-p))` (per order, not per contract); Polymarket
`C * p * rate * (p*(1-p))^exponent` with `rate`/`exponent`/`feesEnabled` read from each market's `feeSchedule`
(taker-only). These change over time - ingestion copies what the venue reports; the constants in
`FeeCalculator` are only fallbacks.

### Quoter
`half_spread = target/2 + uncertainty_mult * fv_uncertainty`, widened near expiry; inventory skew is
capped at the half-spread; prices respect the market's `tick_size`. A side is omitted (`None`) when
quoting it would give value away (e.g. no bid on a 0.2c long shot).
