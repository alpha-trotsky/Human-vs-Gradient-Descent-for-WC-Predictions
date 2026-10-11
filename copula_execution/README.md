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
| **Pricing & Quoting** | [`copula_execution/pricer.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/pricer.py) | Calibrates distribution parameters from liquid brackets, integrates with [`models.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/models.py) & [`core/McHale_Copula.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/core/McHale_Copula.py), and generates inventory-skewed two-sided quotes. |
| **Risk Management** | [`copula_execution/risk_manager.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/risk_manager.py) | Exchange fee schedules, rate-limiting, event and portfolio exposure limits, and daily circuit breakers. |
| **Execution Gateway** | [`copula_execution/gateway.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/gateway.py) | `PaperExecutionGateway` with realistic book walking & fill models, and `KalshiExecutionGateway` for live authenticated trading. |
| **Engine Orchestrator** | [`copula_execution/engine.py`](file:///c:/Users/Belia/OneDrive/Documents/GitHub/WC_prediction/Human-vs-Gradient-Descent-for-WC-Predictions/copula_execution/engine.py) | Wires scanner, risk gates, gateway, and logs convergence rate metrics over time. |

---

## 3. Usage & Testing

### Running Tests
```bash
python -m unittest copula_execution.test_execution
```

### Running the Live / Paper Execution Demo
```bash
python copula_execution/run_demo.py
```
