"""
Copula Execution Layer: Real-Book Backtest Engine.

Backtests the production decision paths (CoherenceScanner, FeeCalculator, EventPricer-style
uncertainty) on historical Kalshi bid/ask snapshots taken ~2h before kickoff (odds.csv,
kalshi_wc_exact_score_odds.csv):

  1. Mutually exclusive 1X2 baskets            -> scanner.check_mutually_exclusive_sum
  2. Over/Under ladder monotonicity            -> scanner.check_monotonicity
  3. Exact-score model vs. market, per match   -> scanner.check_model_disagreement

Everything is evaluated against the *realised* outcomes in the data (no payoff is assumed), with
the market's own fee schedule applied per order at the lot size the scanner would trade.

Limits worth keeping in mind when reading the numbers:
  * snapshots carry only the touch (best bid / best ask); depth is assumed to be >= the lot size;
  * a single snapshot per match, so no queueing/latency effects;
  * for test 3 the model grid is the walk-forward M3 prediction (fitted only on matches before the
    kickoff date, market-independent). M4/M5 are calibrated to Kalshi prices and are deliberately
    not used: pricing the market with a model fitted to that market would be circular.
"""
import math
import os
import pickle
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from copula_execution.types import (
    MarketVenue, MarketContract, OrderBook, OrderBookLevel, EventKind
)
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.risk_manager import FeeCalculator

SIZE = 10                     # contracts per leg: the lot the scanner suggests (fees are rounded per order)
MIN_EDGE = 0.015
MODEL_ABS_UNC = 0.005         # model uncertainty used for test 3: sqrt(abs^2 + (rel * fv)^2),
MODEL_REL_UNC = 0.25          # the same relative term the EventPricer uses
MODELS_CACHE = ROOT / "results" / "models_records.pkl"
N_BOOT = 2000


def _book(ticker, bid, ask):
    bids = [OrderBookLevel(float(bid), 100)] if bid is not None and bid > 0 else []
    asks = [OrderBookLevel(float(ask), 100)] if ask is not None and ask > 0 else []
    return OrderBook(ticker, MarketVenue.KALSHI, datetime.now(timezone.utc), yes_bids=bids, yes_asks=asks)


def _num(x):
    return float(x) if pd.notnull(x) else None


def _won(row):
    return str(row["result"]).strip().lower() == "yes"


def _binary_settled(grp):
    """
    True iff every leg settled plain yes/no. Voided / partially settled markets resolve as 'scalar'
    (a fixed cash value we do not have), so their payoff cannot be computed and they are excluded
    rather than mis-scored as 'no'.
    """
    return grp["result"].astype(str).str.strip().str.lower().isin(["yes", "no"]).all()


def _bootstrap_ci(per_cluster, n_boot=N_BOOT, seed=7):
    """90% CI of sum(pnl)/sum(capital) resampling whole clusters (matches), not individual bets."""
    if len(per_cluster) < 2:
        return None
    arr = np.array(per_cluster, dtype=float)   # columns: pnl, capital
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    sums = arr[idx].sum(axis=1)
    roi = sums[:, 0] / np.where(sums[:, 1] > 0, sums[:, 1], np.nan)
    lo, hi = np.nanpercentile(roi, [5, 95])
    return float(lo), float(hi)


def run_historical_backtest():
    print("=" * 80)
    print(" COPULA EXECUTION LAYER - REAL ORDER BOOK BACKTEST")
    print(" Pre-kickoff Kalshi bid/ask snapshots; realised outcomes; per-order fee schedule")
    print(f" Lot size {SIZE} contracts per leg; touch depth assumed >= lot")
    print("=" * 80)

    odds_df = pd.read_csv(ROOT / "odds.csv")
    score_df = pd.read_csv(ROOT / "kalshi_wc_exact_score_odds.csv")
    scanner = CoherenceScanner(min_edge=MIN_EDGE, trade_size=SIZE)

    totals = {}

    # -------------------------------------------------------------
    # 1. MUTUALLY EXCLUSIVE 1X2 BASKETS
    # -------------------------------------------------------------
    print("\n[BACKTEST 1] Mutually exclusive 1X2 baskets (home / draw / away)")
    ml = odds_df[odds_df["series"].isin(["KXWCGAME", "KXINTLFRIENDLYGAME"])]
    events = baskets_buy = baskets_sell = 0
    pnls, fees_total, bad_payoff, void_events = [], 0.0, 0, 0
    for ev_ticker, grp in ml.groupby("event_ticker"):
        if len(grp) != 3:
            continue   # need exactly home / draw / away
        if not _binary_settled(grp):
            void_events += 1
            continue
        events += 1
        contracts, books, won = [], {}, {}
        for _, row in grp.iterrows():
            t = row["market_ticker"]
            # Regulation-time 1X2 with exactly three legs is exhaustive by construction (the leg
            # count is checked above); the scanner cannot prove this from the CSV, so we assert it.
            contracts.append(MarketContract(t, ev_ticker, MarketVenue.KALSHI, row["outcome"],
                                            event_kind=EventKind.PARTITION, exhaustive=True))
            books[t] = _book(t, _num(row["yes_bid"]), _num(row["yes_ask"]))
            won[t] = _won(row)
        cmap = {c.ticker: c for c in contracts}
        yes_legs = sum(won.values())
        if yes_legs != 1:
            bad_payoff += 1   # data check: exactly one leg of a 1X2 must resolve YES

        for f in scanner.check_mutually_exclusive_sum(ev_ticker, contracts, books):
            legs = [t["ticker"] for t in f.suggested_trades]
            realised_yes = sum(won[t] for t in legs)
            if f.flag_id.startswith("arb-sum-under"):
                baskets_buy += 1
                cost = SIZE * sum(books[t].best_yes_ask for t in legs)
                fees = sum(FeeCalculator.fee_for_contract(cmap[t], books[t].best_yes_ask, SIZE) for t in legs)
                pnl = SIZE * realised_yes - cost - fees
            else:
                baskets_sell += 1
                proceeds = SIZE * sum(books[t].best_yes_bid for t in legs)
                fees = sum(FeeCalculator.fee_for_contract(cmap[t], books[t].best_yes_bid, SIZE) for t in legs)
                pnl = proceeds - SIZE * realised_yes - fees
            fees_total += fees
            pnls.append(pnl)

    wins = sum(1 for p in pnls if p > 0)
    print(f"  > Events evaluated:               {events}  (excluded {void_events} voided/scalar-settled; "
          f"data check: {bad_payoff} with != 1 winning leg)")
    print(f"  > Buy-basket flags / sell-basket: {baskets_buy} / {baskets_sell}")
    print(f"  > Fees paid:                      ${fees_total:.2f}")
    print(f"  > Realised net PnL:               ${sum(pnls):+.2f}")
    print(f"  > Profitable baskets:             {wins}/{len(pnls)}"
          + ("   (arbitrage is risk-free only if every leg fills at the touch)" if pnls else ""))
    totals["1X2 basket arbitrage"] = sum(pnls)

    # -------------------------------------------------------------
    # 2. OVER/UNDER LADDER MONOTONICITY
    # -------------------------------------------------------------
    print("\n[BACKTEST 2] Over/Under ladder monotonicity (P(Over k) must fall as k rises)")
    tot = odds_df[odds_df["series"].isin(["KXWCTOTAL", "KXINTLFRIENDLYTOTAL"])]
    n_events = n_legs = n_parse_fail = n_void = 0
    mono_flags, mono_pnls, mono_fees = 0, [], 0.0
    pat = re.compile(r"Over\s+([0-9]+(?:\.[0-9]+)?)\s+goals", re.IGNORECASE)
    for ev_ticker, grp in tot.groupby("event_ticker"):
        if not _binary_settled(grp):
            n_void += 1
            continue
        contracts, books, won = [], {}, {}
        for _, row in grp.iterrows():
            m = pat.search(str(row["outcome"]))
            if not m:
                n_parse_fail += 1
                continue
            k = float(m.group(1))
            t = row["market_ticker"]
            c = MarketContract(t, ev_ticker, MarketVenue.KALSHI, row["outcome"], floor_strike=k,
                               event_kind=EventKind.CUMULATIVE_THRESHOLD)
            c.model_edges = (k, None)
            contracts.append(c)
            books[t] = _book(t, _num(row["yes_bid"]), _num(row["yes_ask"]))
            won[t] = _won(row)
        if len(contracts) < 2:
            continue
        n_events += 1
        n_legs += len(contracts)
        cmap = {c.ticker: c for c in contracts}
        for f in scanner.check_monotonicity(ev_ticker, contracts, books):
            mono_flags += 1
            buy, sell = f.suggested_trades          # buy the superset (lower strike), sell the subset
            fees = (FeeCalculator.fee_for_contract(cmap[buy["ticker"]], buy["price"], SIZE)
                    + FeeCalculator.fee_for_contract(cmap[sell["ticker"]], sell["price"], SIZE))
            payoff = SIZE * (int(won[buy["ticker"]]) - int(won[sell["ticker"]]))
            pnl = SIZE * (sell["price"] - buy["price"]) + payoff - fees
            mono_fees += fees
            mono_pnls.append(pnl)

    print(f"  > Ladders evaluated: {n_events} ({n_legs} threshold contracts; {n_parse_fail} unparsable rows; "
          f"{n_void} voided/scalar-settled excluded)")
    print(f"  > Monotonicity inversions flagged (net of fees): {mono_flags}")
    print(f"  > Fees paid: ${mono_fees:.2f} | Realised net PnL: ${sum(mono_pnls):+.2f}")
    if mono_pnls:
        print(f"  > Payoff floor check: worst single trade ${min(mono_pnls):+.2f} (must be >= 0 for a true arbitrage)")
    totals["O/U monotonicity"] = sum(mono_pnls)

    # -------------------------------------------------------------
    # 3. EXACT-SCORE MODEL VS MARKET (per-match model grids)
    # -------------------------------------------------------------
    print("\n[BACKTEST 3] Exact-score pricing: walk-forward M3 grid vs Kalshi bid/ask")
    import models as mdl     # reuse the project's own scoreline parser (handles home/away orientation)

    if not MODELS_CACHE.exists():
        print(f"  ! {MODELS_CACHE.name} not found - run `python models.py` to build the walk-forward records. Skipping.")
        totals["Exact-score model"] = 0.0
    else:
        with open(MODELS_CACHE, "rb") as fh:
            records = {r["match_id"]: r for r in pickle.load(fh)}

        n_matches = n_unmatched = n_unparsed = n_rows = n_void = 0
        bets = []   # (match, side, pnl, capital, fee, won_bet)
        cal_model, cal_hit = [], []
        for ev_ticker, grp in score_df.groupby("event_ticker"):
            if not _binary_settled(grp):
                n_void += 1
                continue
            rec = records.get(ev_ticker.split("-", 1)[1])
            if rec is None:
                n_unmatched += 1
                continue
            n_matches += 1
            grid = rec["grids"]["nbinom_frank"]
            home_raw, away_raw = mdl._split_match_teams(grp["match"].iloc[0])
            for _, row in grp.iterrows():
                ij = mdl._parse_score_outcome(row["outcome"], home_raw, away_raw)
                if ij is None:
                    n_unparsed += 1
                    continue
                n_rows += 1
                i, j = ij
                fv = float(grid[i, j]) if i < grid.shape[0] and j < grid.shape[1] else 0.0
                hit = _won(row)
                cal_model.append(fv)
                cal_hit.append(float(hit))
                bid, ask = _num(row["yes_bid"]), _num(row["yes_ask"])
                c = MarketContract(row["market_ticker"], ev_ticker, MarketVenue.KALSHI, row["outcome"])
                book = _book(c.ticker, bid, ask)
                unc = math.hypot(MODEL_ABS_UNC, MODEL_REL_UNC * fv)
                # BUY and SELL are evaluated independently inside the scanner
                flag = scanner.check_model_disagreement(ev_ticker, c, fv, book, uncertainty=unc)
                if flag is None:
                    continue
                trade = flag.suggested_trades[0]
                fee = FeeCalculator.fee_for_contract(c, trade["price"], SIZE)
                if trade["action"].value == "buy":
                    pnl = SIZE * ((1.0 if hit else 0.0) - trade["price"]) - fee
                    capital = SIZE * trade["price"] + fee
                    side = "buy"
                    win = hit
                else:
                    pnl = SIZE * (trade["price"] - (1.0 if hit else 0.0)) - fee
                    capital = SIZE * (1.0 - trade["price"]) + fee   # margin at risk when short YES
                    side = "sell"
                    win = not hit
                bets.append((ev_ticker, side, pnl, capital, fee, win))

        print(f"  > Matches with model grids: {n_matches} ({n_unmatched} score events have no walk-forward record "
              f"- not in odds.csv; {n_void} voided/scalar-settled excluded) | "
              f"scoreline contracts priced: {n_rows} ({n_unparsed} unparsable)")
        if cal_model:
            cm, ch = np.array(cal_model), np.array(cal_hit)
            print(f"  > Model sanity: mean model P = {cm.mean():.4f} vs realised frequency {ch.mean():.4f} "
                  f"(log-loss {-np.mean(ch * np.log(np.clip(cm, 1e-9, 1)) + (1 - ch) * np.log(np.clip(1 - cm, 1e-9, 1))):.4f})")
        n_buy = sum(1 for b in bets if b[1] == "buy")
        n_sell = len(bets) - n_buy
        pnl_total = sum(b[2] for b in bets)
        capital_total = sum(b[3] for b in bets)
        fees3 = sum(b[4] for b in bets)
        wins3 = sum(1 for b in bets if b[5])
        print(f"  > Tradable mispricings (net of fees and {MODEL_REL_UNC:.0%} relative model error): "
              f"{len(bets)}  ({n_buy} buys / {n_sell} sells)")
        print(f"  > Win rate: {wins3}/{len(bets)} | fees ${fees3:.2f} | net PnL ${pnl_total:+.2f}")
        if capital_total > 0:
            by_match = {}
            for m, _, p, cap, _, _ in bets:
                a = by_match.setdefault(m, [0.0, 0.0])
                a[0] += p
                a[1] += cap
            ci = _bootstrap_ci(list(by_match.values()))
            roi = pnl_total / capital_total
            ci_txt = f" | 90% bootstrap CI (resampling matches): [{ci[0]:+.1%}, {ci[1]:+.1%}]" if ci else ""
            print(f"  > ROI on capital at risk: {roi:+.2%}{ci_txt}")
            if ci and ci[0] <= 0 <= ci[1]:
                print("    -> interval contains zero: no evidence of edge at this sample size")
        totals["Exact-score model"] = pnl_total

    print("\n" + "=" * 80)
    print(" SUMMARY (independent strategies; not additive capital)")
    print("=" * 80)
    for name, v in totals.items():
        print(f"  {name:<28s} ${v:+.2f}")
    print("=" * 80)


if __name__ == "__main__":
    run_historical_backtest()
