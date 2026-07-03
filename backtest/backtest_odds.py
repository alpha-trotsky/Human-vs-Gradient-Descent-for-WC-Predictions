#!/usr/bin/env python
"""
Backtest: McHale-Copula WDL predictions vs Kalshi WC 2026 prediction-market odds.

Model  : NegBin marginals + Frank copula, trained on data < TRAIN_END (2024-01-01)
         using BEST_COPULA_ALPHA / BEST_COPULA_K_MODE from wc_simulation.py.
Features: pre-tournament ELO + FC26 squad stats, frozen at WC_CUTOFF.
          Neutral prediction: average of both home/away feature orientations.

Strategy (prediction-market, $1 notional per contract):
  BUY  when model_prob > implied_prob + THRESHOLD  -> market undervalues outcome
  SELL when implied_prob > model_prob + THRESHOLD  -> market overvalues outcome
  No trade when |diff| <= THRESHOLD (2 cents)

PnL per $1 contract (standard prediction-market accounting):
  BUY  at price P -> +( 1 - P ) if outcome occurs,  -P otherwise
  SELL at price P -> +P if outcome does NOT occur,  -(1 - P) otherwise
  Equivalently: pnl_buy = hit - P,  pnl_sell = P - hit  (hit in {0, 1})

Score-based buckets use actual goals from results.csv (only WC matches with
scores filled in qualify; NA rows are excluded from those buckets).

Usage:
    python backtest_odds.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parent.parent  # project root (one up from backtest/)
sys.path.insert(0, str(_root))

from core.experiments import load_split, build_X, FULL_FEATURES, TRAIN_END, VAL_END
from core.McHale_Copula import McHaleCopulaModel
from wc_simulation import (
    load_bracket, build_team_features, make_feat_builder,
    BEST_COPULA_ALPHA, BEST_COPULA_K_MODE,
    norm, ALIAS,
)

THRESHOLD   = 0.02
ODDS_CSV    = _root / 'odds.csv'
RESULTS_CSV = _root / 'results' / 'results.csv'

# -- Name alias tables ---------------------------------------------------------
# Kalshi (normed) -> openfootball (normed) -- only entries that differ
_ODDS_TO_OF = {
    'czechia':        'czech republic',
    'congo dr':       'dr congo',
    'korea republic': 'south korea',
    'ksa':            'saudi arabia',
    'turkiye':        'turkey',
    'ir iran':        'iran',
    'cabo verde':     'cape verde',
}
# Kalshi (normed) -> results.csv (normed) -- for score lookup
_ODDS_TO_RES = {
    'czechia':        'czech republic',
    'congo dr':       'dr congo',
    'korea republic': 'south korea',
    'ksa':            'saudi arabia',
    'turkiye':        'turkey',
    'ir iran':        'iran',
    'usa':            'united states',
    'cabo verde':     'cabo verde',
}


def to_of(odds_name: str) -> str:
    """Odds name -> normalized openfootball key (for feature/team-index lookup)."""
    n = norm(odds_name)
    return _ODDS_TO_OF.get(n, n)


def to_res(odds_name: str) -> str:
    """Odds name -> normalized results.csv key (for score lookup)."""
    n = norm(odds_name)
    return _ODDS_TO_RES.get(n, n)


def parse_outcome(s: str) -> str:
    """Strip 'Reg Time: ' prefix; return bare team name or 'tie'."""
    s = s.strip()
    return s[len('Reg Time: '):] if s.startswith('Reg Time: ') else s


def resolve_team(odds_name: str, norm_idx: dict):
    """Find team index by trying OF alias, then direct norm."""
    k = to_of(odds_name)
    if k in norm_idx:
        return norm_idx[k]
    k2 = norm(odds_name)            # fallback: direct (catches e.g. 'usa' if OF uses 'usa')
    return norm_idx.get(k2)


def neutral_wdl(copula, feat_fn, iA: int, iB: int):
    """
    Neutral WDL probabilities: average of both home/away orientations.
    Returns (p_A_wins, p_draw, p_B_wins) as floats.
    """
    ab = copula.predict_wdl(feat_fn([iA], [iB]))[0]   # A home: [win, draw, lose]
    ba = copula.predict_wdl(feat_fn([iB], [iA]))[0]   # B home: [win, draw, lose]
    return (ab[0] + ba[2]) / 2, (ab[1] + ba[1]) / 2, (ab[2] + ba[0]) / 2


def find_score(team_A: str, team_B: str, score_lut: dict):
    """
    Return (goals_A, goals_B) from results.csv, or None if not found.
    Tries both orientations (results.csv home/away may differ from odds.csv).
    """
    rA, rB = to_res(team_A), to_res(team_B)
    if (rA, rB) in score_lut:
        return score_lut[(rA, rB)]
    if (rB, rA) in score_lut:
        gh, ga = score_lut[(rB, rA)]
        return ga, gh           # swap: rB was home, so rA gets the away score
    return None


# -----------------------------------------------------------------------------
def bucket_report(label: str, mask: pd.Series, df: pd.DataFrame, note: str = ''):
    SEP2 = '  ' + '-' * 60
    sub = df[mask]
    if sub.empty:
        print(f"\n  [ {label} ]  ->  0 bets{(' - ' + note) if note else ''}")
        return
    total_pnl = sub['pnl'].sum()
    roi = total_pnl / len(sub) * 100
    print(f"\n  [ {label} ]"
          + (f"  ({note})" if note else ''))
    print(f"  {len(sub)} bets   net PnL ${total_pnl:+.2f}   ROI {roi:+.1f}%")
    print(SEP2)
    for act in ('BUY', 'SELL'):
        s = sub[sub['action'] == act]
        if s.empty:
            continue
        p = s['pnl'].sum()
        print(f"    {act:4} {len(s):3} bets  ${p:+.2f}  "
              f"{p/len(s)*100:+.1f}% ROI  "
              f"hit {s['hit'].mean()*100:.0f}%")


# -----------------------------------------------------------------------------
def main():
    SEP = '=' * 70

    print(SEP)
    print("  WC 2026  -  McHale-Copula vs Kalshi  -  Backtest")
    print(f"  Strategy: BUY model>implied+{THRESHOLD:.2f}  |  "
          f"SELL implied>model+{THRESHOLD:.2f}  |  $1/contract")
    print(SEP)

    # -- 1. Bracket + pre-tournament features ---------------------------------
    print("\n[1] Loading WC bracket and pre-tournament features ...")
    groups, _, _ = load_bracket()
    team_names  = sorted({t for ts in groups.values() for t in ts})
    norm_idx    = {norm(t): i for i, t in enumerate(team_names)}
    F           = build_team_features(team_names)
    feat        = make_feat_builder(F)
    print(f"    {len(team_names)} WC teams | features frozen at WC_CUTOFF")

    # -- 2. Train McHale copula on training split ------------------------------
    print(f"\n[2] Training McHale-Copula  "
          f"alpha={BEST_COPULA_ALPHA}  k_mode={BEST_COPULA_K_MODE} ...")
    df_tr, hg_tr, ag_tr, w_tr = load_split(end=VAL_END)  # train+val (< 2025-10-01)
    X_tr   = build_X(df_tr, FULL_FEATURES)
    copula = McHaleCopulaModel(X_tr.shape[1], k_mode=BEST_COPULA_K_MODE)
    copula.fit(X_tr, hg_tr, ag_tr, weights=w_tr,
               alpha=BEST_COPULA_ALPHA, max_iter=500, verbose=False)
    r_h, r_a, kp = copula._unpack_extras(copula.coefficients)
    kp_str = '  '.join(f'{v:+.4f}' for v in kp)
    print(f"    NLL converged  r_home={r_h:.3f}  r_away={r_a:.3f}  k=[{kp_str}]")
    print(f"    Train set: {len(X_tr):,} matches  (< {VAL_END})")

    # -- 3. Parse odds.csv -----------------------------------------------------
    print(f"\n[3] Parsing {ODDS_CSV.name} ...")
    raw = pd.read_csv(ODDS_CSV)
    raw.columns = raw.columns.str.strip()

    # Score lookup from results.csv (only WC 2026 rows with actual scores)
    res = pd.read_csv(RESULTS_CSV)
    res = res[res['tournament'] == 'FIFA World Cup'].copy()
    res['date'] = pd.to_datetime(res['date'], errors='coerce')
    res = res[res['date'].dt.year == 2026]
    res['hs'] = pd.to_numeric(res['home_score'], errors='coerce')
    res['as_'] = pd.to_numeric(res['away_score'], errors='coerce')
    res = res.dropna(subset=['hs', 'as_'])
    score_lut = {
        (norm(r['home_team']), norm(r['away_team'])): (int(r['hs']), int(r['as_']))
        for _, r in res.iterrows()
    }
    print(f"    {len(score_lut)} WC 2026 matches have actual scores in results.csv")

    # WC 2026 start date - filter out pre-tournament friendlies in odds.csv
    WC_START = pd.Timestamp('2026-06-11', tz='UTC')

    # -- 4. Build bets ---------------------------------------------------------
    print(f"\n[4] Evaluating matches (threshold = {THRESHOLD:.2f}, WC matches only)\n")
    bet_rows  = []
    skipped   = []

    for ticker, grp in raw.groupby('event_ticker'):
        # Skip non-moneyline markets (total goals, etc.)
        if not ticker.startswith('KXWCGAME-'):
            continue
        kickoff_ts = pd.to_datetime(grp['kickoff_utc'].iloc[0], utc=True)
        if kickoff_ts < WC_START:
            continue   # skip pre-tournament friendlies / qualifiers
        title    = grp['match'].iloc[0]
        vs_parts = title.split(' vs ')
        if len(vs_parts) < 2:
            continue
        team_A_raw = vs_parts[0].strip()
        team_B_raw = vs_parts[1].split(':')[0].strip()

        iA = resolve_team(team_A_raw, norm_idx)
        iB = resolve_team(team_B_raw, norm_idx)
        if iA is None or iB is None:
            skipped.append(f"{team_A_raw} vs {team_B_raw}")
            continue

        p_home, p_draw, p_away = neutral_wdl(copula, feat, iA, iB)
        model_p = {'home': p_home, 'draw': p_draw, 'away': p_away}

        # Parse three outcome rows -> mkt dict
        # Match on openfootball alias OR the raw normalized name (handles e.g. "Korea Republic")
        of_A  = to_of(team_A_raw)     # openfootball normalized name of team A
        raw_A = norm(team_A_raw)      # direct normalized name (pre-alias)
        mkt = {}
        for _, row in grp.iterrows():
            out_name = parse_outcome(str(row['outcome']))
            implied  = float(row['implied_prob'])
            hit      = str(row['result']).strip().lower() == 'yes'
            norm_out = norm(out_name)
            if norm_out in ('tie',):
                mkt['draw'] = (implied, hit)
            elif norm_out == of_A or norm_out == raw_A:
                mkt['home'] = (implied, hit)
            else:
                mkt['away'] = (implied, hit)

        if len(mkt) != 3:
            skipped.append(f"{team_A_raw} vs {team_B_raw} (could not parse 3 outcomes)")
            continue

        # Actual score for bucket analysis
        sc  = find_score(team_A_raw, team_B_raw, score_lut)
        g_A = sc[0] if sc else None
        g_B = sc[1] if sc else None

        kickoff = str(kickoff_ts.date())

        for outcome in ('home', 'draw', 'away'):
            imp, hit = mkt[outcome]
            mdl      = model_p[outcome]
            diff     = mdl - imp      # >0 -> model > market -> BUY; <0 -> market > model -> SELL
            if abs(diff) <= THRESHOLD:
                continue
            action = 'BUY' if diff > 0 else 'SELL'
            # BUY:  pnl = hit - implied;  SELL: pnl = implied - hit
            pnl = (float(hit) - imp) if action == 'BUY' else (imp - float(hit))

            bet_rows.append(dict(
                match   = f"{team_A_raw} vs {team_B_raw}",
                kickoff = kickoff,
                outcome = outcome,
                action  = action,
                model   = round(mdl, 4),
                implied = round(imp, 4),
                diff    = round(diff, 4),
                hit     = hit,
                pnl     = round(pnl, 4),
                g_A     = g_A,
                g_B     = g_B,
            ))

    if skipped:
        print(f"  Skipped (unresolvable teams): {skipped}\n")

    df = pd.DataFrame(bet_rows)
    if df.empty:
        print("No bets generated -- check odds.csv or threshold.")
        return

    # -- 5. Print report -------------------------------------------------------
    n_bets    = len(df)
    n_matches = df['match'].nunique()
    total_pnl = df['pnl'].sum()
    roi       = total_pnl / n_bets * 100

    print(SEP)
    print("  OVERALL RESULTS")
    print(SEP)
    print(f"  Matches in odds.csv       :  {n_matches}")
    print(f"  Bets placed               :  {n_bets}  (${n_bets:.0f} total staked at $1/contract)")
    print(f"  Net PnL                   :  ${total_pnl:+.2f}")
    print(f"  ROI on staked             :  {roi:+.1f}%")
    print(f"  Overall hit rate          :  {df['hit'].mean()*100:.1f}%")

    print(f"\n  {'Action':6}  {'Bets':>5}  {'Net PnL':>9}  {'ROI':>8}  {'Hit%':>7}")
    print("  " + '-' * 44)
    for act in ('BUY', 'SELL'):
        s = df[df['action'] == act]
        if s.empty:
            continue
        p = s['pnl'].sum()
        print(f"  {act:6}  {len(s):5}  ${p:+8.2f}  {p/len(s)*100:+7.1f}%  {s['hit'].mean()*100:6.1f}%")

    print(f"\n  {'Outcome':6}  {'Bets':>5}  {'Net PnL':>9}  {'ROI':>8}  {'Hit%':>7}")
    print("  " + '-' * 44)
    for oc in ('home', 'draw', 'away'):
        s = df[df['outcome'] == oc]
        if s.empty:
            continue
        p = s['pnl'].sum()
        print(f"  {oc:6}  {len(s):5}  ${p:+8.2f}  {p/len(s)*100:+7.1f}%  {s['hit'].mean()*100:6.1f}%")

    # Per-match breakdown
    print(f"\n  {'Match':42}  {'Bets':>4}  {'PnL':>7}")
    print("  " + '-' * 58)
    for match, grp in df.groupby('match', sort=False):
        print(f"  {match:42}  {len(grp):4}  ${grp['pnl'].sum():+.2f}")

    # Top / bottom bets
    cols_show = ['match', 'outcome', 'action', 'model', 'implied', 'diff', 'hit', 'pnl']
    print(f"\n  Top-5 bets by PnL:")
    print(df.nlargest(5, 'pnl')[cols_show].to_string(index=False))
    print(f"\n  Worst-5 bets by PnL:")
    print(df.nsmallest(5, 'pnl')[cols_show].to_string(index=False))

    # -- 6. Bucket analysis ----------------------------------------------------
    print(f"\n{SEP}")
    print("  BUCKET ANALYSIS  ")
    print(SEP)

    has_sc = df['g_A'].notna()
    n_with_scores = df[has_sc]['match'].nunique()
    n_score_bets  = has_sc.sum()

    print(f"\n  Bets with actual scores available: "
          f"{n_score_bets} bets from {n_with_scores} matches")

    # Bucket 1: both teams score > 3
    b1 = has_sc & (df['g_A'] > 3) & (df['g_B'] > 3)
    bucket_report(
        "Both teams score > 3 goals  (each >= 4)", b1, df,
        note=f"{b1.sum()} qualifying bets"
    )

    # Bucket 2: one team <=1, other >=3
    b2 = has_sc & (
        ((df['g_A'] <= 1) & (df['g_B'] >= 3)) |
        ((df['g_B'] <= 1) & (df['g_A'] >= 3))
    )
    bucket_report(
        "Asymmetric  (min goals <= 1, max goals >= 3)", b2, df,
        note=f"{b2.sum()} qualifying bets"
    )

    # Bucket 3: BUY with implied > 80%
    b3 = (df['action'] == 'BUY') & (df['implied'] > 0.80)
    bucket_report("BUY  with implied > 80%", b3, df, note=f"{b3.sum()} bets")

    # Bucket 4: SELL with implied > 80%
    b4 = (df['action'] == 'SELL') & (df['implied'] > 0.80)
    bucket_report("SELL with implied > 80%", b4, df, note=f"{b4.sum()} bets")

    # -- 7. Save bet log -------------------------------------------------------
    out = _root / 'results' / 'backtest_wdl_bets.csv'
    df.to_csv(out, index=False)
    print(f"\n{SEP}")
    print(f"  Full bet log -> {out}")
    print(SEP)


if __name__ == '__main__':
    main()



