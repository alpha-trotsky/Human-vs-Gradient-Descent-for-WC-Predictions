#!/usr/bin/env python
"""
Backtest: asymmetric-result hypothesis on Kalshi WC 2026 total-goals markets.

Hypothesis: when the market strongly prices one team as a heavy favourite
(implied win probability > FAV_THRESHOLD = 65%), the match is more likely to
produce an asymmetric result (strong team 3+, weak team 0-1).  That pushes
TOTAL match goals higher than the market's total-goals prices may reflect.

Strategy: for every match where one team's implied win prob > FAV_THRESHOLD,
BUY over 2.5, 3.5, 4.5, 5.5, 6.5 total goals ($1 per line, per match).

Market data (odds.csv):
  - KXWCGAME-*  : WDL moneyline (3 rows per match)
  - KXWCTOTAL-* : total goals over/under (6-9 rows per match, same ticker suffix)

NOTE on "under 0.5 for the weaker team": Kalshi only offered total-match-goals
markets, NOT individual team scoring markets.  "Under 0.5 for team B" (team B
gets shut out) cannot be directly bet on from this data.  We flag matches where
team B actually scored 0 goals in the results as context, but cannot place that
specific bet.

Training: McHale-Copula on data < VAL_END (2025-10-01) -- full train+val window,
one period before the WC test set.

PnL (prediction market, $1 notional):
  BUY at implied P:  +( 1-P ) if yes,  -P if no
  Equivalently:  pnl = hit - P
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parent.parent  # project root (one up from backtest/)
sys.path.insert(0, str(_root))

from core.experiments import load_split, build_X, FULL_FEATURES, VAL_END
from core.McHale_Copula import McHaleCopulaModel
from wc_simulation import (
    load_bracket, build_team_features, make_feat_builder,
    BEST_COPULA_ALPHA, BEST_COPULA_K_MODE,
    norm, ALIAS,
)

FAV_THRESHOLD = 0.65    # market implied win prob to qualify as strong favourite
OVER_LINES    = [2.5, 3.5, 4.5, 5.5, 6.5]
ODDS_CSV      = _root / 'odds.csv'
RESULTS_CSV   = _root / 'results' / 'results.csv'
WC_START      = pd.Timestamp('2026-06-11', tz='UTC')

# -- same alias tables as backtest_odds.py ------------------------------------
_ODDS_TO_OF = {
    'czechia': 'czech republic', 'congo dr': 'dr congo',
    'korea republic': 'south korea', 'ksa': 'saudi arabia',
    'turkiye': 'turkey', 'ir iran': 'iran', 'cabo verde': 'cape verde',
}
_ODDS_TO_RES = {
    'czechia': 'czech republic', 'congo dr': 'dr congo',
    'korea republic': 'south korea', 'ksa': 'saudi arabia',
    'turkiye': 'turkey', 'ir iran': 'iran',
    'usa': 'united states', 'cabo verde': 'cabo verde',
}


def to_of(name: str) -> str:
    n = norm(name)
    return _ODDS_TO_OF.get(n, n)

def to_res(name: str) -> str:
    n = norm(name)
    return _ODDS_TO_RES.get(n, n)

def resolve_team(name: str, norm_idx: dict):
    k = to_of(name)
    if k in norm_idx: return norm_idx[k]
    return norm_idx.get(norm(name))

def parse_outcome(s: str) -> str:
    s = s.strip()
    return s[len('Reg Time: '):] if s.startswith('Reg Time: ') else s

def neutral_wdl(copula, feat_fn, iA: int, iB: int):
    ab = copula.predict_wdl(feat_fn([iA], [iB]))[0]
    ba = copula.predict_wdl(feat_fn([iB], [iA]))[0]
    return (ab[0]+ba[2])/2, (ab[1]+ba[1])/2, (ab[2]+ba[0])/2

def find_score(team_A: str, team_B: str, score_lut: dict):
    rA, rB = to_res(team_A), to_res(team_B)
    if (rA, rB) in score_lut: return score_lut[(rA, rB)]
    if (rB, rA) in score_lut:
        gh, ga = score_lut[(rB, rA)]
        return ga, gh
    return None


# -----------------------------------------------------------------------------
def main():
    SEP = '=' * 70

    print(SEP)
    print("  WC 2026 -- Asymmetric-Result / Total-Goals Backtest")
    print(f"  Trigger: market win-prob > {FAV_THRESHOLD:.0%}")
    print(f"  Action : BUY over {', '.join(str(x) for x in OVER_LINES)} total goals  ($1 each)")
    print(SEP)

    # -- 1. Bracket + features ------------------------------------------------
    print("\n[1] Loading WC bracket and pre-tournament features ...")
    groups, _, _ = load_bracket()
    team_names = sorted({t for ts in groups.values() for t in ts})
    norm_idx   = {norm(t): i for i, t in enumerate(team_names)}
    F          = build_team_features(team_names)
    feat       = make_feat_builder(F)
    print(f"    {len(team_names)} WC teams")

    # -- 2. Train copula (train + val, up to test set start) ------------------
    print(f"\n[2] Training McHale-Copula (alpha={BEST_COPULA_ALPHA}, k_mode={BEST_COPULA_K_MODE}) ...")
    print(f"    Training window: all data < {VAL_END}  (train + val combined)")
    df_tr, hg_tr, ag_tr, w_tr = load_split(end=VAL_END)
    X_tr   = build_X(df_tr, FULL_FEATURES)
    copula = McHaleCopulaModel(X_tr.shape[1], k_mode=BEST_COPULA_K_MODE)
    copula.fit(X_tr, hg_tr, ag_tr, weights=w_tr,
               alpha=BEST_COPULA_ALPHA, max_iter=500, verbose=False)
    r_h, r_a, kp = copula._unpack_extras(copula.coefficients)
    print(f"    Converged  r_home={r_h:.3f}  r_away={r_a:.3f}  k={kp}")
    print(f"    {len(X_tr):,} training matches")

    # -- 3. Load odds.csv -----------------------------------------------------
    print(f"\n[3] Loading {ODDS_CSV.name} ...")
    raw = pd.read_csv(ODDS_CSV)
    raw.columns = raw.columns.str.strip()

    # Parse KXWCGAME moneyline -> per-match implied probs
    game_rows = raw[raw['event_ticker'].str.startswith('KXWCGAME-')].copy()

    # Parse KXWCTOTAL markets -> dict: suffix -> DataFrame of over lines
    tot_rows  = raw[raw['event_ticker'].str.startswith('KXWCTOTAL-')].copy()
    tot_by_sfx = {}
    for ticker, grp in tot_rows.groupby('event_ticker'):
        sfx = ticker.replace('KXWCTOTAL-', '')
        tot_by_sfx[sfx] = grp

    # Score lookup (WC 2026 actual results)
    res = pd.read_csv(RESULTS_CSV)
    res = res[res['tournament'] == 'FIFA World Cup'].copy()
    res['date'] = pd.to_datetime(res['date'], errors='coerce')
    res = res[res['date'].dt.year == 2026]
    res['hs']  = pd.to_numeric(res['home_score'],  errors='coerce')
    res['as_'] = pd.to_numeric(res['away_score'], errors='coerce')
    res = res.dropna(subset=['hs', 'as_'])
    score_lut = {
        (norm(r['home_team']), norm(r['away_team'])): (int(r['hs']), int(r['as_']))
        for _, r in res.iterrows()
    }
    print(f"    {len(score_lut)} WC 2026 matches with actual scores")

    # -- 4. Build bets --------------------------------------------------------
    print(f"\n[4] Finding matches with one team implied win > {FAV_THRESHOLD:.0%} ...\n")

    bet_rows      = []
    match_summary = []
    no_total_mkt  = []

    for ticker, grp in game_rows.groupby('event_ticker'):
        sfx = ticker.replace('KXWCGAME-', '')
        kickoff_ts = pd.to_datetime(grp['kickoff_utc'].iloc[0], utc=True)
        if kickoff_ts < WC_START:
            continue

        title    = grp['match'].iloc[0]
        vs_parts = title.split(' vs ')
        if len(vs_parts) < 2: continue
        team_A_raw = vs_parts[0].strip()
        team_B_raw = vs_parts[1].split(':')[0].strip()

        iA = resolve_team(team_A_raw, norm_idx)
        iB = resolve_team(team_B_raw, norm_idx)
        if iA is None or iB is None:
            continue

        # Parse moneyline implied probs
        of_A  = to_of(team_A_raw)
        raw_A = norm(team_A_raw)
        mkt   = {}
        for _, row in grp.iterrows():
            out = parse_outcome(str(row['outcome']))
            imp = float(row['implied_prob'])
            hit = str(row['result']).strip().lower() == 'yes'
            n   = norm(out)
            if n == 'tie':
                mkt['draw'] = (imp, hit)
            elif n == of_A or n == raw_A:
                mkt['home'] = (imp, hit)
            else:
                mkt['away'] = (imp, hit)
        if len(mkt) != 3:
            continue

        home_impl, _ = mkt['home']
        away_impl, _ = mkt['away']
        max_win_impl = max(home_impl, away_impl)

        if max_win_impl <= FAV_THRESHOLD:
            continue   # no clear favourite -- skip

        # Identify strong/weak team
        if home_impl > away_impl:
            strong_raw, weak_raw = team_A_raw, team_B_raw
            strong_impl          = home_impl
        else:
            strong_raw, weak_raw = team_B_raw, team_A_raw
            strong_impl          = away_impl

        # Copula model WDL (neutral)
        p_home, p_draw, p_away = neutral_wdl(copula, feat, iA, iB)
        model_strong = p_home if home_impl > away_impl else p_away
        model_weak   = p_away if home_impl > away_impl else p_home

        # Actual score
        sc = find_score(team_A_raw, team_B_raw, score_lut)
        if sc:
            g_A, g_B = sc
            if home_impl > away_impl:
                g_strong, g_weak = g_A, g_B
            else:
                g_strong, g_weak = g_B, g_A
            total_goals   = g_A + g_B
            is_asymmetric = (g_strong >= 3 and g_weak <= 1)
        else:
            g_A = g_B = g_strong = g_weak = total_goals = None
            is_asymmetric = None

        kickoff_str = str(kickoff_ts.date())

        # Find total goals market (same suffix)
        if sfx not in tot_by_sfx:
            no_total_mkt.append(f"{team_A_raw} vs {team_B_raw}")
            continue
        total_grp = tot_by_sfx[sfx]

        # BUY over lines in OVER_LINES
        lines_placed = []
        for _, trow in total_grp.iterrows():
            out_raw = parse_outcome(str(trow['outcome']))
            # match e.g. "Over 2.5 goals scored"
            if not out_raw.startswith('Over '):
                continue
            try:
                line = float(out_raw.split(' ')[1])
            except (ValueError, IndexError):
                continue
            if line not in OVER_LINES:
                continue
            imp = float(trow['implied_prob'])
            hit = str(trow['result']).strip().lower() == 'yes'
            pnl = float(hit) - imp   # BUY pnl

            bet_rows.append(dict(
                match         = f"{team_A_raw} vs {team_B_raw}",
                kickoff       = kickoff_str,
                strong_team   = strong_raw,
                weak_team     = weak_raw,
                strong_impl   = round(strong_impl, 3),
                model_strong  = round(model_strong, 3),
                line          = line,
                implied       = round(imp, 3),
                hit           = hit,
                pnl           = round(pnl, 4),
                g_strong      = g_strong,
                g_weak        = g_weak,
                total_goals   = total_goals,
                is_asymmetric = is_asymmetric,
            ))
            lines_placed.append(line)

        match_summary.append(dict(
            match         = f"{team_A_raw} vs {team_B_raw}",
            kickoff       = kickoff_str,
            strong        = strong_raw,
            strong_impl   = round(strong_impl, 3),
            model_strong  = round(model_strong, 3),
            model_draw    = round(p_draw, 3),
            model_weak    = round(model_weak, 3),
            g_strong      = g_strong,
            g_weak        = g_weak,
            total_goals   = total_goals,
            is_asymmetric = is_asymmetric,
            n_lines       = len(lines_placed),
        ))

    if no_total_mkt:
        print(f"  WARNING: no total-goals market found for: {no_total_mkt}")

    df   = pd.DataFrame(bet_rows)
    df_m = pd.DataFrame(match_summary)

    if df.empty:
        print("No bets generated."); return

    # -- 5. Report ------------------------------------------------------------
    n_matches = df['match'].nunique()
    n_bets    = len(df)
    total_pnl = df['pnl'].sum()
    roi       = total_pnl / n_bets * 100
    hit_rate  = df['hit'].mean() * 100

    print(SEP)
    print("  OVERALL RESULTS")
    print(SEP)
    print(f"  Qualifying matches (one team >{FAV_THRESHOLD:.0%} implied win) : {n_matches}")
    print(f"  Total bets placed  (BUY over lines)           : {n_bets}  (${n_bets:.0f} staked)")
    print(f"  Net PnL                                        : ${total_pnl:+.2f}")
    print(f"  ROI                                            : {roi:+.1f}%")
    print(f"  Overall hit rate                               : {hit_rate:.1f}%")

    # By line
    print(f"\n  {'Over line':12} {'Bets':>5} {'Hits':>5} {'Hit%':>7} {'Net PnL':>9} {'ROI':>8} {'Avg impl':>10}")
    print("  " + "-" * 60)
    for line in sorted(df['line'].unique()):
        s = df[df['line'] == line]
        p = s['pnl'].sum()
        print(f"  Over {line:<7.1f} {len(s):5} {s['hit'].sum():5} "
              f"{s['hit'].mean()*100:6.1f}%  ${p:+7.2f}  "
              f"{p/len(s)*100:+6.1f}%  {s['implied'].mean()*100:8.1f}%")

    # Actual score breakdown (where scores are available)
    has_sc = df['g_strong'].notna()
    if has_sc.any():
        print(f"\n  Actual score breakdown ({has_sc.sum()} bets with known scores):")
        print(f"\n  {'Result type':36} {'Bets':>5} {'PnL':>8} {'ROI':>7}")
        print("  " + "-" * 55)

        b_asym = has_sc & df['is_asymmetric'].astype(bool)
        b_draw = has_sc & (df['g_strong'] == df['g_weak'])
        b_upset = has_sc & (df['g_weak'] > df['g_strong'])
        b_strong_win_not_asym = has_sc & (df['g_strong'] > df['g_weak']) & ~df['is_asymmetric'].astype(bool)

        for label, mask in [
            ("Strong team wins asymmetrically (3+, 0-1)", b_asym),
            ("Strong team wins but NOT asymmetric",        b_strong_win_not_asym),
            ("Draw",                                       b_draw),
            ("Upset (weak team wins)",                     b_upset),
        ]:
            sub = df[mask]
            if sub.empty:
                print(f"  {label:36}  {0:5}   -")
                continue
            p = sub['pnl'].sum()
            print(f"  {label:36}  {len(sub):5}  ${p:+6.2f}  {p/len(sub)*100:+5.1f}%")

    # Per-match detail
    print(f"\n  {'Match':40} {'Strong':18} {'Mkt':6} {'Model':6} {'Score':8} {'PnL':>7}")
    print("  " + "-" * 90)
    for _, m in df_m.iterrows():
        sc_str = (f"{m['g_strong']}-{m['g_weak']}"
                  if m['g_strong'] is not None else "?-?")
        asym   = " [ASYM]" if m['is_asymmetric'] else ""
        print(f"  {m['match']:40} {m['strong']:18} "
              f"{m['strong_impl']*100:5.1f}%  "
              f"{m['model_strong']*100:5.1f}%  "
              f"{sc_str:8}{asym}")

    # Top and worst matches by PnL
    match_pnl = df.groupby('match')['pnl'].sum().sort_values(ascending=False)
    print(f"\n  Top-5 matches by PnL:")
    for match, p in match_pnl.head(5).items():
        print(f"    {match:44}  ${p:+.2f}")
    print(f"\n  Worst-5 matches by PnL:")
    for match, p in match_pnl.tail(5).items():
        print(f"    {match:44}  ${p:+.2f}")

    # -- 6. Data note ---------------------------------------------------------
    print(f"\n{SEP}")
    print("  NOTE: 'Under 0.5 goals for the weaker team'")
    print("  Kalshi only offered TOTAL match goals markets, not individual team")
    print("  scoring markets.  The closest proxy is 'Over 0.5 total goals = No' ")
    print("  (i.e., 0-0 result), which is not the same as 'team B scores 0'.")
    print("  Below is how often the WEAK team actually scored 0 in these matches:")
    has_sc_m = df_m[df_m['g_weak'].notna()]
    if not has_sc_m.empty:
        n_shutout = (has_sc_m['g_weak'] == 0).sum()
        n_total_m = len(has_sc_m)
        pct = n_shutout / n_total_m * 100
        print(f"  Weak team scored 0 in {n_shutout}/{n_total_m} matches ({pct:.0f}%).")
        print(f"  Weak team scored 0-1 in "
              f"{(has_sc_m['g_weak'] <= 1).sum()}/{n_total_m} matches "
              f"({(has_sc_m['g_weak']<=1).mean()*100:.0f}%).")
    print(SEP)

    # -- 7. Save --------------------------------------------------------------
    out = _root / 'results' / 'backtest_totals_bets.csv'
    df.to_csv(out, index=False)
    print(f"  Full bet log -> {out}")
    print(SEP)


if __name__ == '__main__':
    main()

