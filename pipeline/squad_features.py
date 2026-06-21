import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

SQUAD_SIZE = 25

ATTACK_POS  = {'ST', 'LW', 'RW', 'CF', 'LF', 'RF', 'LS', 'RS'}
MID_POS     = {'CM', 'CAM', 'CDM', 'RM', 'LM', 'LCM', 'RCM', 'LAM', 'RAM', 'LDM', 'RDM'}
DEF_POS     = {'CB', 'RB', 'LB', 'LWB', 'RWB', 'LCB', 'RCB', 'GK'}


def _primary_pos(pos_str):
    """Return the first (primary) position from a comma-separated position string."""
    if pd.isna(pos_str):
        return np.nan
    return pos_str.split(',')[0].strip().upper()


def compute_squad_features(df, nationality_col, overall_col, position_col, year):
    """
    For each national team (grouped by nationality_col), take the top SQUAD_SIZE players
    by overall rating and compute:
        squad_avg, squad_std      — mean/std of overall across the squad
        attack_avg                — mean overall of attack-position players
        midfield_avg              — mean overall of midfield-position players
        defence_avg               — mean overall of defence/GK-position players
        bench_avg, bench_std      — mean/std of overall for players ranked 12–23
        gk_rating                 — best (max) overall among goalkeepers in the squad
    """
    df = df[[nationality_col, overall_col, position_col]].dropna(subset=[overall_col])
    df = df.copy()
    df['primary_pos'] = df[position_col].apply(_primary_pos)

    rows = []
    for team, group in df.groupby(nationality_col):
        squad = group.nlargest(SQUAD_SIZE, overall_col).reset_index(drop=True)
        bench = squad.iloc[11:]  # players ranked 12th and beyond

        attack  = squad.loc[squad['primary_pos'].isin(ATTACK_POS), overall_col]
        mid     = squad.loc[squad['primary_pos'].isin(MID_POS),    overall_col]
        defence = squad.loc[squad['primary_pos'].isin(DEF_POS),    overall_col]
        gk      = squad.loc[squad['primary_pos'] == 'GK',          overall_col]

        rows.append({
            'team':         team,
            'year':         year,
            'squad_size':   len(squad),
            'squad_avg':    squad[overall_col].mean(),
            'squad_std':    squad[overall_col].std() if len(squad) > 1 else np.nan,
            'attack_avg':   attack.mean()  if len(attack)  > 0 else np.nan,
            'midfield_avg': mid.mean()     if len(mid)     > 0 else np.nan,
            'defence_avg':  defence.mean() if len(defence) > 0 else np.nan,
            'bench_avg':    bench[overall_col].mean() if len(bench) > 0 else np.nan,
            'bench_std':    bench[overall_col].std()  if len(bench) > 1 else np.nan,
            'gk_rating':    gk.max() if len(gk) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def repair_thin_squads(feats):
    """Make features robust to teams with too few FC players (e.g. Egypt=10, Iran=6).

    Three fixes, all fit on the well-populated teams so thin squads stop being
    out-of-distribution (a 0 bench_avg vs. a ~67 mean blows up the MLP):

      1. Dispersion -> standard error: squad_std/bench_std are divided by
         sqrt(n_players). A small squad's inflated std shrinks toward the SE of a
         normal squad. Then each is capped at its 85th percentile (top 15% clipped).
      2. Bench average imputation: thin squads (<=20 players, empty/blank bench) get
         bench_avg = squad_avg - gap, where gap is the mean (squad_avg - bench_avg)
         over full (>20) squads. I.e. assume the usual first-team-to-bench drop-off.
      3. gk_rating falls back to defence_avg when no goalkeeper is listed.
    """
    feats = feats.copy()

    # (1) standard-error transform + 85th-percentile cap
    sq_n = feats['squad_size'].clip(lower=1)
    bench_n = (feats['squad_size'] - 11).clip(lower=1)
    feats['squad_std'] = feats['squad_std'] / np.sqrt(sq_n)
    feats['bench_std'] = feats['bench_std'] / np.sqrt(bench_n)
    for col in ('squad_std', 'bench_std'):
        cap = feats[col].quantile(0.85)
        feats[col] = feats[col].clip(upper=cap)

    # (2) bench-average imputation from the typical squad->bench gap (full squads only)
    full = feats[feats['squad_size'] > 20]
    gap = (full['squad_avg'] - full['bench_avg']).mean()
    thin = (feats['squad_size'] <= 20) | feats['bench_avg'].isna()
    feats.loc[thin, 'bench_avg'] = feats.loc[thin, 'squad_avg'] - gap
    feats['bench_std'] = feats['bench_std'].fillna(feats['bench_std'].median())

    # (3) goalkeeper fallback
    feats['gk_rating'] = feats['gk_rating'].fillna(feats['defence_avg'])
    return feats

# --- FIFA 15–24 (all in fc24/male_players.csv — one snapshot per version) ---
# NB: the 5.4 GB data/fifa_15_23/male_players.csv is the SAME data with every
# weekly update snapshot, so it's redundant and crashes on load. This 92 MB file
# already holds versions 15–24 with one row per player per version. usecols keeps
# only the 4 columns we need so peak memory stays low.
print("Loading FIFA 15-24 (one file, all versions)...")
hist_raw = pd.read_csv(
    base_dir / 'data' / 'fc24' / 'male_players.csv',
    usecols=['nationality_name', 'overall', 'player_positions', 'fifa_version'],
    low_memory=False,
)
hist_features_list = []
for version in sorted(hist_raw['fifa_version'].dropna().unique()):
    sub = hist_raw[hist_raw['fifa_version'] == version].drop(columns='fifa_version')
    year = int(2000 + version)  # FIFA 15 -> 2015 ... FIFA 24 -> 2024
    feats = compute_squad_features(
        sub, 'nationality_name', 'overall', 'player_positions', year=year
    )
    hist_features_list.append(feats)
    print(f"  FIFA {int(version)} (year={year}): {len(feats)} teams")
hist_features = pd.concat(hist_features_list, ignore_index=True)

# --- FC25 (no per-position rating columns — use overall_rating + positions) ---
print("Loading FC25...")
fc25_raw = pd.read_csv(
    base_dir / 'data' / 'fc25' / 'player-data-full-2025-june.csv',
    usecols=['country_name', 'overall_rating', 'positions'],
    low_memory=False,
)
fc25_raw = fc25_raw[fc25_raw['country_name'] != 'Friendly International']
fc25_features = compute_squad_features(
    fc25_raw, 'country_name', 'overall_rating', 'positions', year=2025
)
print(f"  FC25: {len(fc25_features)} teams")

# --- FC26 ---
print("Loading FC26...")
fc26_raw = pd.read_csv(
    base_dir / 'data' / 'fc26' / 'FC26_20250921.csv',
    usecols=['nationality_name', 'overall', 'player_positions'],
    low_memory=False,
)
fc26_features = compute_squad_features(
    fc26_raw, 'nationality_name', 'overall', 'player_positions', year=2026
)
print(f"  FC26: {len(fc26_features)} teams")

# --- Combine, repair thin squads, and save ---
all_features = pd.concat([hist_features, fc25_features, fc26_features], ignore_index=True)
all_features = repair_thin_squads(all_features)
out_path = base_dir / 'results' / 'squad_features.csv'
all_features.to_csv(out_path, index=False)

print(f"\nSaved {len(all_features)} rows to {out_path}")
print(f"Years covered: {sorted(all_features['year'].unique())}")
print(f"  FIFA 15-24 (train/val): {len(hist_features)} team-years")
print(f"  FC25: {len(fc25_features)} (dataset limited), FC26: {len(fc26_features)}")
print()
print(all_features[all_features['squad_size'] >= 23].head(10).to_string(index=False))
