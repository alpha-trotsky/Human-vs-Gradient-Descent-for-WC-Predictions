import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson, skellam

# Resolve results file relative to this script to avoid cwd issues
base_dir = Path(__file__).resolve().parent
results_path = base_dir / 'results' / 'results.csv'


def load_base_matches():
    """Load the raw results, select columns, and keep matches from 2000 onward."""
    match_results = pd.read_csv(results_path)
    match_results = match_results[["date", "home_team", "away_team", "home_score",
                                   "away_score", "tournament", "city", "country", "neutral"]]
    match_results["date"] = pd.to_datetime(match_results["date"])
    match_results = match_results[match_results["date"] >= "2000-01-01"]
    return match_results


# Feature 1 -- ELO ratings for home and away teams

def elo_calculator(match_results, divisor=800, fixed_k=None):
    """Walk-forward ELO ratings.

    Args:
        divisor: scale in the expected-score logistic (smaller => bigger rating gaps
                 translate to more extreme win probabilities).
        fixed_k: if given, use this K for every match; otherwise K depends on the
                 tournament (World Cup 60, Friendly 20, else 40).
    """
    teams = pd.unique(match_results[["home_team", "away_team"]].values.ravel())
    elo_ratings = {team: 1300 for team in teams}

    match_results = match_results.sort_values("date").copy()

    home_elos, away_elos = [], []

    for _, row in match_results.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_home, r_away = elo_ratings[home], elo_ratings[away]

        home_elos.append(r_home)
        away_elos.append(r_away)

        if fixed_k is not None:
            k = fixed_k
        else:
            tournament = row["tournament"]
            if tournament == "FIFA World Cup":
                k = 60
            elif tournament == "Friendly":
                k = 20
            else:
                k = 40

        e_home = 1 / (1 + 10 ** ((r_away - r_home) / divisor))
        e_away = 1 - e_home

        if row["home_score"] > row["away_score"]:
            w_home, w_away = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            w_home, w_away = 0.0, 1.0
        else:
            w_home, w_away = 0.5, 0.5

        elo_ratings[home] = r_home + k * (w_home - e_home)
        elo_ratings[away] = r_away + k * (w_away - e_away)

    match_results["home_elo"] = home_elos
    match_results["away_elo"] = away_elos

    return match_results

# Feature 2 -- Goals scored and conceded in last 10 matches (regardless of home/away)

def goals_last10(match_results):
    match_results = match_results.sort_values("date").copy()
    scored_history = {}
    conceded_history = {}
    home_scored10, home_conceded10 = [], []
    away_scored10, away_conceded10 = [], []

    for _, row in match_results.iterrows():
        home, away = row["home_team"], row["away_team"]

        # Get each team's totals from their last 10 games (all games, not just home/away specific)
        home_scored10.append(sum(scored_history.get(home, [])[-10:]) if home in scored_history else np.nan)
        home_conceded10.append(sum(conceded_history.get(home, [])[-10:]) if home in conceded_history else np.nan)
        away_scored10.append(sum(scored_history.get(away, [])[-10:]) if away in scored_history else np.nan)
        away_conceded10.append(sum(conceded_history.get(away, [])[-10:]) if away in conceded_history else np.nan)

        # Track each team's performance regardless of position (home scored/conceded applies to home team, etc.)
        scored_history.setdefault(home, []).append(row["home_score"])
        conceded_history.setdefault(home, []).append(row["away_score"])
        scored_history.setdefault(away, []).append(row["away_score"])
        conceded_history.setdefault(away, []).append(row["home_score"])

    match_results["home_scored_last10"] = home_scored10
    match_results["home_conceded_last10"] = home_conceded10
    match_results["away_scored_last10"] = away_scored10
    match_results["away_conceded_last10"] = away_conceded10
    return match_results

def build_match_features(divisor=800, fixed_k=None, since="2014-10-10"):
    """End-to-end: base matches -> ELO -> last-10 form -> filter to `since`."""
    df = load_base_matches()
    df = elo_calculator(df, divisor=divisor, fixed_k=fixed_k)
    df = goals_last10(df)
    df = df[df["date"] >= since]
    return df


if __name__ == "__main__":
    # Default pipeline: tournament-based K, divisor 800.
    match_results = build_match_features(divisor=800, fixed_k=None)
    match_results.to_csv(base_dir / 'results' / 'match_results_elo.csv', index=False)

    home_final = match_results[["home_team", "date", "home_elo"]].rename(columns={"home_team": "team", "home_elo": "elo"})
    away_final = match_results[["away_team", "date", "away_elo"]].rename(columns={"away_team": "team", "away_elo": "elo"})
    all_elos = pd.concat([home_final, away_final])
    final_elos = all_elos.sort_values("date").groupby("team")["elo"].last().sort_values(ascending=False)
    print(final_elos.head(20))