import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.elo import EloRating
from core.form_engine import FormEngine

matches = pd.read_csv("data/processed/la_liga_clean_matches.csv")
matches["Date"] = pd.to_datetime(matches["Date"])
matches = matches.sort_values("Date")

elo = EloRating(initial_rating=1500, k_factor=20)
form = FormEngine(window=5)

feature_rows = []

for _, match in matches.iterrows():
    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]
    home_goals = match["FTHG"]
    away_goals = match["FTAG"]

    home_elo_before = elo.get_rating(home_team)
    away_elo_before = elo.get_rating(away_team)

    home_form = form.get_form_features(home_team)
    away_form = form.get_form_features(away_team)

    if home_goals > away_goals:
        result = 2
    elif home_goals < away_goals:
        result = 0
    else:
        result = 1

    feature_rows.append({
        "Date": match["Date"],
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "HomeElo": home_elo_before,
        "AwayElo": away_elo_before,
        "EloDiff": home_elo_before - away_elo_before,
        "HomeFormPoints": home_form["form_points"],
        "AwayFormPoints": away_form["form_points"],
        "HomeFormGoalsFor": home_form["form_goals_for"],
        "AwayFormGoalsFor": away_form["form_goals_for"],
        "HomeFormGoalsAgainst": home_form["form_goals_against"],
        "AwayFormGoalsAgainst": away_form["form_goals_against"],
        "Result": result
    })

    elo.update_match(home_team, away_team, home_goals, away_goals)
    form.update_match(home_team, away_team, home_goals, away_goals)

features = pd.DataFrame(feature_rows)

features.to_csv("data/processed/match_features.csv", index=False)

print("Created data/processed/match_features.csv")
print(features.head())