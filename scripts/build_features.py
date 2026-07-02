import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.elo import EloRating
from core.form_engine import FormEngine
from core.home_away_elo import HomeAwayElo
from core.attack_defense import AttackDefenseEngine

matches = pd.read_csv("data/processed/la_liga_clean_matches.csv")
matches["Date"] = pd.to_datetime(matches["Date"])
matches = matches.sort_values("Date")

elo = EloRating(initial_rating=1500, k_factor=20)
home_away_elo = HomeAwayElo(initial_rating=1500, k_factor=20)
form = FormEngine(window=5)
attack_defense = AttackDefenseEngine(window=5)

feature_rows = []

for _, match in matches.iterrows():
    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]
    home_goals = match["FTHG"]
    away_goals = match["FTAG"]

    home_elo_before = elo.get_rating(home_team)
    away_elo_before = elo.get_rating(away_team)

    home_team_home_elo_before = home_away_elo.get_home_rating(home_team)
    away_team_away_elo_before = home_away_elo.get_away_rating(away_team)

    home_form = form.get_form_features(home_team)
    away_form = form.get_form_features(away_team)
    home_attack_defense = attack_defense.get_features(home_team)
    away_attack_defense = attack_defense.get_features(away_team)

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

        "HomeTeamHomeElo": home_team_home_elo_before,
        "AwayTeamAwayElo": away_team_away_elo_before,
        "HomeAwayEloDiff": home_team_home_elo_before - away_team_away_elo_before,

        "HomeFormPoints": home_form["form_points"],
        "AwayFormPoints": away_form["form_points"],
        "HomeFormGoalsFor": home_form["form_goals_for"],
        "AwayFormGoalsFor": away_form["form_goals_for"],
        "HomeFormGoalsAgainst": home_form["form_goals_against"],
        "AwayFormGoalsAgainst": away_form["form_goals_against"],
        "HomeAttackRating": home_attack_defense["attack_rating"],
"AwayAttackRating": away_attack_defense["attack_rating"],
"HomeDefenseRating": home_attack_defense["defense_rating"],
"AwayDefenseRating": away_attack_defense["defense_rating"],
"HomeGoalDiffRating": home_attack_defense["goal_diff_rating"],
"AwayGoalDiffRating": away_attack_defense["goal_diff_rating"],
"AttackRatingDiff": home_attack_defense["attack_rating"] - away_attack_defense["attack_rating"],
"DefenseRatingDiff": home_attack_defense["defense_rating"] - away_attack_defense["defense_rating"],
"GoalDiffRatingDiff": home_attack_defense["goal_diff_rating"] - away_attack_defense["goal_diff_rating"],

        "Result": result
    })

    elo.update_match(home_team, away_team, home_goals, away_goals)
    home_away_elo.update_match(home_team, away_team, home_goals, away_goals)
    form.update_match(home_team, away_team, home_goals, away_goals)
    attack_defense.update_match(home_team, away_team, home_goals, away_goals)

features = pd.DataFrame(feature_rows)

features.to_csv("data/processed/match_features.csv", index=False)

print("Created data/processed/match_features.csv")
print(features.head())
print("\nColumns:")
print(features.columns.tolist())
print("\nRows:", len(features))