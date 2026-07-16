import pandas as pd
from pathlib import Path
from collections import defaultdict, deque


# =========================
# Config
# =========================

INPUT_FILE = Path("data/processed/multi_league_matches.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "multi_league_feature_store_v1.csv"

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE_ELO = 50
FORM_WINDOW = 5


# =========================
# Elo Helpers
# =========================

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(home_elo: float, away_elo: float, home_goals: int, away_goals: int):
    home_adjusted_elo = home_elo + HOME_ADVANTAGE_ELO

    home_expected = expected_score(home_adjusted_elo, away_elo)
    away_expected = 1 - home_expected

    if home_goals > away_goals:
        home_actual = 1
        away_actual = 0
    elif home_goals < away_goals:
        home_actual = 0
        away_actual = 1
    else:
        home_actual = 0.5
        away_actual = 0.5

    new_home_elo = home_elo + K_FACTOR * (home_actual - home_expected)
    new_away_elo = away_elo + K_FACTOR * (away_actual - away_expected)

    return new_home_elo, new_away_elo


# =========================
# Rolling Feature Helpers
# =========================

def safe_average(values, default_value=0):
    if len(values) == 0:
        return default_value

    return sum(values) / len(values)


def get_recent_points(result_history):
    return safe_average(result_history, default_value=1)


def get_recent_goals(goal_history):
    return safe_average(goal_history, default_value=1.25)


def result_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    else:
        return 0


# =========================
# Main Feature Builder
# =========================

def build_features_for_league(league_df: pd.DataFrame) -> pd.DataFrame:
    league_df = league_df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    team_elo = defaultdict(lambda: INITIAL_ELO)
    team_home_elo = defaultdict(lambda: INITIAL_ELO)
    team_away_elo = defaultdict(lambda: INITIAL_ELO)

    team_points_form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    team_goals_for_form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    team_goals_against_form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    team_goal_diff_form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))

    feature_rows = []

    for _, match in league_df.iterrows():
        league = match["League"]
        league_code = match["LeagueCode"]
        season_code = match["SeasonCode"]
        season = match["Season"]
        date = match["Date"]

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        home_goals = int(match["FTHG"])
        away_goals = int(match["FTAG"])

        home_elo = team_elo[home_team]
        away_elo = team_elo[away_team]

        home_team_home_elo = team_home_elo[home_team]
        away_team_away_elo = team_away_elo[away_team]

        home_form_points = get_recent_points(team_points_form[home_team])
        away_form_points = get_recent_points(team_points_form[away_team])

        home_form_goals_for = get_recent_goals(team_goals_for_form[home_team])
        away_form_goals_for = get_recent_goals(team_goals_for_form[away_team])

        home_form_goals_against = get_recent_goals(team_goals_against_form[home_team])
        away_form_goals_against = get_recent_goals(team_goals_against_form[away_team])

        home_attack_rating = get_recent_goals(team_goals_for_form[home_team])
        away_attack_rating = get_recent_goals(team_goals_for_form[away_team])

        home_defense_rating = get_recent_goals(team_goals_against_form[home_team])
        away_defense_rating = get_recent_goals(team_goals_against_form[away_team])

        home_goal_diff_rating = safe_average(team_goal_diff_form[home_team], default_value=0)
        away_goal_diff_rating = safe_average(team_goal_diff_form[away_team], default_value=0)

        row = {
            "League": league,
            "LeagueCode": league_code,
            "SeasonCode": season_code,
            "Season": season,
            "Date": date,
            "HomeTeam": home_team,
            "AwayTeam": away_team,

            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": match["FTR"],
            "Result": match["Result"],
            "ResultLabel": match["ResultLabel"],

            "B365H": match["B365H"],
            "B365D": match["B365D"],
            "B365A": match["B365A"],

            "HomeElo": home_elo,
            "AwayElo": away_elo,
            "EloDiff": home_elo + HOME_ADVANTAGE_ELO - away_elo,

            "HomeTeamHomeElo": home_team_home_elo,
            "AwayTeamAwayElo": away_team_away_elo,
            "HomeAwayEloDiff": home_team_home_elo + HOME_ADVANTAGE_ELO - away_team_away_elo,

            "HomeFormPoints": home_form_points,
            "AwayFormPoints": away_form_points,
            "HomeFormGoalsFor": home_form_goals_for,
            "AwayFormGoalsFor": away_form_goals_for,
            "HomeFormGoalsAgainst": home_form_goals_against,
            "AwayFormGoalsAgainst": away_form_goals_against,

            "HomeAdvantage": 1,

            "FormPointDiff": home_form_points - away_form_points,
            "FormGoalsForDiff": home_form_goals_for - away_form_goals_for,
            "FormGoalsAgainstDiff": home_form_goals_against - away_form_goals_against,

            "HomeAttackRating": home_attack_rating,
            "AwayAttackRating": away_attack_rating,
            "HomeDefenseRating": home_defense_rating,
            "AwayDefenseRating": away_defense_rating,
            "HomeGoalDiffRating": home_goal_diff_rating,
            "AwayGoalDiffRating": away_goal_diff_rating,

            "AttackRatingDiff": home_attack_rating - away_attack_rating,
            "DefenseRatingDiff": away_defense_rating - home_defense_rating,
            "GoalDiffRatingDiff": home_goal_diff_rating - away_goal_diff_rating,
        }

        optional_match_stats = [
            "HS", "AS", "HST", "AST", "HC", "AC",
            "HF", "AF", "HY", "AY", "HR", "AR",
        ]

        for col in optional_match_stats:
            if col in match.index:
                row[col] = match[col]

        feature_rows.append(row)

        # Update overall Elo after match
        new_home_elo, new_away_elo = update_elo(
            home_elo=home_elo,
            away_elo=away_elo,
            home_goals=home_goals,
            away_goals=away_goals,
        )

        team_elo[home_team] = new_home_elo
        team_elo[away_team] = new_away_elo

        # Update venue-specific Elo after match
        new_home_venue_elo, _ = update_elo(
            home_elo=home_team_home_elo,
            away_elo=away_team_away_elo,
            home_goals=home_goals,
            away_goals=away_goals,
        )

        _, new_away_venue_elo = update_elo(
            home_elo=home_team_home_elo,
            away_elo=away_team_away_elo,
            home_goals=home_goals,
            away_goals=away_goals,
        )

        team_home_elo[home_team] = new_home_venue_elo
        team_away_elo[away_team] = new_away_venue_elo

        # Update rolling form after match
        home_points = result_points(home_goals, away_goals)
        away_points = result_points(away_goals, home_goals)

        team_points_form[home_team].append(home_points)
        team_points_form[away_team].append(away_points)

        team_goals_for_form[home_team].append(home_goals)
        team_goals_for_form[away_team].append(away_goals)

        team_goals_against_form[home_team].append(away_goals)
        team_goals_against_form[away_team].append(home_goals)

        team_goal_diff_form[home_team].append(home_goals - away_goals)
        team_goal_diff_form[away_team].append(away_goals - home_goals)

    return pd.DataFrame(feature_rows)


def main():
    print("===================================")
    print("Soccer Edge Lab - Multi-League Feature Builder")
    print("===================================")

    if not INPUT_FILE.exists():
        print(f"Missing input file: {INPUT_FILE}")
        print("Run this first:")
        print("python scripts/build_multi_league_dataset.py")
        raise SystemExit

    matches = pd.read_csv(INPUT_FILE)

    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")

    numeric_columns = [
        "Season",
        "FTHG",
        "FTAG",
        "Result",
        "B365H",
        "B365D",
        "B365A",
    ]

    for col in numeric_columns:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    matches = matches.dropna(
        subset=[
            "League",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "Result",
            "B365H",
            "B365D",
            "B365A",
        ]
    ).copy()

    matches["Season"] = matches["Season"].astype(int)
    matches["FTHG"] = matches["FTHG"].astype(int)
    matches["FTAG"] = matches["FTAG"].astype(int)
    matches["Result"] = matches["Result"].astype(int)

    feature_datasets = []

    for league_name, league_df in matches.groupby("League"):
        print(f"Building features for: {league_name} | Matches: {len(league_df):,}")

        league_features = build_features_for_league(league_df)
        feature_datasets.append(league_features)

    features = pd.concat(feature_datasets, ignore_index=True)

    features = features.sort_values(
        ["Date", "League", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    features.to_csv(OUTPUT_FILE, index=False)

    print("\nMulti-league feature store created.")
    print(f"Rows: {len(features):,}")
    print(f"Leagues: {features['League'].nunique()}")
    print(f"Seasons: {features['Season'].nunique()}")
    print(f"Date range: {features['Date'].min()} to {features['Date'].max()}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nRows by league:")
    print(features.groupby("League").size().sort_values(ascending=False).to_string())

    print("\nRows by season:")
    print(features.groupby("Season").size().sort_index().to_string())

    print("\nFeature columns:")
    feature_columns = [
        "HomeElo",
        "AwayElo",
        "EloDiff",
        "HomeTeamHomeElo",
        "AwayTeamAwayElo",
        "HomeAwayEloDiff",
        "HomeFormPoints",
        "AwayFormPoints",
        "HomeFormGoalsFor",
        "AwayFormGoalsFor",
        "HomeFormGoalsAgainst",
        "AwayFormGoalsAgainst",
        "HomeAdvantage",
        "FormPointDiff",
        "FormGoalsForDiff",
        "FormGoalsAgainstDiff",
        "HomeAttackRating",
        "AwayAttackRating",
        "HomeDefenseRating",
        "AwayDefenseRating",
        "HomeGoalDiffRating",
        "AwayGoalDiffRating",
        "AttackRatingDiff",
        "DefenseRatingDiff",
        "GoalDiffRatingDiff",
    ]

    for col in feature_columns:
        print(f"- {col}")

    print("\nSample rows:")
    print(features.head(10).to_string(index=False))


if __name__ == "__main__":
    main()