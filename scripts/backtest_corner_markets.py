import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


# =========================
# Config
# =========================

INPUT_FILE = Path("data/processed/multi_league_feature_store_v1.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "corner_market_summary_v1.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "corner_market_predictions_v1.csv"

ROLLING_WINDOW = 5

FEATURE_COLUMNS = [
    # Existing pre-match team strength features
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

    # New pre-match rolling corner features
    "HomeRecentCornersFor",
    "AwayRecentCornersFor",
    "HomeRecentCornersAgainst",
    "AwayRecentCornersAgainst",
    "HomeRecentTotalCorners",
    "AwayRecentTotalCorners",
    "HomeCornerForDiff",
    "AwayCornerForDiff",
    "CornerForDiff",
    "CornerAgainstDiff",
    "TotalCornerTempo",
]

CORNER_MARKETS = {
    "TotalCorners_Over_8_5": {
        "target_type": "total",
        "line": 8.5,
    },
    "TotalCorners_Over_9_5": {
        "target_type": "total",
        "line": 9.5,
    },
    "TotalCorners_Over_10_5": {
        "target_type": "total",
        "line": 10.5,
    },
    "HomeCorners_Over_4_5": {
        "target_type": "home",
        "line": 4.5,
    },
    "HomeCorners_Over_5_5": {
        "target_type": "home",
        "line": 5.5,
    },
    "AwayCorners_Over_3_5": {
        "target_type": "away",
        "line": 3.5,
    },
    "AwayCorners_Over_4_5": {
        "target_type": "away",
        "line": 4.5,
    },
}


# =========================
# Helpers
# =========================

def safe_average(values, default_value=0):
    if len(values) == 0:
        return default_value

    return sum(values) / len(values)


def fair_odds(probability: float) -> float:
    if probability <= 0:
        return 999.0

    return 1 / probability


def add_rolling_corner_features_for_league(league_df: pd.DataFrame) -> pd.DataFrame:
    league_df = league_df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    corners_for_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    corners_against_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    total_corners_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))

    rows = []

    for _, match in league_df.iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        home_corners = match["HC"]
        away_corners = match["AC"]
        total_corners = home_corners + away_corners

        home_recent_corners_for = safe_average(corners_for_history[home_team], default_value=5.0)
        away_recent_corners_for = safe_average(corners_for_history[away_team], default_value=4.5)

        home_recent_corners_against = safe_average(corners_against_history[home_team], default_value=4.5)
        away_recent_corners_against = safe_average(corners_against_history[away_team], default_value=5.0)

        home_recent_total_corners = safe_average(total_corners_history[home_team], default_value=9.5)
        away_recent_total_corners = safe_average(total_corners_history[away_team], default_value=9.5)

        row = match.to_dict()

        row["TotalCorners"] = total_corners
        row["HomeCorners"] = home_corners
        row["AwayCorners"] = away_corners

        row["HomeRecentCornersFor"] = home_recent_corners_for
        row["AwayRecentCornersFor"] = away_recent_corners_for
        row["HomeRecentCornersAgainst"] = home_recent_corners_against
        row["AwayRecentCornersAgainst"] = away_recent_corners_against
        row["HomeRecentTotalCorners"] = home_recent_total_corners
        row["AwayRecentTotalCorners"] = away_recent_total_corners

        row["HomeCornerForDiff"] = home_recent_corners_for - away_recent_corners_against
        row["AwayCornerForDiff"] = away_recent_corners_for - home_recent_corners_against
        row["CornerForDiff"] = home_recent_corners_for - away_recent_corners_for
        row["CornerAgainstDiff"] = away_recent_corners_against - home_recent_corners_against
        row["TotalCornerTempo"] = (home_recent_total_corners + away_recent_total_corners) / 2

        rows.append(row)

        # Update histories after the match only
        corners_for_history[home_team].append(home_corners)
        corners_for_history[away_team].append(away_corners)

        corners_against_history[home_team].append(away_corners)
        corners_against_history[away_team].append(home_corners)

        total_corners_history[home_team].append(total_corners)
        total_corners_history[away_team].append(total_corners)

    return pd.DataFrame(rows)


def create_corner_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for market_name, config in CORNER_MARKETS.items():
        line = config["line"]
        target_type = config["target_type"]

        if target_type == "total":
            df[market_name] = (df["TotalCorners"] > line).astype(int)
        elif target_type == "home":
            df[market_name] = (df["HomeCorners"] > line).astype(int)
        elif target_type == "away":
            df[market_name] = (df["AwayCorners"] > line).astype(int)
        else:
            raise ValueError(f"Unknown target type: {target_type}")

    return df


def calculate_market_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for market_name, market_df in predictions_df.groupby("Market"):
        accuracy = accuracy_score(market_df["Actual"], market_df["Prediction"])
        loss = log_loss(market_df["Actual"], market_df["Probability"], labels=[0, 1])
        brier = brier_score_loss(market_df["Actual"], market_df["Probability"])

        actual_hit_rate = market_df["Actual"].mean()
        avg_probability = market_df["Probability"].mean()
        avg_fair_odds = market_df["FairOdds"].mean()

        high_confidence = market_df[market_df["Probability"] >= 0.55].copy()

        if len(high_confidence) > 0:
            high_confidence_hit_rate = high_confidence["Actual"].mean()
            high_confidence_count = len(high_confidence)
            high_confidence_avg_prob = high_confidence["Probability"].mean()
            high_confidence_avg_fair_odds = high_confidence["FairOdds"].mean()
        else:
            high_confidence_hit_rate = 0
            high_confidence_count = 0
            high_confidence_avg_prob = 0
            high_confidence_avg_fair_odds = 0

        rows.append({
            "Market": market_name,
            "Predictions": len(market_df),
            "Accuracy": accuracy,
            "LogLoss": loss,
            "BrierScore": brier,
            "ActualHitRate": actual_hit_rate,
            "AverageModelProbability": avg_probability,
            "AverageFairOdds": avg_fair_odds,
            "HighConfidencePicks": high_confidence_count,
            "HighConfidenceHitRate": high_confidence_hit_rate,
            "HighConfidenceAverageProbability": high_confidence_avg_prob,
            "HighConfidenceAverageFairOdds": high_confidence_avg_fair_odds,
        })

    return pd.DataFrame(rows).sort_values(
        ["HighConfidenceHitRate", "LogLoss"],
        ascending=[False, True]
    )


# =========================
# Main
# =========================

def main():
    print("===================================")
    print("Soccer Edge Lab - Corner Market Model v1")
    print("===================================")

    if not INPUT_FILE.exists():
        print(f"Missing input file: {INPUT_FILE}")
        print("Run this first:")
        print("python scripts/build_multi_league_features.py")
        raise SystemExit

    data = pd.read_csv(INPUT_FILE)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    required_columns = [
        "League",
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HC",
        "AC",
    ] + FEATURE_COLUMNS[:25]

    data = data.dropna(subset=required_columns).copy()

    data["Season"] = pd.to_numeric(data["Season"], errors="coerce")
    data["HC"] = pd.to_numeric(data["HC"], errors="coerce")
    data["AC"] = pd.to_numeric(data["AC"], errors="coerce")

    data = data.dropna(subset=["Season", "HC", "AC"]).copy()

    data["Season"] = data["Season"].astype(int)
    data["HC"] = data["HC"].astype(int)
    data["AC"] = data["AC"].astype(int)

    print("\nInput dataset:")
    print(f"Matches: {len(data):,}")
    print(f"Leagues: {data['League'].nunique()}")
    print(f"Seasons: {data['Season'].nunique()}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

    feature_datasets = []

    for league_name, league_df in data.groupby("League"):
        print(f"Adding rolling corner features for: {league_name} | Matches: {len(league_df):,}")
        feature_datasets.append(add_rolling_corner_features_for_league(league_df))

    corner_data = pd.concat(feature_datasets, ignore_index=True)

    corner_data = corner_data.sort_values(
        ["Date", "League", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    corner_data = create_corner_targets(corner_data)

    corner_data = corner_data.dropna(subset=FEATURE_COLUMNS).copy()

    seasons = sorted(corner_data["Season"].unique())

    all_predictions = []

    for market_name in CORNER_MARKETS.keys():
        print(f"\nBacktesting market: {market_name}")

        for test_season in seasons[2:]:
            train = corner_data[corner_data["Season"] < test_season].copy()
            test = corner_data[corner_data["Season"] == test_season].copy()

            if len(train) == 0 or len(test) == 0:
                continue

            X_train = train[FEATURE_COLUMNS]
            y_train = train[market_name]

            X_test = test[FEATURE_COLUMNS]
            y_test = test[market_name]

            # Skip if target has only one class in training
            if y_train.nunique() < 2:
                print(f"Skipping {market_name} {test_season}: only one class in training.")
                continue

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=5000))
            ])

            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.50).astype(int)

            for i, (_, match) in enumerate(test.iterrows()):
                probability = probabilities[i]

                all_predictions.append({
                    "Market": market_name,
                    "TestSeason": test_season,
                    "League": match["League"],
                    "Date": match["Date"],
                    "HomeTeam": match["HomeTeam"],
                    "AwayTeam": match["AwayTeam"],
                    "HomeCorners": match["HomeCorners"],
                    "AwayCorners": match["AwayCorners"],
                    "TotalCorners": match["TotalCorners"],
                    "Actual": int(y_test.iloc[i]),
                    "Prediction": int(predictions[i]),
                    "Probability": probability,
                    "FairOdds": fair_odds(probability),
                })

    predictions_df = pd.DataFrame(all_predictions)

    if len(predictions_df) == 0:
        print("No predictions created.")
        raise SystemExit

    summary_df = calculate_market_summary(predictions_df)

    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\nCorner Market Summary:")
    print(summary_df.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format,
        "LogLoss": "{:.4f}".format,
        "BrierScore": "{:.4f}".format,
        "ActualHitRate": "{:.2%}".format,
        "AverageModelProbability": "{:.2%}".format,
        "AverageFairOdds": "{:.2f}".format,
        "HighConfidenceHitRate": "{:.2%}".format,
        "HighConfidenceAverageProbability": "{:.2%}".format,
        "HighConfidenceAverageFairOdds": "{:.2f}".format,
    }))

    print("\nTop high-confidence examples:")
    top_examples = (
        predictions_df[predictions_df["Probability"] >= 0.60]
        .sort_values("Probability", ascending=False)
        .head(20)
    )

    if len(top_examples) > 0:
        print(top_examples.to_string(index=False, formatters={
            "Probability": "{:.2%}".format,
            "FairOdds": "{:.2f}".format,
        }))
    else:
        print("No examples above 60% probability.")

    print("\nSaved results to:")
    print(SUMMARY_FILE)
    print(PREDICTIONS_FILE)


if __name__ == "__main__":
    main()