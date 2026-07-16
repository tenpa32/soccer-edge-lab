import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


# =========================
# Config
# =========================

DATA_FILE = Path("data/processed/multi_league_feature_store_v1.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SUMMARY_FILE = OUTPUT_DIR / "multi_league_model_summary_v1.csv"
SEASON_SUMMARY_FILE = OUTPUT_DIR / "multi_league_strategy_season_summary_v1.csv"
LEAGUE_SUMMARY_FILE = OUTPUT_DIR / "multi_league_strategy_league_summary_v1.csv"
BETS_FILE = OUTPUT_DIR / "multi_league_strategy_bets_v1.csv"

STAKE = 1

EV_THRESHOLD = 0.15
MIN_ODDS = 2.00
MAX_ODDS = 3.00

FEATURE_COLUMNS = [
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

CLASS_MAPPING = {
    0: "away",
    1: "draw",
    2: "home",
}


# =========================
# Helpers
# =========================

def calculate_max_drawdown(profits: pd.Series) -> float:
    if len(profits) == 0:
        return 0

    cumulative_profit = profits.cumsum()
    running_peak = cumulative_profit.cummax()
    drawdown = running_peak - cumulative_profit

    return drawdown.max()


def summarize_bets(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []

    for group_value, group_df in df.groupby(group_col):
        bets = len(group_df)
        profit = group_df["Profit"].sum()
        roi = profit / bets if bets > 0 else 0
        win_rate = group_df["Won"].mean() if bets > 0 else 0
        avg_odds = group_df["Odds"].mean() if bets > 0 else 0
        avg_ev = group_df["EV"].mean() if bets > 0 else 0
        max_drawdown = calculate_max_drawdown(group_df["Profit"])

        rows.append({
            group_col: group_value,
            "Bets": bets,
            "Wins": int(group_df["Won"].sum()),
            "WinRate": win_rate,
            "AverageOdds": avg_odds,
            "AverageEV": avg_ev,
            "Profit": profit,
            "ROI": roi,
            "MaxDrawdown": max_drawdown,
        })

    return pd.DataFrame(rows)


# =========================
# Main
# =========================

def main():
    print("===================================")
    print("Soccer Edge Lab - Multi-League Strategy Backtest")
    print("===================================")

    if not DATA_FILE.exists():
        print(f"Missing file: {DATA_FILE}")
        print("Run this first:")
        print("python scripts/build_multi_league_features.py")
        raise SystemExit

    features = pd.read_csv(DATA_FILE)
    features["Date"] = pd.to_datetime(features["Date"], errors="coerce")

    required_columns = FEATURE_COLUMNS + [
        "League",
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "Result",
        "B365H",
        "B365D",
        "B365A",
    ]

    features = features.dropna(subset=required_columns).copy()

    features["Season"] = pd.to_numeric(features["Season"], errors="coerce")
    features["Result"] = pd.to_numeric(features["Result"], errors="coerce")

    features = features.dropna(subset=["Season", "Result"]).copy()
    features["Season"] = features["Season"].astype(int)
    features["Result"] = features["Result"].astype(int)

    seasons = sorted(features["Season"].unique())

    print("\nStrategy Rules:")
    print(f"EV Threshold: {EV_THRESHOLD:.0%}")
    print(f"Minimum Odds: {MIN_ODDS:.2f}")
    print(f"Maximum Odds: {MAX_ODDS:.2f}")
    print(f"Stake per bet: {STAKE} unit")

    print("\nDataset:")
    print(f"Matches: {len(features):,}")
    print(f"Leagues: {features['League'].nunique()}")
    print(f"Seasons: {features['Season'].nunique()}")
    print(f"Date range: {features['Date'].min()} to {features['Date'].max()}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])

    model_rows = []
    bet_rows = []

    for test_season in seasons[2:]:
        train = features[features["Season"] < test_season].copy()
        test = features[features["Season"] == test_season].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        X_train = train[FEATURE_COLUMNS]
        y_train = train["Result"]

        X_test = test[FEATURE_COLUMNS]
        y_test = test["Result"]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        loss = log_loss(y_test, probabilities, labels=model.classes_)

        season_bets = []

        for i, (_, match) in enumerate(test.iterrows()):
            class_index = {
                class_label: list(model.classes_).index(class_label)
                for class_label in model.classes_
            }

            probs = {
                "away": probabilities[i][class_index[0]],
                "draw": probabilities[i][class_index[1]],
                "home": probabilities[i][class_index[2]],
            }

            odds = {
                "home": match["B365H"],
                "draw": match["B365D"],
                "away": match["B365A"],
            }

            evs = {
                outcome: (probs[outcome] * odds[outcome]) - 1
                for outcome in ["home", "draw", "away"]
            }

            best_outcome = max(evs, key=evs.get)
            best_ev = evs[best_outcome]
            best_odds = odds[best_outcome]
            model_prob = probs[best_outcome]
            actual_result = CLASS_MAPPING[match["Result"]]

            if (
                best_ev >= EV_THRESHOLD
                and best_odds >= MIN_ODDS
                and best_odds < MAX_ODDS
            ):
                won = best_outcome == actual_result

                if won:
                    profit = (best_odds - 1) * STAKE
                else:
                    profit = -STAKE

                season_bets.append({
                    "TestSeason": test_season,
                    "League": match["League"],
                    "Date": match["Date"],
                    "HomeTeam": match["HomeTeam"],
                    "AwayTeam": match["AwayTeam"],
                    "Bet": best_outcome,
                    "ModelProb": model_prob,
                    "Odds": best_odds,
                    "EV": best_ev,
                    "ActualResult": actual_result,
                    "Profit": profit,
                    "Won": won,
                })

        season_bets_df = pd.DataFrame(season_bets)

        if len(season_bets_df) > 0:
            bets = len(season_bets_df)
            profit = season_bets_df["Profit"].sum()
            roi = profit / bets
            win_rate = season_bets_df["Won"].mean()
            avg_odds = season_bets_df["Odds"].mean()
            avg_ev = season_bets_df["EV"].mean()
            max_drawdown = calculate_max_drawdown(season_bets_df["Profit"])
        else:
            bets = 0
            profit = 0
            roi = 0
            win_rate = 0
            avg_odds = 0
            avg_ev = 0
            max_drawdown = 0

        model_rows.append({
            "TestSeason": test_season,
            "TrainingMatches": len(train),
            "TestingMatches": len(test),
            "Accuracy": accuracy,
            "LogLoss": loss,
            "Bets": bets,
            "WinRate": win_rate,
            "AverageOdds": avg_odds,
            "AverageEV": avg_ev,
            "Profit": profit,
            "ROI": roi,
            "MaxDrawdown": max_drawdown,
        })

        bet_rows.extend(season_bets)

    model_summary = pd.DataFrame(model_rows)
    bets_df = pd.DataFrame(bet_rows)

    if len(bets_df) == 0:
        print("\nNo bets found with current strategy rules.")
        raise SystemExit

    bets_df["CumulativeProfit"] = bets_df["Profit"].cumsum()

    season_summary = summarize_bets(bets_df, "TestSeason")
    league_summary = summarize_bets(bets_df, "League")

    season_summary = season_summary.sort_values("TestSeason")
    league_summary = league_summary.sort_values("Profit", ascending=False)

    model_summary.to_csv(MODEL_SUMMARY_FILE, index=False)
    season_summary.to_csv(SEASON_SUMMARY_FILE, index=False)
    league_summary.to_csv(LEAGUE_SUMMARY_FILE, index=False)
    bets_df.to_csv(BETS_FILE, index=False)

    total_bets = len(bets_df)
    total_profit = bets_df["Profit"].sum()
    overall_roi = total_profit / total_bets
    overall_win_rate = bets_df["Won"].mean()
    avg_odds = bets_df["Odds"].mean()
    avg_ev = bets_df["EV"].mean()
    max_drawdown = calculate_max_drawdown(bets_df["Profit"])

    profitable_seasons = (season_summary["Profit"] > 0).sum()
    profitable_leagues = (league_summary["Profit"] > 0).sum()

    print("\nModel Summary:")
    print(model_summary.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format,
        "LogLoss": "{:.4f}".format,
        "WinRate": "{:.2%}".format,
        "AverageOdds": "{:.2f}".format,
        "AverageEV": "{:.2%}".format,
        "Profit": "{:.2f}".format,
        "ROI": "{:.2%}".format,
        "MaxDrawdown": "{:.2f}".format,
    }))

    print("\nOverall Strategy Results:")
    print(f"Total Bets: {total_bets}")
    print(f"Win Rate: {overall_win_rate:.2%}")
    print(f"Average Odds: {avg_odds:.2f}")
    print(f"Average EV: {avg_ev:.2%}")
    print(f"Profit: {total_profit:.2f} units")
    print(f"ROI: {overall_roi:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2f} units")
    print(f"Profitable Seasons: {profitable_seasons} / {len(season_summary)}")
    print(f"Profitable Leagues: {profitable_leagues} / {len(league_summary)}")

    print("\nSeason Results:")
    print(season_summary.to_string(index=False, formatters={
        "WinRate": "{:.2%}".format,
        "AverageOdds": "{:.2f}".format,
        "AverageEV": "{:.2%}".format,
        "Profit": "{:.2f}".format,
        "ROI": "{:.2%}".format,
        "MaxDrawdown": "{:.2f}".format,
    }))

    print("\nLeague Results:")
    print(league_summary.to_string(index=False, formatters={
        "WinRate": "{:.2%}".format,
        "AverageOdds": "{:.2f}".format,
        "AverageEV": "{:.2%}".format,
        "Profit": "{:.2f}".format,
        "ROI": "{:.2%}".format,
        "MaxDrawdown": "{:.2f}".format,
    }))

    print("\nSaved results to:")
    print(MODEL_SUMMARY_FILE)
    print(SEASON_SUMMARY_FILE)
    print(LEAGUE_SUMMARY_FILE)
    print(BETS_FILE)


if __name__ == "__main__":
    main()