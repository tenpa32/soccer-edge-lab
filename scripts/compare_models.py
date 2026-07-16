import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss


# =========================
# Config
# =========================

DATA_FILE = "data/processed/feature_store_v1.csv"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAKE = 1

EV_THRESHOLD = 0.15
MIN_ODDS = 2.00
MAX_ODDS = 3.00

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

class_mapping = {
    0: "away",
    1: "draw",
    2: "home",
}


# =========================
# Models to Compare
# =========================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        random_state=42
    ),
}


# =========================
# Helper Functions
# =========================

def calculate_max_drawdown(profits):
    if len(profits) == 0:
        return 0

    cumulative_profit = profits.cumsum()
    running_peak = cumulative_profit.cummax()
    drawdown = running_peak - cumulative_profit

    return drawdown.max()


def evaluate_model(model_name, model, features, seasons):
    model_rows = []
    bet_rows = []

    for test_season in seasons[2:]:
        train = features[features["Season"] < test_season].copy()
        test = features[features["Season"] == test_season].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        X_train = train[feature_columns]
        y_train = train["Result"]

        X_test = test[feature_columns]
        y_test = test["Result"]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        loss = log_loss(y_test, probabilities, labels=model.classes_)

        season_bets = []

        for i, (_, match) in enumerate(test.iterrows()):
            probs = {
                "away": probabilities[i][list(model.classes_).index(0)],
                "draw": probabilities[i][list(model.classes_).index(1)],
                "home": probabilities[i][list(model.classes_).index(2)],
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
            actual_result = class_mapping[match["Result"]]

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
                    "Model": model_name,
                    "TestSeason": test_season,
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
            "Model": model_name,
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

    return model_rows, bet_rows


# =========================
# Main
# =========================

print("===================================")
print("Soccer Edge Lab - Model Benchmark v1")
print("===================================")

features = pd.read_csv(DATA_FILE)
features["Date"] = pd.to_datetime(features["Date"])

features = features.dropna(subset=feature_columns + ["Result", "B365H", "B365D", "B365A"])

if "Season" not in features.columns:
    print("ERROR: Season column not found in feature_store_v1.csv")
    raise SystemExit

seasons = sorted(features["Season"].unique())

all_model_rows = []
all_bet_rows = []

for model_name, model in models.items():
    print(f"\nRunning model: {model_name}")

    model_rows, bet_rows = evaluate_model(
        model_name=model_name,
        model=model,
        features=features,
        seasons=seasons
    )

    all_model_rows.extend(model_rows)
    all_bet_rows.extend(bet_rows)

benchmark_results = pd.DataFrame(all_model_rows)
benchmark_bets = pd.DataFrame(all_bet_rows)

overall_results = (
    benchmark_results
    .groupby("Model")
    .agg(
        AvgAccuracy=("Accuracy", "mean"),
        AvgLogLoss=("LogLoss", "mean"),
        TotalBets=("Bets", "sum"),
        AvgWinRate=("WinRate", "mean"),
        AvgOdds=("AverageOdds", "mean"),
        AvgEV=("AverageEV", "mean"),
        TotalProfit=("Profit", "sum"),
        AvgROI=("ROI", "mean"),
        MaxDrawdown=("MaxDrawdown", "max"),
        ProfitableSeasons=("Profit", lambda x: (x > 0).sum()),
    )
    .reset_index()
)

overall_results = overall_results.sort_values(
    by=["TotalProfit", "AvgROI", "AvgLogLoss"],
    ascending=[False, False, True]
)

benchmark_results.to_csv(
    OUTPUT_DIR / "model_benchmark_by_season_v1.csv",
    index=False
)

benchmark_bets.to_csv(
    OUTPUT_DIR / "model_benchmark_bets_v1.csv",
    index=False
)

overall_results.to_csv(
    OUTPUT_DIR / "model_benchmark_overall_v1.csv",
    index=False
)

print("\nOverall Model Results:")
print(overall_results.to_string(index=False, formatters={
    "AvgAccuracy": "{:.2%}".format,
    "AvgLogLoss": "{:.4f}".format,
    "AvgWinRate": "{:.2%}".format,
    "AvgOdds": "{:.2f}".format,
    "AvgEV": "{:.2%}".format,
    "TotalProfit": "{:.2f}".format,
    "AvgROI": "{:.2%}".format,
    "MaxDrawdown": "{:.2f}".format,
}))

print("\nSaved results to:")
print("data/processed/model_benchmark_by_season_v1.csv")
print("data/processed/model_benchmark_bets_v1.csv")
print("data/processed/model_benchmark_overall_v1.csv")