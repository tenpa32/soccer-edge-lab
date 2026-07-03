import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

DATA_FILE = "data/processed/feature_store_v1.csv"

EV_THRESHOLD = 0.15
MIN_ODDS = 2.00
MAX_ODDS = 3.00
STAKE = 1

features = pd.read_csv(DATA_FILE)
features["Date"] = pd.to_datetime(features["Date"])

features = features.dropna(subset=["B365H", "B365D", "B365A"])

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

seasons = sorted(features["Season"].unique())

all_bets = []
season_summary_rows = []
model_summary_rows = []

for test_season in seasons[2:]:
    train = features[features["Season"] < test_season]
    test = features[features["Season"] == test_season].copy()

    if len(train) == 0 or len(test) == 0:
        continue

    X_train = train[feature_columns]
    y_train = train["Result"]

    X_test = test[feature_columns]
    y_test = test["Result"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities)

    model_summary_rows.append({
        "TestSeason": test_season,
        "TrainingMatches": len(train),
        "TestingMatches": len(test),
        "Accuracy": accuracy,
        "LogLoss": loss,
    })

    season_bets = []

    for i, (_, match) in enumerate(test.iterrows()):
        probs = {
            "away": probabilities[i][0],
            "draw": probabilities[i][1],
            "home": probabilities[i][2],
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

        passes_strategy_filter = (
            best_ev >= EV_THRESHOLD
            and best_odds >= MIN_ODDS
            and best_odds < MAX_ODDS
        )

        if passes_strategy_filter:
            bet_won = best_outcome == actual_result
            profit = (best_odds - 1) * STAKE if bet_won else -STAKE

            bet_row = {
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
                "Won": bet_won,
            }

            season_bets.append(bet_row)
            all_bets.append(bet_row)

    season_bets_df = pd.DataFrame(season_bets)

    if len(season_bets_df) == 0:
        season_summary_rows.append({
            "TestSeason": test_season,
            "Bets": 0,
            "WinRate": 0,
            "AverageOdds": 0,
            "AverageEV": 0,
            "Profit": 0,
            "ROI": 0,
            "MaxDrawdown": 0,
        })
    else:
        season_bets_df = season_bets_df.sort_values("Date")
        season_bets_df["CumulativeProfit"] = season_bets_df["Profit"].cumsum()

        total_bets = len(season_bets_df)
        total_profit = season_bets_df["Profit"].sum()
        roi = total_profit / total_bets
        win_rate = season_bets_df["Won"].mean()
        avg_odds = season_bets_df["Odds"].mean()
        avg_ev = season_bets_df["EV"].mean()
        max_drawdown = (
            season_bets_df["CumulativeProfit"].cummax()
            - season_bets_df["CumulativeProfit"]
        ).max()

        season_summary_rows.append({
            "TestSeason": test_season,
            "Bets": total_bets,
            "WinRate": win_rate,
            "AverageOdds": avg_odds,
            "AverageEV": avg_ev,
            "Profit": total_profit,
            "ROI": roi,
            "MaxDrawdown": max_drawdown,
        })

bets = pd.DataFrame(all_bets)
season_summary = pd.DataFrame(season_summary_rows)
model_summary = pd.DataFrame(model_summary_rows)

print("===================================")
print("Soccer Edge Lab - Best Strategy Backtest")
print("===================================")

print("\nStrategy Rules:")
print(f"EV Threshold: {EV_THRESHOLD:.0%}")
print(f"Minimum Odds: {MIN_ODDS:.2f}")
print(f"Maximum Odds: {MAX_ODDS:.2f}")
print("Stake per bet:", STAKE, "unit")

print("\nModel Summary:")
print(model_summary.to_string(index=False, formatters={
    "Accuracy": "{:.2%}".format,
    "LogLoss": "{:.4f}".format,
}))

print("\nSeason-by-Season Strategy Results:")
print(season_summary.to_string(index=False, formatters={
    "WinRate": "{:.2%}".format,
    "AverageOdds": "{:.2f}".format,
    "AverageEV": "{:.2%}".format,
    "Profit": "{:.2f}".format,
    "ROI": "{:.2%}".format,
    "MaxDrawdown": "{:.2f}".format,
}))

if len(bets) > 0:
    bets = bets.sort_values("Date")
    bets["CumulativeProfit"] = bets["Profit"].cumsum()

    total_bets = len(bets)
    total_profit = bets["Profit"].sum()
    total_staked = total_bets * STAKE
    roi = total_profit / total_staked
    win_rate = bets["Won"].mean()
    avg_odds = bets["Odds"].mean()
    avg_ev = bets["EV"].mean()
    avg_model_prob = bets["ModelProb"].mean()
    profitable_seasons = (season_summary["ROI"] > 0).sum()
    max_drawdown = (
        bets["CumulativeProfit"].cummax()
        - bets["CumulativeProfit"]
    ).max()

    print("\nOverall Strategy Results:")
    print("Total Bets:", total_bets)
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Odds: {avg_odds:.2f}")
    print(f"Average Model Probability: {avg_model_prob:.2%}")
    print(f"Average EV: {avg_ev:.2%}")
    print(f"Total Staked: {total_staked:.2f} units")
    print(f"Profit: {total_profit:.2f} units")
    print(f"ROI: {roi:.2%}")
    print(f"Profitable Seasons: {profitable_seasons} / {len(season_summary)}")
    print(f"Max Drawdown: {max_drawdown:.2f} units")

    bets.to_csv("data/processed/best_strategy_bets_v1.csv", index=False)
    season_summary.to_csv("data/processed/best_strategy_season_summary_v1.csv", index=False)
    model_summary.to_csv("data/processed/best_strategy_model_summary_v1.csv", index=False)

    print("\nSaved results to:")
    print("data/processed/best_strategy_bets_v1.csv")
    print("data/processed/best_strategy_season_summary_v1.csv")
    print("data/processed/best_strategy_model_summary_v1.csv")

    print("\nSample Bets:")
    print(bets.head(10))
else:
    print("\nNo bets found.")