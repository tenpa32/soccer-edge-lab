import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

DATA_FILE = "data/processed/feature_store_v1.csv"

EV_THRESHOLD = 0.03
MIN_ODDS = 2.00
MAX_ODDS = 5.00
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

seasons = sorted(features["Season"].unique()) if "Season" in features.columns else []

if len(seasons) == 0:
    print("ERROR: Season column not found in feature_store_v1.csv")
    print("We need Season in the feature store before walk-forward testing.")
    raise SystemExit

summary_rows = []
all_bets = []

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

        actual_result = class_mapping[match["Result"]]

        passes_strategy_filter = (
            best_ev >= EV_THRESHOLD
            and best_odds >= MIN_ODDS
            and best_odds < MAX_ODDS
        )

        if passes_strategy_filter:
            bet_won = best_outcome == actual_result

            if bet_won:
                profit = (best_odds - 1) * STAKE
            else:
                profit = -STAKE

            bet_row = {
                "TestSeason": test_season,
                "Date": match["Date"],
                "HomeTeam": match["HomeTeam"],
                "AwayTeam": match["AwayTeam"],
                "Bet": best_outcome,
                "ModelProb": probs[best_outcome],
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
        summary_rows.append({
            "Test Season": test_season,
            "Training Matches": len(train),
            "Testing Matches": len(test),
            "Accuracy": accuracy,
            "Log Loss": loss,
            "Bets": 0,
            "Win Rate": 0,
            "Average Odds": 0,
            "Profit": 0,
            "ROI": 0,
        })
    else:
        total_bets = len(season_bets_df)
        total_profit = season_bets_df["Profit"].sum()
        total_staked = total_bets * STAKE
        roi = total_profit / total_staked
        win_rate = season_bets_df["Won"].mean()
        avg_odds = season_bets_df["Odds"].mean()

        summary_rows.append({
            "Test Season": test_season,
            "Training Matches": len(train),
            "Testing Matches": len(test),
            "Accuracy": accuracy,
            "Log Loss": loss,
            "Bets": total_bets,
            "Win Rate": win_rate,
            "Average Odds": avg_odds,
            "Profit": total_profit,
            "ROI": roi,
        })

summary = pd.DataFrame(summary_rows)
bets = pd.DataFrame(all_bets)

print("===================================")
print("Soccer Edge Lab - Walk-Forward Backtest v1")
print("===================================")

print("\nStrategy Rules:")
print(f"EV Threshold: {EV_THRESHOLD:.0%}")
print(f"Minimum Odds: {MIN_ODDS:.2f}")
print(f"Maximum Odds: {MAX_ODDS:.2f}")
print("Stake per bet:", STAKE, "unit")

print("\nSeason Results:")
print(summary.to_string(index=False, formatters={
    "Accuracy": "{:.2%}".format,
    "Log Loss": "{:.4f}".format,
    "Win Rate": "{:.2%}".format,
    "Average Odds": "{:.2f}".format,
    "Profit": "{:.2f}".format,
    "ROI": "{:.2%}".format,
}))

if len(bets) > 0:
    total_bets = len(bets)
    total_profit = bets["Profit"].sum()
    total_staked = total_bets * STAKE
    overall_roi = total_profit / total_staked
    overall_win_rate = bets["Won"].mean()
    avg_odds = bets["Odds"].mean()

    bets["CumulativeProfit"] = bets["Profit"].cumsum()
    max_drawdown = (bets["CumulativeProfit"].cummax() - bets["CumulativeProfit"]).max()

    print("\nOverall Walk-Forward Results:")
    print("Total Bets:", total_bets)
    print(f"Win Rate: {overall_win_rate:.2%}")
    print(f"Average Odds: {avg_odds:.2f}")
    print(f"Profit: {total_profit:.2f} units")
    print(f"ROI: {overall_roi:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2f} units")

    summary.to_csv("data/processed/walk_forward_summary_v1.csv", index=False)
    bets.to_csv("data/processed/walk_forward_bets_v1.csv", index=False)

    print("\nSaved results to:")
    print("data/processed/walk_forward_summary_v1.csv")
    print("data/processed/walk_forward_bets_v1.csv")
else:
    print("\nNo bets found across walk-forward test.")