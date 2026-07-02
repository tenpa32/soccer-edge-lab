import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

DATA_FILE = "data/processed/feature_store_v1.csv"

EV_THRESHOLD = 0.03
STAKE = 1

features = pd.read_csv(DATA_FILE)

features["Date"] = pd.to_datetime(features["Date"])

features = features.dropna(subset=["B365H", "B365D", "B365A"])

train = features[features["Date"] < "2025-01-01"]
test = features[features["Date"] >= "2025-01-01"].copy()

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

results = []

class_mapping = {
    0: "away",
    1: "draw",
    2: "home",
}

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

    if best_ev >= EV_THRESHOLD:
        bet_won = best_outcome == actual_result

        if bet_won:
            profit = (best_odds - 1) * STAKE
        else:
            profit = -STAKE

        results.append({
            "Date": match["Date"],
            "HomeTeam": match["HomeTeam"],
            "AwayTeam": match["AwayTeam"],
            "Bet": best_outcome,
            "Odds": best_odds,
            "EV": best_ev,
            "ActualResult": actual_result,
            "Profit": profit,
            "Won": bet_won,
        })

bets = pd.DataFrame(results)

print("===================================")
print("Soccer Edge Lab - Backtest v1")
print("===================================")

print("Training matches:", len(train))
print("Testing matches:", len(test))
print(f"Model Accuracy: {accuracy:.2%}")
print(f"Model Log Loss: {loss:.4f}")

print("\nBetting Rules:")
print("EV Threshold:", EV_THRESHOLD)
print("Stake per bet:", STAKE, "unit")

if len(bets) == 0:
    print("\nNo bets found.")
else:
    total_bets = len(bets)
    total_profit = bets["Profit"].sum()
    total_staked = total_bets * STAKE
    roi = total_profit / total_staked
    win_rate = bets["Won"].mean()
    avg_odds = bets["Odds"].mean()
    avg_ev = bets["EV"].mean()

    print("\nBacktest Results:")
    print("Bets Placed:", total_bets)
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Odds: {avg_odds:.2f}")
    print(f"Average EV: {avg_ev:.2%}")
    print(f"Total Staked: {total_staked:.2f} units")
    print(f"Profit: {total_profit:.2f} units")
    print(f"ROI: {roi:.2%}")

    bets.to_csv("data/processed/backtest_results_v1.csv", index=False)

    print("\nSaved results to:")
    print("data/processed/backtest_results_v1.csv")

    print("\nSample Bets:")
    print(bets.head(10))