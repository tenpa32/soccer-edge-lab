import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

DATA_FILE = "data/processed/feature_store_v1.csv"

EV_THRESHOLD = 0.03
STAKE = 1

ODDS_BANDS = [
    (1.00, 1.50),
    (1.50, 2.00),
    (2.00, 3.00),
    (3.00, 5.00),
    (5.00, 10.00),
    (10.00, 100.00),
]

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

class_mapping = {
    0: "away",
    1: "draw",
    2: "home",
}

all_bets = []

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

        all_bets.append({
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

bets = pd.DataFrame(all_bets)

summary_rows = []

for low, high in ODDS_BANDS:
    band_bets = bets[
        (bets["Odds"] >= low) &
        (bets["Odds"] < high)
    ].copy()

    label = f"{low:.2f}-{high:.2f}"

    if len(band_bets) == 0:
        summary_rows.append({
            "Odds Band": label,
            "Bets": 0,
            "Win Rate": 0,
            "Average Odds": 0,
            "Average EV": 0,
            "Profit": 0,
            "ROI": 0,
        })
        continue

    total_bets = len(band_bets)
    total_profit = band_bets["Profit"].sum()
    total_staked = total_bets * STAKE
    roi = total_profit / total_staked
    win_rate = band_bets["Won"].mean()
    avg_odds = band_bets["Odds"].mean()
    avg_ev = band_bets["EV"].mean()

    summary_rows.append({
        "Odds Band": label,
        "Bets": total_bets,
        "Win Rate": win_rate,
        "Average Odds": avg_odds,
        "Average EV": avg_ev,
        "Profit": total_profit,
        "ROI": roi,
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv("data/processed/odds_band_analysis_v1.csv", index=False)

print("===================================")
print("Soccer Edge Lab - Odds Band Analysis v1")
print("===================================")

print("Training matches:", len(train))
print("Testing matches:", len(test))
print(f"Model Accuracy: {accuracy:.2%}")
print(f"Model Log Loss: {loss:.4f}")

print("\nBetting Rules:")
print(f"EV Threshold: {EV_THRESHOLD:.0%}")
print("Stake per bet:", STAKE, "unit")

print("\nOdds Band Results:")
print(summary.to_string(index=False, formatters={
    "Win Rate": "{:.2%}".format,
    "Average Odds": "{:.2f}".format,
    "Average EV": "{:.2%}".format,
    "Profit": "{:.2f}".format,
    "ROI": "{:.2%}".format,
}))

print("\nSaved results to:")
print("data/processed/odds_band_analysis_v1.csv")

print("\nBest ROI Band:")
if len(summary) > 0:
    best_band = summary.sort_values("ROI", ascending=False).iloc[0]
    print(best_band)