import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

DATA_FILE = "data/processed/feature_store_v1.csv"

STAKE = 1

EV_THRESHOLDS = [
    0.03,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
]

ODDS_RANGES = [
    (1.50, 3.00),
    (1.50, 4.00),
    (1.50, 5.00),
    (2.00, 3.00),
    (2.00, 4.00),
    (2.00, 5.00),
    (3.00, 5.00),
]

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

features = pd.read_csv(DATA_FILE)
features["Date"] = pd.to_datetime(features["Date"])

features = features.dropna(subset=["B365H", "B365D", "B365A"])

if "Season" not in features.columns:
    print("ERROR: Season column not found in feature_store_v1.csv")
    raise SystemExit

seasons = sorted(features["Season"].unique())

all_opportunities = []
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
        "Test Season": test_season,
        "Training Matches": len(train),
        "Testing Matches": len(test),
        "Accuracy": accuracy,
        "Log Loss": loss,
    })

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

        bet_won = best_outcome == actual_result

        if bet_won:
            profit = (best_odds - 1) * STAKE
        else:
            profit = -STAKE

        all_opportunities.append({
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
        })

opportunities = pd.DataFrame(all_opportunities)

strategy_rows = []

for ev_threshold in EV_THRESHOLDS:
    for min_odds, max_odds in ODDS_RANGES:
        strategy_bets = opportunities[
            (opportunities["EV"] >= ev_threshold)
            & (opportunities["Odds"] >= min_odds)
            & (opportunities["Odds"] < max_odds)
        ].copy()

        strategy_name = f"EV>={ev_threshold:.0%}, Odds {min_odds:.2f}-{max_odds:.2f}"

        if len(strategy_bets) == 0:
            strategy_rows.append({
                "Strategy": strategy_name,
                "EV Threshold": ev_threshold,
                "Min Odds": min_odds,
                "Max Odds": max_odds,
                "Total Bets": 0,
                "Win Rate": 0,
                "Average Odds": 0,
                "Average EV": 0,
                "Profit": 0,
                "ROI": 0,
                "Profitable Seasons": 0,
                "Worst Season ROI": 0,
                "Best Season ROI": 0,
                "Max Drawdown": 0,
            })
            continue

        total_bets = len(strategy_bets)
        total_profit = strategy_bets["Profit"].sum()
        total_staked = total_bets * STAKE
        roi = total_profit / total_staked
        win_rate = strategy_bets["Won"].mean()
        avg_odds = strategy_bets["Odds"].mean()
        avg_ev = strategy_bets["EV"].mean()

        season_results = []

        for season in seasons[2:]:
            season_bets = strategy_bets[strategy_bets["TestSeason"] == season]

            if len(season_bets) == 0:
                season_roi = 0
                season_profit = 0
                season_total_bets = 0
            else:
                season_profit = season_bets["Profit"].sum()
                season_total_bets = len(season_bets)
                season_roi = season_profit / season_total_bets

            season_results.append({
                "Season": season,
                "Bets": season_total_bets,
                "Profit": season_profit,
                "ROI": season_roi,
            })

        season_results_df = pd.DataFrame(season_results)

        profitable_seasons = (season_results_df["ROI"] > 0).sum()
        worst_season_roi = season_results_df["ROI"].min()
        best_season_roi = season_results_df["ROI"].max()

        strategy_bets = strategy_bets.sort_values("Date")
        strategy_bets["CumulativeProfit"] = strategy_bets["Profit"].cumsum()
        max_drawdown = (
            strategy_bets["CumulativeProfit"].cummax()
            - strategy_bets["CumulativeProfit"]
        ).max()

        strategy_rows.append({
            "Strategy": strategy_name,
            "EV Threshold": ev_threshold,
            "Min Odds": min_odds,
            "Max Odds": max_odds,
            "Total Bets": total_bets,
            "Win Rate": win_rate,
            "Average Odds": avg_odds,
            "Average EV": avg_ev,
            "Profit": total_profit,
            "ROI": roi,
            "Profitable Seasons": profitable_seasons,
            "Worst Season ROI": worst_season_roi,
            "Best Season ROI": best_season_roi,
            "Max Drawdown": max_drawdown,
        })

summary = pd.DataFrame(strategy_rows)

summary = summary.sort_values(
    by=["Profitable Seasons", "ROI", "Worst Season ROI"],
    ascending=[False, False, False]
)

model_summary = pd.DataFrame(model_summary_rows)

summary.to_csv("data/processed/walk_forward_grid_search_v1.csv", index=False)
opportunities.to_csv("data/processed/walk_forward_opportunities_v1.csv", index=False)
model_summary.to_csv("data/processed/walk_forward_model_summary_v1.csv", index=False)

print("===================================")
print("Soccer Edge Lab - Walk-Forward Grid Search v1")
print("===================================")

print("\nModel Summary:")
print(model_summary.to_string(index=False, formatters={
    "Accuracy": "{:.2%}".format,
    "Log Loss": "{:.4f}".format,
}))

print("\nTop Strategies:")
print(summary.head(15).to_string(index=False, formatters={
    "EV Threshold": "{:.0%}".format,
    "Win Rate": "{:.2%}".format,
    "Average Odds": "{:.2f}".format,
    "Average EV": "{:.2%}".format,
    "Profit": "{:.2f}".format,
    "ROI": "{:.2%}".format,
    "Worst Season ROI": "{:.2%}".format,
    "Best Season ROI": "{:.2%}".format,
    "Max Drawdown": "{:.2f}".format,
}))

print("\nSaved results to:")
print("data/processed/walk_forward_grid_search_v1.csv")
print("data/processed/walk_forward_opportunities_v1.csv")
print("data/processed/walk_forward_model_summary_v1.csv")