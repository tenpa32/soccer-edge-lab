import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss

features = pd.read_csv("data/processed/feature_store_v1.csv")

train = features[features["Date"] < "2025-01-01"]
test = features[features["Date"] >= "2025-01-01"]

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

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)
loss = log_loss(y_test, probabilities)

print("===================================")
print("Soccer Edge Lab - Time-Based ML Model")
print("===================================")

print("Training matches:", len(train))
print("Testing matches:", len(test))

print(f"\nAccuracy: {accuracy:.2%}")
print(f"Log Loss: {loss:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))