import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

features = pd.read_csv("data/processed/match_features.csv")

X = features[
    [
        "HomeElo",
        "AwayElo",
        "EloDiff",
        "HomeFormPoints",
        "AwayFormPoints",
        "HomeFormGoalsFor",
        "AwayFormGoalsFor",
        "HomeFormGoalsAgainst",
        "AwayFormGoalsAgainst",
    ]
]

y = features["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("===================================")
print("Soccer Edge Lab - First ML Model")
print("===================================")

print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))