import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/match_features.csv"
OUTPUT_FILE = "data/processed/feature_store_v1.csv"

features = pd.read_csv(INPUT_FILE)

feature_store = features.copy()

feature_store["HomeAdvantage"] = 1

feature_store["FormPointDiff"] = (
    feature_store["HomeFormPoints"] - feature_store["AwayFormPoints"]
)

feature_store["FormGoalsForDiff"] = (
    feature_store["HomeFormGoalsFor"] - feature_store["AwayFormGoalsFor"]
)

feature_store["FormGoalsAgainstDiff"] = (
    feature_store["HomeFormGoalsAgainst"] - feature_store["AwayFormGoalsAgainst"]
)

Path("data/processed").mkdir(parents=True, exist_ok=True)

feature_store.to_csv(OUTPUT_FILE, index=False)

print("===================================")
print("Soccer Edge Lab - Feature Store v1")
print("===================================")
print("Input:", INPUT_FILE)
print("Output:", OUTPUT_FILE)
print("Rows:", len(feature_store))
print("Columns:", len(feature_store.columns))
print("\nColumns:")
print(feature_store.columns.tolist())