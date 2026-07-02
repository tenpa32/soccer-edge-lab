import pandas as pd

RAW_FILE = "data/processed/la_liga_all_matches.csv"
OUTPUT_FILE = "data/processed/la_liga_clean_matches.csv"

KEEP_COLUMNS = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
    "B365H",
    "B365D",
    "B365A",
]

matches = pd.read_csv(RAW_FILE)

existing_columns = [col for col in KEEP_COLUMNS if col in matches.columns]

clean = matches[existing_columns].copy()

clean = clean.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

clean["Date"] = pd.to_datetime(clean["Date"], dayfirst=True, errors="coerce")

clean = clean.dropna(subset=["Date"])

clean = clean.sort_values(["Date", "HomeTeam", "AwayTeam"])

clean.to_csv(OUTPUT_FILE, index=False)

print("===================================")
print("Soccer Edge Lab - Data Cleaner")
print("===================================")
print("Input file:", RAW_FILE)
print("Output file:", OUTPUT_FILE)
print("Rows:", len(clean))
print("Columns:", len(clean.columns))
print("\nColumns kept:")
print(clean.columns.tolist())