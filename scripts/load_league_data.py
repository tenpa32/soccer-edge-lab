import pandas as pd
from pathlib import Path

league_path = Path("data/raw/la_liga")

all_matches = []

for file in sorted(league_path.glob("*.csv")):
    season = file.stem

    df = pd.read_csv(file)
    df["Season"] = season

    all_matches.append(df)

matches = pd.concat(all_matches, ignore_index=True)

matches.to_csv("data/processed/la_liga_all_matches.csv", index=False)

print("Loaded seasons:")
print([file.name for file in sorted(league_path.glob("*.csv"))])

print("\nTotal matches:", len(matches))
print("Columns:", len(matches.columns))

print("\nSaved to:")
print("data/processed/la_liga_all_matches.csv")