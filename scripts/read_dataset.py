import pandas as pd

matches = pd.read_csv("data/raw/EPL_2025.csv")

# Home goals
home_goals = matches.groupby("HomeTeam")["FTHG"].sum()

# Away goals
away_goals = matches.groupby("AwayTeam")["FTAG"].sum()

# Total goals
total_goals = home_goals.add(away_goals, fill_value=0)

print("===================================")
print("Top Scoring Teams")
print("===================================")

print(total_goals.sort_values(ascending=False))