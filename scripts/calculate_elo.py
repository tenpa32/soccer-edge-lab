import os
import sys
import pandas as pd

# Add the project root to Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.elo import EloRating

matches = pd.read_csv("data/raw/EPL_2025.csv")

elo = EloRating(initial_rating=1500, k_factor=20)

for _, match in matches.iterrows():
    elo.update_match(
        home_team=match["HomeTeam"],
        away_team=match["AwayTeam"],
        home_goals=match["FTHG"],
        away_goals=match["FTAG"]
    )

ratings = pd.DataFrame(
    list(elo.ratings.items()),
    columns=["Team", "Elo Rating"]
)

ratings = ratings.sort_values("Elo Rating", ascending=False)

print("===================================")
print("Final La Liga Elo Ratings")
print("===================================")
print(ratings)
ratings.to_csv("data/processed/elo_ratings.csv", index=False)

print("\nSaved Elo ratings to data/processed/elo_ratings.csv")