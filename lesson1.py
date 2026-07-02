home_team = "Liverpool"
away_team = "Chelsea"

home_odds = 2.20
draw_odds = 3.50
away_odds = 4.20

model_probability = 0.58

expected_value = (model_probability * home_odds) - 1

print("Home Team:", home_team)
print("Away Team:", away_team)
print(f"Expected Value: {expected_value:.3f}")
