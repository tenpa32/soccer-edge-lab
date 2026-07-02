def expected_value(probability, odds):
    return (probability * odds) - 1


home_team = "Liverpool"
away_team = "Chelsea"

home_odds = 2.20
model_probability = 0.58

ev = expected_value(model_probability, home_odds)

print("Home Team:", home_team)
print("Away Team:", away_team)
print(f"Expected Value: {ev:.3f}")

if ev > 0:
    print("This is a positive EV bet")
else:
    print("This is not a value bet")
    