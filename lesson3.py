matches = [
    {
        "home": "Liverpool",
        "away": "Chelsea",
        "home_odds": 2.20,
        "probability": 0.58
    },
    {
        "home": "Arsenal",
        "away": "Tottenham",
        "home_odds": 1.90,
        "probability": 0.61
    },
    {
        "home": "Barcelona",
        "away": "Real Madrid",
        "home_odds": 2.75,
        "probability": 0.44
    }
]


def expected_value(probability, odds):
    return (probability * odds) - 1


for match in matches:

    ev = expected_value(match["probability"], match["home_odds"])

    print("------------------------")
    print(match["home"], "vs", match["away"])
    print(f"Expected Value: {ev:.3f}")

    if ev > 0:
        print("✅ VALUE BET")
    else:
        print("❌ No value")