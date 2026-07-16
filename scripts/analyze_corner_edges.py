import pandas as pd
from pathlib import Path


# =========================
# Config
# =========================

INPUT_FILE = Path("data/processed/corner_market_predictions_v1.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_SUMMARY_FILE = OUTPUT_DIR / "corner_edge_summary_by_market_v1.csv"
LEAGUE_SUMMARY_FILE = OUTPUT_DIR / "corner_edge_summary_by_league_v1.csv"
MARKET_LEAGUE_SUMMARY_FILE = OUTPUT_DIR / "corner_edge_summary_by_market_league_v1.csv"
WATCHLIST_FILE = OUTPUT_DIR / "corner_edge_watchlist_v1.csv"

MIN_PROBABILITY = 0.55
STRONG_PROBABILITY = 0.60
ELITE_PROBABILITY = 0.70

# We need a margin above fair odds because sportsbooks include vig.
# Example: if fair odds are 1.50, requiring 5% edge means sportsbook odds must be at least 1.575.
MIN_EDGE_MARGIN = 0.05


# =========================
# Helpers
# =========================

def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b


def fair_odds(probability):
    if probability <= 0:
        return 999.0
    return 1 / probability


def minimum_bettable_odds(probability, margin=MIN_EDGE_MARGIN):
    return fair_odds(probability) * (1 + margin)


def summarize_group(df, group_columns):
    rows = []

    grouped = df.groupby(group_columns)

    for group_value, group_df in grouped:
        predictions = len(group_df)
        hit_rate = group_df["Actual"].mean()
        avg_probability = group_df["Probability"].mean()
        avg_fair_odds = group_df["FairOdds"].mean()

        high_confidence = group_df[group_df["Probability"] >= MIN_PROBABILITY].copy()
        strong_confidence = group_df[group_df["Probability"] >= STRONG_PROBABILITY].copy()
        elite_confidence = group_df[group_df["Probability"] >= ELITE_PROBABILITY].copy()

        row = {}

        if isinstance(group_value, tuple):
            for col, value in zip(group_columns, group_value):
                row[col] = value
        else:
            row[group_columns[0]] = group_value

        row.update({
            "Predictions": predictions,
            "HitRate": hit_rate,
            "AverageProbability": avg_probability,
            "AverageFairOdds": avg_fair_odds,

            "HighConfidencePicks": len(high_confidence),
            "HighConfidenceHitRate": high_confidence["Actual"].mean() if len(high_confidence) > 0 else 0,
            "HighConfidenceAverageProbability": high_confidence["Probability"].mean() if len(high_confidence) > 0 else 0,
            "HighConfidenceAverageFairOdds": high_confidence["FairOdds"].mean() if len(high_confidence) > 0 else 0,

            "StrongConfidencePicks": len(strong_confidence),
            "StrongConfidenceHitRate": strong_confidence["Actual"].mean() if len(strong_confidence) > 0 else 0,
            "StrongConfidenceAverageProbability": strong_confidence["Probability"].mean() if len(strong_confidence) > 0 else 0,
            "StrongConfidenceAverageFairOdds": strong_confidence["FairOdds"].mean() if len(strong_confidence) > 0 else 0,

            "EliteConfidencePicks": len(elite_confidence),
            "EliteConfidenceHitRate": elite_confidence["Actual"].mean() if len(elite_confidence) > 0 else 0,
            "EliteConfidenceAverageProbability": elite_confidence["Probability"].mean() if len(elite_confidence) > 0 else 0,
            "EliteConfidenceAverageFairOdds": elite_confidence["FairOdds"].mean() if len(elite_confidence) > 0 else 0,
        })

        rows.append(row)

    return pd.DataFrame(rows)


def create_watchlist(df):
    watchlist = df[df["Probability"] >= MIN_PROBABILITY].copy()

    watchlist["FairOdds"] = watchlist["Probability"].apply(fair_odds)
    watchlist["MinimumBettableOdds_5pctEdge"] = watchlist["Probability"].apply(
        lambda p: minimum_bettable_odds(p, MIN_EDGE_MARGIN)
    )

    watchlist["ConfidenceTier"] = "High"

    watchlist.loc[
        watchlist["Probability"] >= STRONG_PROBABILITY,
        "ConfidenceTier"
    ] = "Strong"

    watchlist.loc[
        watchlist["Probability"] >= ELITE_PROBABILITY,
        "ConfidenceTier"
    ] = "Elite"

    watchlist = watchlist.sort_values(
        by=["Probability", "FairOdds"],
        ascending=[False, True]
    ).reset_index(drop=True)

    output_columns = [
        "ConfidenceTier",
        "Market",
        "TestSeason",
        "League",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "Probability",
        "FairOdds",
        "MinimumBettableOdds_5pctEdge",
        "Actual",
        "HomeCorners",
        "AwayCorners",
        "TotalCorners",
    ]

    available_columns = [col for col in output_columns if col in watchlist.columns]

    return watchlist[available_columns].copy()


# =========================
# Main
# =========================

def main():
    print("===================================")
    print("Soccer Edge Lab - Corner Edge Finder")
    print("===================================")

    if not INPUT_FILE.exists():
        print(f"Missing file: {INPUT_FILE}")
        print("Run this first:")
        print("python scripts/backtest_corner_markets.py")
        raise SystemExit

    predictions = pd.read_csv(INPUT_FILE)

    required_columns = [
        "Market",
        "TestSeason",
        "League",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "Actual",
        "Probability",
        "FairOdds",
        "HomeCorners",
        "AwayCorners",
        "TotalCorners",
    ]

    missing = [col for col in required_columns if col not in predictions.columns]

    if missing:
        print(f"Missing columns in predictions file: {missing}")
        raise SystemExit

    predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")
    predictions["Probability"] = pd.to_numeric(predictions["Probability"], errors="coerce")
    predictions["Actual"] = pd.to_numeric(predictions["Actual"], errors="coerce")

    predictions = predictions.dropna(
        subset=[
            "Market",
            "League",
            "Date",
            "Actual",
            "Probability",
        ]
    ).copy()

    predictions["Actual"] = predictions["Actual"].astype(int)
    predictions["FairOdds"] = predictions["Probability"].apply(fair_odds)
    predictions["MinimumBettableOdds_5pctEdge"] = predictions["Probability"].apply(
        lambda p: minimum_bettable_odds(p, MIN_EDGE_MARGIN)
    )

    print("\nInput:")
    print(f"Predictions: {len(predictions):,}")
    print(f"Markets: {predictions['Market'].nunique()}")
    print(f"Leagues: {predictions['League'].nunique()}")
    print(f"Date range: {predictions['Date'].min()} to {predictions['Date'].max()}")

    market_summary = summarize_group(predictions, ["Market"])
    league_summary = summarize_group(predictions, ["League"])
    market_league_summary = summarize_group(predictions, ["Market", "League"])
    watchlist = create_watchlist(predictions)

    market_summary = market_summary.sort_values(
        by=["HighConfidenceHitRate", "HighConfidencePicks"],
        ascending=[False, False]
    )

    league_summary = league_summary.sort_values(
        by=["HighConfidenceHitRate", "HighConfidencePicks"],
        ascending=[False, False]
    )

    market_league_summary = market_league_summary.sort_values(
        by=["HighConfidenceHitRate", "HighConfidencePicks"],
        ascending=[False, False]
    )

    market_summary.to_csv(MARKET_SUMMARY_FILE, index=False)
    league_summary.to_csv(LEAGUE_SUMMARY_FILE, index=False)
    market_league_summary.to_csv(MARKET_LEAGUE_SUMMARY_FILE, index=False)
    watchlist.to_csv(WATCHLIST_FILE, index=False)

    print("\nBest Markets:")
    print(market_summary.to_string(index=False, formatters={
        "HitRate": "{:.2%}".format,
        "AverageProbability": "{:.2%}".format,
        "AverageFairOdds": "{:.2f}".format,
        "HighConfidenceHitRate": "{:.2%}".format,
        "HighConfidenceAverageProbability": "{:.2%}".format,
        "HighConfidenceAverageFairOdds": "{:.2f}".format,
        "StrongConfidenceHitRate": "{:.2%}".format,
        "StrongConfidenceAverageProbability": "{:.2%}".format,
        "StrongConfidenceAverageFairOdds": "{:.2f}".format,
        "EliteConfidenceHitRate": "{:.2%}".format,
        "EliteConfidenceAverageProbability": "{:.2%}".format,
        "EliteConfidenceAverageFairOdds": "{:.2f}".format,
    }))

    print("\nBest Leagues:")
    print(league_summary.to_string(index=False, formatters={
        "HitRate": "{:.2%}".format,
        "AverageProbability": "{:.2%}".format,
        "AverageFairOdds": "{:.2f}".format,
        "HighConfidenceHitRate": "{:.2%}".format,
        "HighConfidenceAverageProbability": "{:.2%}".format,
        "HighConfidenceAverageFairOdds": "{:.2f}".format,
        "StrongConfidenceHitRate": "{:.2%}".format,
        "StrongConfidenceAverageProbability": "{:.2%}".format,
        "StrongConfidenceAverageFairOdds": "{:.2f}".format,
        "EliteConfidenceHitRate": "{:.2%}".format,
        "EliteConfidenceAverageProbability": "{:.2%}".format,
        "EliteConfidenceAverageFairOdds": "{:.2f}".format,
    }))

    print("\nBest Market + League Combinations:")
    top_combos = market_league_summary[
        market_league_summary["HighConfidencePicks"] >= 100
    ].head(25)

    print(top_combos.to_string(index=False, formatters={
        "HitRate": "{:.2%}".format,
        "AverageProbability": "{:.2%}".format,
        "AverageFairOdds": "{:.2f}".format,
        "HighConfidenceHitRate": "{:.2%}".format,
        "HighConfidenceAverageProbability": "{:.2%}".format,
        "HighConfidenceAverageFairOdds": "{:.2f}".format,
        "StrongConfidenceHitRate": "{:.2%}".format,
        "StrongConfidenceAverageProbability": "{:.2%}".format,
        "StrongConfidenceAverageFairOdds": "{:.2f}".format,
        "EliteConfidenceHitRate": "{:.2%}".format,
        "EliteConfidenceAverageProbability": "{:.2%}".format,
        "EliteConfidenceAverageFairOdds": "{:.2f}".format,
    }))

    print("\nTop Watchlist Examples:")
    top_watchlist = watchlist.head(30)

    print(top_watchlist.to_string(index=False, formatters={
        "Probability": "{:.2%}".format,
        "FairOdds": "{:.2f}".format,
        "MinimumBettableOdds_5pctEdge": "{:.2f}".format,
    }))

    print("\nSaved results to:")
    print(MARKET_SUMMARY_FILE)
    print(LEAGUE_SUMMARY_FILE)
    print(MARKET_LEAGUE_SUMMARY_FILE)
    print(WATCHLIST_FILE)

    print("\nHow to use this:")
    print("If Probability = 60%, fair odds are 1.67.")
    print("With a 5% edge requirement, minimum bettable odds are about 1.75.")
    print("Only bet if sportsbook odds are ABOVE the minimum bettable odds.")


if __name__ == "__main__":
    main()