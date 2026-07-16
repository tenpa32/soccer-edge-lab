import pandas as pd
from pathlib import Path


# =========================
# Config
# =========================

RAW_DIR = Path("data/raw/football_data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "multi_league_matches.csv"

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]

ODDS_COLUMNS = [
    "B365H",
    "B365D",
    "B365A",
]


# =========================
# Helpers
# =========================

def parse_season_code(filename: str) -> str:
    # Example: 2324_E0.csv -> 2324
    return filename.split("_")[0]


def parse_league_code(filename: str) -> str:
    # Example: 2324_E0.csv -> E0
    return filename.split("_")[1].replace(".csv", "")


def season_start_year_from_code(season_code: str) -> int:
    # Examples:
    # 1920 -> 2019
    # 2021 -> 2020
    # 2122 -> 2021
    return int("20" + season_code[:2])


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    return df


def clean_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

    df = df[df["FTR"].isin(["H", "D", "A"])].copy()

    result_map = {
        "H": 2,  # home
        "D": 1,  # draw
        "A": 0,  # away
    }

    result_label_map = {
        "H": "home",
        "D": "draw",
        "A": "away",
    }

    df["Result"] = df["FTR"].map(result_map)
    df["ResultLabel"] = df["FTR"].map(result_label_map)

    return df


def clean_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ODDS_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_single_file(file_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(file_path)
    except Exception as error:
        print(f"Could not read {file_path}: {error}")
        return None

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        print(f"Skipping {file_path} because missing columns: {missing}")
        return None

    season_code = parse_season_code(file_path.name)
    league_code = parse_league_code(file_path.name)
    league_name = file_path.parent.name
    season_start_year = season_start_year_from_code(season_code)

    df = df.copy()

    df["League"] = league_name
    df["LeagueCode"] = league_code
    df["SeasonCode"] = season_code
    df["Season"] = season_start_year

    df = normalize_date_column(df)
    df = clean_result_columns(df)
    df = clean_odds_columns(df)

    df = df.dropna(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "Result",
        ]
    )

    preferred_columns = [
        "League",
        "LeagueCode",
        "SeasonCode",
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "Result",
        "ResultLabel",
        "B365H",
        "B365D",
        "B365A",
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HF",
        "AF",
        "HY",
        "AY",
        "HR",
        "AR",
    ]

    available_columns = [
        col for col in preferred_columns
        if col in df.columns
    ]

    return df[available_columns].copy()


# =========================
# Main
# =========================

def main():
    print("===================================")
    print("Soccer Edge Lab - Multi-League Dataset Builder")
    print("===================================")

    if not RAW_DIR.exists():
        print(f"Raw folder not found: {RAW_DIR}")
        print("Run this first:")
        print("python scripts/download_football_data.py")
        raise SystemExit

    csv_files = sorted(RAW_DIR.glob("*/*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {RAW_DIR}")
        print("Run this first:")
        print("python scripts/download_football_data.py")
        raise SystemExit

    datasets = []

    for file_path in csv_files:
        print(f"Loading: {file_path}")

        df = load_single_file(file_path)

        if df is not None and len(df) > 0:
            datasets.append(df)

    if not datasets:
        print("No valid datasets found.")
        raise SystemExit

    combined = pd.concat(datasets, ignore_index=True)

    combined = combined.sort_values(
        by=[
            "Date",
            "League",
            "HomeTeam",
            "AwayTeam",
        ]
    ).reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nMulti-league dataset created.")
    print(f"Rows: {len(combined):,}")
    print(f"Leagues: {combined['League'].nunique()}")
    print(f"Seasons: {combined['Season'].nunique()}")
    print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nRows by league:")
    print(
        combined
        .groupby("League")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nRows by season:")
    print(
        combined
        .groupby("Season")
        .size()
        .sort_index()
        .to_string()
    )

    odds_available = combined.dropna(
        subset=[
            "B365H",
            "B365D",
            "B365A",
        ]
    )

    print(f"\nRows with Bet365 1X2 odds: {len(odds_available):,}")

    print("\nSample rows:")
    print(
        combined.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()