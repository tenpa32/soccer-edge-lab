from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import time


# =========================
# Config
# =========================

BASE_URL = "https://www.football-data.co.uk/mmz4281"

RAW_DIR = Path("data/raw/football_data")
RAW_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = {
    "E0": "Premier_League",
    "E1": "Championship",
    "SP1": "La_Liga",
    "I1": "Serie_A",
    "D1": "Bundesliga",
    "F1": "Ligue_1",
    "N1": "Eredivisie",
    "P1": "Portugal_Primeira_Liga",
    "B1": "Belgium_Jupiler",
    "SC0": "Scotland_Premiership",
}

SEASONS = [
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]


# =========================
# Helpers
# =========================

def season_label_from_code(season_code: str) -> str:
    first_year = int("20" + season_code[:2])
    second_year_short = season_code[2:]
    return f"{first_year}-{second_year_short}"


def download_csv(url: str, output_path: Path) -> bool:
    try:
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urlopen(request, timeout=30) as response:
            content = response.read()

        if len(content) < 100:
            print(f"SKIPPED small/empty file: {url}")
            return False

        output_path.write_bytes(content)
        return True

    except HTTPError as error:
        print(f"HTTP error {error.code}: {url}")
        return False

    except URLError as error:
        print(f"URL error: {url} | {error}")
        return False

    except Exception as error:
        print(f"Unexpected error: {url} | {error}")
        return False


# =========================
# Main
# =========================

def main():
    print("===================================")
    print("Soccer Edge Lab - Football Data Downloader")
    print("===================================")

    downloaded = 0
    skipped = 0

    for season_code in SEASONS:
        season_label = season_label_from_code(season_code)

        for league_code, league_name in LEAGUES.items():
            league_dir = RAW_DIR / league_name
            league_dir.mkdir(parents=True, exist_ok=True)

            output_file = league_dir / f"{season_code}_{league_code}.csv"

            if output_file.exists():
                print(f"Already exists: {output_file}")
                skipped += 1
                continue

            url = f"{BASE_URL}/{season_code}/{league_code}.csv"

            print(f"Downloading {league_name} {season_label}: {url}")

            success = download_csv(url, output_file)

            if success:
                print(f"Saved: {output_file}")
                downloaded += 1
            else:
                print(f"Failed or unavailable: {url}")
                skipped += 1

            time.sleep(0.5)

    print("\nDownload complete.")
    print(f"Downloaded files: {downloaded}")
    print(f"Skipped/failed files: {skipped}")
    print(f"Raw data folder: {RAW_DIR}")


if __name__ == "__main__":
    main()