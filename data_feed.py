import pandas as pd

class DataFeed:

    def get_matches(self):
        return pd.DataFrame([
            {
                "match": "Brazil vs Japan",
                "home": "Brazil",
                "away": "Japan",
                "home_odds": 1.75,
                "draw_odds": 3.60,
                "away_odds": 4.50
            },
            {
                "match": "France vs Germany",
                "home": "France",
                "away": "Germany",
                "home_odds": 2.10,
                "draw_odds": 3.30,
                "away_odds": 3.40
            }
        ])