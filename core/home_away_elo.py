class HomeAwayElo:
    def __init__(self, initial_rating=1500, k_factor=20):
        self.initial_rating = initial_rating
        self.k_factor = k_factor

        self.home_ratings = {}
        self.away_ratings = {}

    def get_home_rating(self, team):
        if team not in self.home_ratings:
            self.home_ratings[team] = self.initial_rating
        return self.home_ratings[team]

    def get_away_rating(self, team):
        if team not in self.away_ratings:
            self.away_ratings[team] = self.initial_rating
        return self.away_ratings[team]

    def expected_score(self, team_rating, opponent_rating):
        return 1 / (1 + 10 ** ((opponent_rating - team_rating) / 400))

    def update_match(self, home_team, away_team, home_goals, away_goals):
        home_rating = self.get_home_rating(home_team)
        away_rating = self.get_away_rating(away_team)

        home_expected = self.expected_score(home_rating, away_rating)
        away_expected = self.expected_score(away_rating, home_rating)

        if home_goals > away_goals:
            home_actual = 1
            away_actual = 0
        elif home_goals < away_goals:
            home_actual = 0
            away_actual = 1
        else:
            home_actual = 0.5
            away_actual = 0.5

        self.home_ratings[home_team] = (
            home_rating + self.k_factor * (home_actual - home_expected)
        )

        self.away_ratings[away_team] = (
            away_rating + self.k_factor * (away_actual - away_expected)
        )

        return self.home_ratings[home_team], self.away_ratings[away_team]