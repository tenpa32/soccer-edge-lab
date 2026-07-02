class FormEngine:
    def __init__(self, window=5):
        self.window = window
        self.team_history = {}

    def get_team_history(self, team):
        if team not in self.team_history:
            self.team_history[team] = []
        return self.team_history[team]

    def get_form_features(self, team):
        history = self.get_team_history(team)
        recent = history[-self.window:]

        if len(recent) == 0:
            return {
                "form_points": 0,
                "form_goals_for": 0,
                "form_goals_against": 0
            }

        return {
            "form_points": sum(match["points"] for match in recent),
            "form_goals_for": sum(match["goals_for"] for match in recent),
            "form_goals_against": sum(match["goals_against"] for match in recent)
        }

    def update_match(self, home_team, away_team, home_goals, away_goals):
        if home_goals > away_goals:
            home_points, away_points = 3, 0
        elif home_goals < away_goals:
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1

        self.get_team_history(home_team).append({
            "points": home_points,
            "goals_for": home_goals,
            "goals_against": away_goals
        })

        self.get_team_history(away_team).append({
            "points": away_points,
            "goals_for": away_goals,
            "goals_against": home_goals
        })