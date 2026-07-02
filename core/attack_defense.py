class AttackDefenseEngine:
    def __init__(self, window=5):
        self.window = window
        self.team_history = {}

    def get_team_history(self, team):
        if team not in self.team_history:
            self.team_history[team] = []
        return self.team_history[team]

    def get_features(self, team):
        history = self.get_team_history(team)
        recent = history[-self.window:]

        if len(recent) == 0:
            return {
                "attack_rating": 0,
                "defense_rating": 0,
                "goal_diff_rating": 0,
            }

        goals_for = sum(match["goals_for"] for match in recent)
        goals_against = sum(match["goals_against"] for match in recent)
        matches_played = len(recent)

        attack_rating = goals_for / matches_played
        defense_rating = goals_against / matches_played
        goal_diff_rating = attack_rating - defense_rating

        return {
            "attack_rating": attack_rating,
            "defense_rating": defense_rating,
            "goal_diff_rating": goal_diff_rating,
        }

    def update_match(self, home_team, away_team, home_goals, away_goals):
        self.get_team_history(home_team).append({
            "goals_for": home_goals,
            "goals_against": away_goals,
        })

        self.get_team_history(away_team).append({
            "goals_for": away_goals,
            "goals_against": home_goals,
        })