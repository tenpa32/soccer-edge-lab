class BetEngine:

    def expected_value(self, prob, odds):
        return (prob * odds) - 1

    def evaluate(self, probs, odds):

        return {
            "home": {
                "ev": self.expected_value(probs["home"], odds["home"])
            },
            "draw": {
                "ev": self.expected_value(probs["draw"], odds["draw"])
            },
            "away": {
                "ev": self.expected_value(probs["away"], odds["away"])
            }
        }