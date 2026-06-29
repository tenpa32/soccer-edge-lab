class ModelService:

    def predict_probs(self, match):

        if match["home"] == "Brazil":
            return {
                "home": 0.66,
                "draw": 0.20,
                "away": 0.14
            }

        return {
            "home": 0.52,
            "draw": 0.25,
            "away": 0.23
        }