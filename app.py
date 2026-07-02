import streamlit as st

from core.data_feed import DataFeed
from core.model_service import ModelService
from core.bet_engine import BetEngine

st.set_page_config(page_title="Soccer Betting Dashboard", layout="wide")

st.title("⚽ Live Soccer Betting Dashboard")

feed = DataFeed()
model = ModelService()
bet_engine = BetEngine()

matches = feed.get_matches()

for _, match in matches.iterrows():

    st.subheader(match["match"])

    probs = model.predict_probs(match)

    odds = {
        "home": match["home_odds"],
        "draw": match["draw_odds"],
        "away": match["away_odds"]
    }

    evaluation = bet_engine.evaluate(probs, odds)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Home", f"{probs['home']:.2%}")
        st.metric("EV", f"{evaluation['home']['ev']:.2f}")

    with col2:
        st.metric("Draw", f"{probs['draw']:.2%}")
        st.metric("EV", f"{evaluation['draw']['ev']:.2f}")

    with col3:
        st.metric("Away", f"{probs['away']:.2%}")
        st.metric("EV", f"{evaluation['away']['ev']:.2f}")

    best = max(evaluation.items(), key=lambda x: x[1]["ev"])

    if best[1]["ev"] > 0.03:
        st.success(f"🔥 VALUE BET: {best[0]}")
    else:
        st.warning("No value bet")
