import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Soccer Edge Lab",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Soccer Edge Lab")
st.caption(
    "Model-based soccer betting dashboard using Elo, ML probabilities, EV filtering, "
    "odds bands, and walk-forward backtesting."
)

# =========================
# File Paths
# =========================

DATA_DIR = Path("data/processed")

BETS_FILE = DATA_DIR / "best_strategy_bets_v1.csv"
SEASON_FILE = DATA_DIR / "best_strategy_season_summary_v1.csv"
MODEL_FILE = DATA_DIR / "best_strategy_model_summary_v1.csv"

# =========================
# Load Data
# =========================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

missing_files = []

for file_path in [BETS_FILE, SEASON_FILE, MODEL_FILE]:
    if not file_path.exists():
        missing_files.append(str(file_path))

if missing_files:
    st.error("Missing dashboard data files.")
    st.write("Run this first:")
    st.code("python scripts/backtest_best_strategy.py", language="powershell")
    st.write("Missing files:")
    for file in missing_files:
        st.write(f"- {file}")
    st.stop()

bets = load_csv(BETS_FILE)
season_results = load_csv(SEASON_FILE)
model_summary = load_csv(MODEL_FILE)

# Build equity curve from bets file
equity_curve = bets.copy()

if "CumulativeProfit" in equity_curve.columns:
    equity_curve["BetNumber"] = range(1, len(equity_curve) + 1)
else:
    equity_curve["CumulativeProfit"] = equity_curve["Profit"].cumsum()
    equity_curve["BetNumber"] = range(1, len(equity_curve) + 1)

# =========================
# Sidebar Filters
# =========================

st.sidebar.header("Strategy Filters")

ev_threshold = st.sidebar.slider(
    "Minimum EV",
    min_value=0.00,
    max_value=0.75,
    value=0.15,
    step=0.01
)

min_odds = st.sidebar.slider(
    "Minimum Odds",
    min_value=1.00,
    max_value=5.00,
    value=2.00,
    step=0.05
)

max_odds = st.sidebar.slider(
    "Maximum Odds",
    min_value=1.00,
    max_value=10.00,
    value=3.00,
    step=0.05
)

# =========================
# Column Helper
# =========================

def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

ev_col = find_column(bets, ["EV", "ev", "expected_value", "ExpectedValue"])
odds_col = find_column(bets, ["Odds", "odds", "selected_odds", "bet_odds"])
profit_col = find_column(bets, ["Profit", "profit", "pnl", "PnL", "return"])
result_col = find_column(bets, ["Won", "won", "result", "Result"])
prob_col = find_column(bets, ["ModelProb", "model_prob", "prob", "probability", "selected_prob"])

# =========================
# Apply Filters
# =========================

filtered_bets = bets.copy()

if ev_col:
    filtered_bets = filtered_bets[filtered_bets[ev_col] >= ev_threshold]

if odds_col:
    filtered_bets = filtered_bets[
        (filtered_bets[odds_col] >= min_odds) &
        (filtered_bets[odds_col] < max_odds)
    ]

# =========================
# Metrics
# =========================

st.header("🏆 Best Strategy Summary")

total_bets = len(filtered_bets)

if profit_col:
    total_profit = filtered_bets[profit_col].sum()
else:
    total_profit = 0

if total_bets > 0:
    roi = total_profit / total_bets
else:
    roi = 0

if result_col:
    win_rate = filtered_bets[result_col].mean()
else:
    win_rate = None

if odds_col and total_bets > 0:
    avg_odds = filtered_bets[odds_col].mean()
else:
    avg_odds = None

if ev_col and total_bets > 0:
    avg_ev = filtered_bets[ev_col].mean()
else:
    avg_ev = None

if prob_col and total_bets > 0:
    avg_prob = filtered_bets[prob_col].mean()
else:
    avg_prob = None

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Bets", f"{total_bets}")

with col2:
    if win_rate is not None:
        st.metric("Win Rate", f"{win_rate:.2%}")
    else:
        st.metric("Win Rate", "N/A")

with col3:
    st.metric("Profit", f"{total_profit:+.2f} units")

with col4:
    st.metric("ROI", f"{roi:.2%}")

col5, col6, col7, col8 = st.columns(4)

with col5:
    if avg_odds is not None:
        st.metric("Average Odds", f"{avg_odds:.2f}")
    else:
        st.metric("Average Odds", "N/A")

with col6:
    if avg_prob is not None:
        st.metric("Average Model Probability", f"{avg_prob:.2%}")
    else:
        st.metric("Average Model Probability", "N/A")

with col7:
    if avg_ev is not None:
        st.metric("Average EV", f"{avg_ev:.2%}")
    else:
        st.metric("Average EV", "N/A")

with col8:
    if profit_col and len(filtered_bets) > 0:
        running_profit = filtered_bets[profit_col].cumsum()
        running_peak = running_profit.cummax()
        drawdown = running_peak - running_profit
        max_drawdown = drawdown.max()
        st.metric("Max Drawdown", f"{max_drawdown:.2f} units")
    else:
        st.metric("Max Drawdown", "N/A")

st.success(
    "Validated strategy: EV >= 15%, odds >= 2.00, odds < 3.00, flat stake = 1 unit."
)

# =========================
# Model Summary
# =========================

st.header("🤖 Model Summary")

st.dataframe(
    model_summary,
    use_container_width=True
)

# =========================
# Season Results
# =========================

st.header("📅 Season-by-Season Results")

st.dataframe(
    season_results,
    use_container_width=True
)

season_col = find_column(season_results, ["TestSeason", "Season", "season"])
season_profit_col = find_column(season_results, ["Profit", "profit", "pnl", "PnL"])

if season_col and season_profit_col:
    fig_profit = px.bar(
        season_results,
        x=season_col,
        y=season_profit_col,
        title="Profit by Season"
    )
    st.plotly_chart(fig_profit, use_container_width=True)

# =========================
# Equity Curve
# =========================

st.header("📈 Equity Curve")

fig_equity = px.line(
    equity_curve,
    x="BetNumber",
    y="CumulativeProfit",
    title="Cumulative Profit Over Time"
)

st.plotly_chart(fig_equity, use_container_width=True)

# =========================
# Qualified Bets
# =========================

st.header("🎯 Strategy-Qualified Bets")

st.write(f"Showing {len(filtered_bets)} bets after current filters.")

st.dataframe(
    filtered_bets,
    use_container_width=True
)

# =========================
# Column Debugger
# =========================

with st.expander("Show detected columns"):
    st.write("Bets columns:")
    st.write(list(bets.columns))

    st.write("Season result columns:")
    st.write(list(season_results.columns))

    st.write("Model summary columns:")
    st.write(list(model_summary.columns))