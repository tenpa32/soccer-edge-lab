import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Soccer Edge Lab",
    page_icon="⚽",
    layout="wide"
)

# =========================
# Header
# =========================

st.title("⚽ Soccer Edge Lab")
st.caption(
    "Model-based soccer betting analytics using Elo ratings, ML probabilities, "
    "expected value filtering, odds bands, and walk-forward backtesting."
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

# =========================
# Helpers
# =========================

def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def format_percent(value):
    return f"{value:.2%}"

def format_units(value):
    return f"{value:+.2f} units"

# =========================
# Column Detection
# =========================

ev_col = find_column(bets, ["EV", "ev", "expected_value", "ExpectedValue"])
odds_col = find_column(bets, ["Odds", "odds", "selected_odds", "bet_odds"])
profit_col = find_column(bets, ["Profit", "profit", "pnl", "PnL", "return"])
result_col = find_column(bets, ["Won", "won", "result", "Result"])
prob_col = find_column(bets, ["ModelProb", "model_prob", "prob", "probability", "selected_prob"])
season_col = find_column(bets, ["TestSeason", "Season", "season"])
date_col = find_column(bets, ["Date", "date", "MatchDate"])
home_col = find_column(bets, ["HomeTeam", "home_team", "Home"])
away_col = find_column(bets, ["AwayTeam", "away_team", "Away"])
bet_col = find_column(bets, ["Bet", "bet", "Selection", "selection"])
actual_result_col = find_column(bets, ["ActualResult", "actual_result", "Result"])

# =========================
# Build Equity Curve
# =========================

equity_curve = bets.copy()

if "CumulativeProfit" in equity_curve.columns:
    equity_curve["BetNumber"] = range(1, len(equity_curve) + 1)
else:
    equity_curve["CumulativeProfit"] = equity_curve[profit_col].cumsum()
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

if season_col:
    available_seasons = sorted(bets[season_col].dropna().unique())
    selected_seasons = st.sidebar.multiselect(
        "Seasons",
        options=available_seasons,
        default=available_seasons
    )
else:
    selected_seasons = []

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

if season_col and selected_seasons:
    filtered_bets = filtered_bets[filtered_bets[season_col].isin(selected_seasons)]

# =========================
# Metrics Calculation
# =========================

total_bets = len(filtered_bets)

if profit_col and total_bets > 0:
    total_profit = filtered_bets[profit_col].sum()
else:
    total_profit = 0

if total_bets > 0:
    roi = total_profit / total_bets
else:
    roi = 0

if result_col and total_bets > 0:
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

if profit_col and total_bets > 0:
    running_profit = filtered_bets[profit_col].cumsum()
    running_peak = running_profit.cummax()
    drawdown = running_peak - running_profit
    max_drawdown = drawdown.max()
else:
    max_drawdown = 0

# =========================
# Tabs
# =========================

tab_overview, tab_model, tab_strategy, tab_bets, tab_risk = st.tabs([
    "🏠 Overview",
    "🤖 Model Performance",
    "📊 Strategy Results",
    "🎯 Bet Explorer",
    "⚠️ Risk / Drawdown"
])

# =========================
# Tab 1: Overview
# =========================

with tab_overview:
    st.header("🏆 Best Strategy Overview")

    st.success(
        "Validated strategy: EV >= 15%, odds >= 2.00, odds < 3.00, flat stake = 1 unit."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Bets", f"{total_bets}")

    with col2:
        if win_rate is not None:
            st.metric("Win Rate", format_percent(win_rate))
        else:
            st.metric("Win Rate", "N/A")

    with col3:
        st.metric("Profit", format_units(total_profit))

    with col4:
        st.metric("ROI", format_percent(roi))

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if avg_odds is not None:
            st.metric("Average Odds", f"{avg_odds:.2f}")
        else:
            st.metric("Average Odds", "N/A")

    with col6:
        if avg_prob is not None:
            st.metric("Average Model Probability", format_percent(avg_prob))
        else:
            st.metric("Average Model Probability", "N/A")

    with col7:
        if avg_ev is not None:
            st.metric("Average EV", format_percent(avg_ev))
        else:
            st.metric("Average EV", "N/A")

    with col8:
        st.metric("Max Drawdown", f"{max_drawdown:.2f} units")

    st.divider()

    st.subheader("📈 Equity Curve")

    fig_equity_overview = px.line(
        equity_curve,
        x="BetNumber",
        y="CumulativeProfit",
        title="Cumulative Profit Over Time"
    )

    st.plotly_chart(fig_equity_overview, use_container_width=True)

# =========================
# Tab 2: Model Performance
# =========================

with tab_model:
    st.header("🤖 Model Performance")

    st.write(
        "This section shows the model's walk-forward performance by test season."
    )

    st.dataframe(
        model_summary,
        use_container_width=True
    )

    model_season_col = find_column(model_summary, ["TestSeason", "Season", "season"])
    accuracy_col = find_column(model_summary, ["Accuracy", "accuracy"])
    logloss_col = find_column(model_summary, ["LogLoss", "log_loss", "Log Loss"])

    if model_season_col and accuracy_col:
        st.subheader("Accuracy by Season")

        accuracy_data = model_summary.copy()

        if accuracy_data[accuracy_col].dtype == object:
            accuracy_data[accuracy_col] = (
                accuracy_data[accuracy_col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .astype(float) / 100
            )

        fig_accuracy = px.line(
            accuracy_data,
            x=model_season_col,
            y=accuracy_col,
            markers=True,
            title="Model Accuracy by Test Season"
        )

        st.plotly_chart(fig_accuracy, use_container_width=True)

    if model_season_col and logloss_col:
        st.subheader("Log Loss by Season")

        fig_logloss = px.line(
            model_summary,
            x=model_season_col,
            y=logloss_col,
            markers=True,
            title="Model Log Loss by Test Season"
        )

        st.plotly_chart(fig_logloss, use_container_width=True)

# =========================
# Tab 3: Strategy Results
# =========================

with tab_strategy:
    st.header("📊 Strategy Results")

    st.write(
        "This section shows season-by-season strategy performance from the best backtest."
    )

    st.dataframe(
        season_results,
        use_container_width=True
    )

    season_summary_col = find_column(season_results, ["TestSeason", "Season", "season"])
    season_profit_col = find_column(season_results, ["Profit", "profit", "pnl", "PnL"])
    season_roi_col = find_column(season_results, ["ROI", "roi"])
    season_bets_col = find_column(season_results, ["Bets", "bets"])
    season_drawdown_col = find_column(season_results, ["MaxDrawdown", "Max Drawdown", "max_drawdown"])

    if season_summary_col and season_profit_col:
        st.subheader("Profit by Season")

        fig_profit = px.bar(
            season_results,
            x=season_summary_col,
            y=season_profit_col,
            title="Profit by Season"
        )

        st.plotly_chart(fig_profit, use_container_width=True)

    if season_summary_col and season_roi_col:
        st.subheader("ROI by Season")

        roi_data = season_results.copy()

        if roi_data[season_roi_col].dtype == object:
            roi_data[season_roi_col] = (
                roi_data[season_roi_col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .astype(float) / 100
            )

        fig_roi = px.bar(
            roi_data,
            x=season_summary_col,
            y=season_roi_col,
            title="ROI by Season"
        )

        st.plotly_chart(fig_roi, use_container_width=True)

    if season_summary_col and season_bets_col:
        st.subheader("Number of Bets by Season")

        fig_bets = px.bar(
            season_results,
            x=season_summary_col,
            y=season_bets_col,
            title="Bets by Season"
        )

        st.plotly_chart(fig_bets, use_container_width=True)

# =========================
# Tab 4: Bet Explorer
# =========================

with tab_bets:
    st.header("🎯 Bet Explorer")

    st.write(f"Showing {len(filtered_bets)} bets after current filters.")

    search_text = st.text_input(
        "Search team name",
        value=""
    )

    explorer_bets = filtered_bets.copy()

    if search_text and home_col and away_col:
        explorer_bets = explorer_bets[
            explorer_bets[home_col].astype(str).str.contains(search_text, case=False, na=False) |
            explorer_bets[away_col].astype(str).str.contains(search_text, case=False, na=False)
        ]

    display_columns = []

    for col in [
        season_col,
        date_col,
        home_col,
        away_col,
        bet_col,
        prob_col,
        odds_col,
        ev_col,
        actual_result_col,
        profit_col,
        result_col,
        "CumulativeProfit"
    ]:
        if col and col in explorer_bets.columns and col not in display_columns:
            display_columns.append(col)

    if display_columns:
        st.dataframe(
            explorer_bets[display_columns],
            use_container_width=True
        )
    else:
        st.dataframe(
            explorer_bets,
            use_container_width=True
        )

    if ev_col and odds_col:
        st.subheader("EV vs Odds")

        fig_ev_odds = px.scatter(
            explorer_bets,
            x=odds_col,
            y=ev_col,
            hover_data=[home_col, away_col, bet_col, profit_col],
            title="Expected Value vs Odds"
        )

        st.plotly_chart(fig_ev_odds, use_container_width=True)

# =========================
# Tab 5: Risk / Drawdown
# =========================

with tab_risk:
    st.header("⚠️ Risk / Drawdown")

    risk_data = bets.copy()

    if profit_col:
        risk_data["BetNumber"] = range(1, len(risk_data) + 1)
        risk_data["RunningProfit"] = risk_data[profit_col].cumsum()
        risk_data["RunningPeak"] = risk_data["RunningProfit"].cummax()
        risk_data["Drawdown"] = risk_data["RunningPeak"] - risk_data["RunningProfit"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Max Drawdown", f"{risk_data['Drawdown'].max():.2f} units")

        with col2:
            losing_bets = (risk_data[profit_col] < 0).sum()
            st.metric("Losing Bets", f"{losing_bets}")

        with col3:
            winning_bets = (risk_data[profit_col] > 0).sum()
            st.metric("Winning Bets", f"{winning_bets}")

        st.subheader("Drawdown Over Time")

        fig_drawdown = px.line(
            risk_data,
            x="BetNumber",
            y="Drawdown",
            title="Drawdown Over Time"
        )

        st.plotly_chart(fig_drawdown, use_container_width=True)

        st.subheader("Profit Distribution")

        fig_profit_dist = px.histogram(
            risk_data,
            x=profit_col,
            title="Profit Distribution per Bet"
        )

        st.plotly_chart(fig_profit_dist, use_container_width=True)

        if season_col:
            st.subheader("Drawdown by Season")

            season_risk = []

            for season, group in risk_data.groupby(season_col):
                group = group.copy()
                group["SeasonRunningProfit"] = group[profit_col].cumsum()
                group["SeasonRunningPeak"] = group["SeasonRunningProfit"].cummax()
                group["SeasonDrawdown"] = group["SeasonRunningPeak"] - group["SeasonRunningProfit"]

                season_risk.append({
                    "Season": season,
                    "Bets": len(group),
                    "Profit": group[profit_col].sum(),
                    "MaxDrawdown": group["SeasonDrawdown"].max()
                })

            season_risk_df = pd.DataFrame(season_risk)

            st.dataframe(
                season_risk_df,
                use_container_width=True
            )

            fig_season_drawdown = px.bar(
                season_risk_df,
                x="Season",
                y="MaxDrawdown",
                title="Max Drawdown by Season"
            )

            st.plotly_chart(fig_season_drawdown, use_container_width=True)

    else:
        st.warning("Profit column not found, so risk charts cannot be calculated.")

# =========================
# Debug Section
# =========================

with st.expander("Developer Debug: Show detected columns"):
    st.write("Bets columns:")
    st.write(list(bets.columns))

    st.write("Season result columns:")
    st.write(list(season_results.columns))

    st.write("Model summary columns:")
    st.write(list(model_summary.columns))