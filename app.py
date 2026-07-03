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
# Safety Checks
# =========================

required_columns = {
    "EV": ev_col,
    "Odds": odds_col,
    "Profit": profit_col,
    "Won": result_col,
    "ModelProb": prob_col,
    "Season": season_col,
}

missing_required_columns = [
    display_name for display_name, detected_column in required_columns.items()
    if detected_column is None
]

if missing_required_columns:
    st.error("Some required columns were not found in the bets file.")
    st.write("Missing detected columns:")
    for col in missing_required_columns:
        st.write(f"- {col}")

    st.write("Actual columns in bets file:")
    st.write(list(bets.columns))
    st.stop()

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

available_seasons = sorted(bets[season_col].dropna().unique())

selected_seasons = st.sidebar.multiselect(
    "Seasons",
    options=available_seasons,
    default=available_seasons
)

# =========================
# Apply Main Filters
# =========================

filtered_bets = bets.copy()

filtered_bets = filtered_bets[filtered_bets[ev_col] >= ev_threshold]

filtered_bets = filtered_bets[
    (filtered_bets[odds_col] >= min_odds) &
    (filtered_bets[odds_col] < max_odds)
]

if selected_seasons:
    filtered_bets = filtered_bets[filtered_bets[season_col].isin(selected_seasons)]

# =========================
# Metrics Function
# =========================

def calculate_metrics(df):
    total_bets = len(df)

    if total_bets == 0:
        return {
            "total_bets": 0,
            "total_profit": 0,
            "roi": 0,
            "win_rate": None,
            "avg_odds": None,
            "avg_ev": None,
            "avg_prob": None,
            "max_drawdown": 0,
        }

    total_profit = df[profit_col].sum()
    roi = total_profit / total_bets
    win_rate = df[result_col].mean()
    avg_odds = df[odds_col].mean()
    avg_ev = df[ev_col].mean()
    avg_prob = df[prob_col].mean()

    running_profit = df[profit_col].cumsum()
    running_peak = running_profit.cummax()
    drawdown = running_peak - running_profit
    max_drawdown = drawdown.max()

    return {
        "total_bets": total_bets,
        "total_profit": total_profit,
        "roi": roi,
        "win_rate": win_rate,
        "avg_odds": avg_odds,
        "avg_ev": avg_ev,
        "avg_prob": avg_prob,
        "max_drawdown": max_drawdown,
    }


main_metrics = calculate_metrics(filtered_bets)

# =========================
# Tabs
# =========================

tab_overview, tab_model, tab_strategy, tab_simulator, tab_bets, tab_risk = st.tabs([
    "🏠 Overview",
    "🤖 Model Performance",
    "📊 Strategy Results",
    "🧪 Strategy Simulator",
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
        st.metric("Total Bets", f"{main_metrics['total_bets']}")

    with col2:
        if main_metrics["win_rate"] is not None:
            st.metric("Win Rate", format_percent(main_metrics["win_rate"]))
        else:
            st.metric("Win Rate", "N/A")

    with col3:
        st.metric("Profit", format_units(main_metrics["total_profit"]))

    with col4:
        st.metric("ROI", format_percent(main_metrics["roi"]))

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if main_metrics["avg_odds"] is not None:
            st.metric("Average Odds", f"{main_metrics['avg_odds']:.2f}")
        else:
            st.metric("Average Odds", "N/A")

    with col6:
        if main_metrics["avg_prob"] is not None:
            st.metric("Average Model Probability", format_percent(main_metrics["avg_prob"]))
        else:
            st.metric("Average Model Probability", "N/A")

    with col7:
        if main_metrics["avg_ev"] is not None:
            st.metric("Average EV", format_percent(main_metrics["avg_ev"]))
        else:
            st.metric("Average EV", "N/A")

    with col8:
        st.metric("Max Drawdown", f"{main_metrics['max_drawdown']:.2f} units")

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

    st.write("This section shows the model's walk-forward performance by test season.")

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

    st.write("This section shows season-by-season strategy performance from the best backtest.")

    st.dataframe(
        season_results,
        use_container_width=True
    )

    season_summary_col = find_column(season_results, ["TestSeason", "Season", "season"])
    season_profit_col = find_column(season_results, ["Profit", "profit", "pnl", "PnL"])
    season_roi_col = find_column(season_results, ["ROI", "roi"])
    season_bets_col = find_column(season_results, ["Bets", "bets"])

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
# Tab 4: Strategy Simulator
# =========================

with tab_simulator:
    st.header("🧪 Strategy Simulator")

    st.write(
        "Test different EV and odds filters using the existing backtest bet universe. "
        "This does not retrain the model — it recalculates strategy performance from saved model predictions."
    )

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        sim_ev_threshold = st.slider(
            "Simulator Minimum EV",
            min_value=0.00,
            max_value=0.75,
            value=0.15,
            step=0.01,
            key="sim_ev_threshold"
        )

    with sim_col2:
        sim_min_odds = st.slider(
            "Simulator Minimum Odds",
            min_value=1.00,
            max_value=5.00,
            value=2.00,
            step=0.05,
            key="sim_min_odds"
        )

    with sim_col3:
        sim_max_odds = st.slider(
            "Simulator Maximum Odds",
            min_value=1.00,
            max_value=10.00,
            value=3.00,
            step=0.05,
            key="sim_max_odds"
        )

    sim_selected_seasons = st.multiselect(
        "Simulator Seasons",
        options=available_seasons,
        default=available_seasons,
        key="sim_selected_seasons"
    )

    simulated_bets = bets.copy()

    simulated_bets = simulated_bets[
        simulated_bets[ev_col] >= sim_ev_threshold
    ]

    simulated_bets = simulated_bets[
        (simulated_bets[odds_col] >= sim_min_odds) &
        (simulated_bets[odds_col] < sim_max_odds)
    ]

    if sim_selected_seasons:
        simulated_bets = simulated_bets[
            simulated_bets[season_col].isin(sim_selected_seasons)
        ]

    sim_metrics = calculate_metrics(simulated_bets)

    st.subheader("Simulator Results")

    sim_m1, sim_m2, sim_m3, sim_m4 = st.columns(4)

    with sim_m1:
        st.metric("Bets", f"{sim_metrics['total_bets']}")

    with sim_m2:
        if sim_metrics["win_rate"] is not None:
            st.metric("Win Rate", format_percent(sim_metrics["win_rate"]))
        else:
            st.metric("Win Rate", "N/A")

    with sim_m3:
        st.metric("Profit", format_units(sim_metrics["total_profit"]))

    with sim_m4:
        st.metric("ROI", format_percent(sim_metrics["roi"]))

    sim_m5, sim_m6, sim_m7, sim_m8 = st.columns(4)

    with sim_m5:
        if sim_metrics["avg_odds"] is not None:
            st.metric("Average Odds", f"{sim_metrics['avg_odds']:.2f}")
        else:
            st.metric("Average Odds", "N/A")

    with sim_m6:
        if sim_metrics["avg_prob"] is not None:
            st.metric("Average Model Probability", format_percent(sim_metrics["avg_prob"]))
        else:
            st.metric("Average Model Probability", "N/A")

    with sim_m7:
        if sim_metrics["avg_ev"] is not None:
            st.metric("Average EV", format_percent(sim_metrics["avg_ev"]))
        else:
            st.metric("Average EV", "N/A")

    with sim_m8:
        st.metric("Max Drawdown", f"{sim_metrics['max_drawdown']:.2f} units")

    st.divider()

    if sim_metrics["total_bets"] > 0:
        simulated_equity = simulated_bets.copy()
        simulated_equity["SimBetNumber"] = range(1, len(simulated_equity) + 1)
        simulated_equity["SimCumulativeProfit"] = simulated_equity[profit_col].cumsum()

        st.subheader("Simulated Equity Curve")

        fig_sim_equity = px.line(
            simulated_equity,
            x="SimBetNumber",
            y="SimCumulativeProfit",
            title="Simulated Cumulative Profit"
        )

        st.plotly_chart(fig_sim_equity, use_container_width=True)

        st.subheader("Simulated Profit by Season")

        simulated_season_summary = (
            simulated_bets
            .groupby(season_col)
            .agg(
                Bets=(profit_col, "count"),
                Profit=(profit_col, "sum"),
                AverageOdds=(odds_col, "mean"),
                AverageEV=(ev_col, "mean"),
                WinRate=(result_col, "mean")
            )
            .reset_index()
        )

        simulated_season_summary["ROI"] = (
            simulated_season_summary["Profit"] /
            simulated_season_summary["Bets"]
        )

        st.dataframe(
            simulated_season_summary,
            use_container_width=True
        )

        fig_sim_season_profit = px.bar(
            simulated_season_summary,
            x=season_col,
            y="Profit",
            title="Simulated Profit by Season"
        )

        st.plotly_chart(fig_sim_season_profit, use_container_width=True)

        st.subheader("Simulated Qualified Bets")

        st.dataframe(
            simulated_bets,
            use_container_width=True
        )

    else:
        st.warning("No bets match the selected simulator filters.")

# =========================
# Tab 5: Bet Explorer
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

        hover_columns = []

        for col in [home_col, away_col, bet_col, profit_col]:
            if col and col in explorer_bets.columns:
                hover_columns.append(col)

        fig_ev_odds = px.scatter(
            explorer_bets,
            x=odds_col,
            y=ev_col,
            hover_data=hover_columns,
            title="Expected Value vs Odds"
        )

        st.plotly_chart(fig_ev_odds, use_container_width=True)

# =========================
# Tab 6: Risk / Drawdown
# =========================

with tab_risk:
    st.header("⚠️ Risk / Drawdown")

    risk_data = bets.copy()

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