import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from dashboard_corner_tab import render_corner_markets_tab


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Soccer Edge Lab",
    page_icon="⚽",
    layout="wide",
)


# =========================
# File Paths
# =========================

BEST_BETS_FILE = Path("data/processed/best_strategy_bets_v1.csv")
SEASON_SUMMARY_FILE = Path("data/processed/best_strategy_season_summary_v1.csv")
MODEL_SUMMARY_FILE = Path("data/processed/best_strategy_model_summary_v1.csv")
GRID_SEARCH_FILE = Path("data/processed/walk_forward_grid_search_v1.csv")
FEATURE_STORE_FILE = Path("data/processed/feature_store_v1.csv")

MULTI_LEAGUE_MODEL_FILE = Path("data/processed/multi_league_model_summary_v1.csv")
MULTI_LEAGUE_SEASON_FILE = Path("data/processed/multi_league_strategy_season_summary_v1.csv")
MULTI_LEAGUE_LEAGUE_FILE = Path("data/processed/multi_league_strategy_league_summary_v1.csv")
MULTI_LEAGUE_BETS_FILE = Path("data/processed/multi_league_strategy_bets_v1.csv")


# =========================
# Helper Functions
# =========================

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def format_percent(value):
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "N/A"


def format_units(value):
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def format_odds(value):
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def calculate_max_drawdown(profits: pd.Series) -> float:
    if len(profits) == 0:
        return 0

    cumulative = profits.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative

    return drawdown.max()


def create_download_button(df: pd.DataFrame, filename: str, label: str):
    if len(df) == 0:
        return

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def show_missing_file_warning():
    st.warning(
        """
        Some required files are missing.

        Run these scripts first:
        """
    )

    st.code(
        """
python scripts/backtest_best_strategy.py
python scripts/walk_forward_grid_search.py
python scripts/backtest_corner_markets.py
python scripts/analyze_corner_edges.py
        """,
        language="powershell",
    )


def clean_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def make_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes", "won"])


# =========================
# Load Data
# =========================

bets = clean_date_column(load_csv(BEST_BETS_FILE))
season_summary = load_csv(SEASON_SUMMARY_FILE)
model_summary = load_csv(MODEL_SUMMARY_FILE)
grid_search = load_csv(GRID_SEARCH_FILE)
feature_store = clean_date_column(load_csv(FEATURE_STORE_FILE))

multi_model = load_csv(MULTI_LEAGUE_MODEL_FILE)
multi_season = load_csv(MULTI_LEAGUE_SEASON_FILE)
multi_league = load_csv(MULTI_LEAGUE_LEAGUE_FILE)
multi_bets = clean_date_column(load_csv(MULTI_LEAGUE_BETS_FILE))


# =========================
# Header
# =========================

st.title("⚽ Soccer Edge Lab")
st.caption("Machine learning, Elo ratings, expected value betting, backtesting, and multi-market soccer analytics.")


# =========================
# Data Availability Check
# =========================

if len(bets) == 0:
    show_missing_file_warning()
    st.stop()


# =========================
# Sidebar Filters
# =========================

st.sidebar.header("Strategy Filters")

ev_threshold = st.sidebar.slider(
    "Minimum EV",
    min_value=0.00,
    max_value=0.50,
    value=0.15,
    step=0.01,
)

min_odds = st.sidebar.slider(
    "Minimum Odds",
    min_value=1.00,
    max_value=5.00,
    value=2.00,
    step=0.05,
)

max_odds = st.sidebar.slider(
    "Maximum Odds",
    min_value=1.00,
    max_value=10.00,
    value=3.00,
    step=0.05,
)

available_seasons = sorted(bets["TestSeason"].dropna().unique()) if "TestSeason" in bets.columns else []

selected_seasons = st.sidebar.multiselect(
    "Seasons",
    options=available_seasons,
    default=available_seasons,
)

filtered_bets = bets.copy()

if "EV" in filtered_bets.columns:
    filtered_bets = filtered_bets[filtered_bets["EV"] >= ev_threshold]

if "Odds" in filtered_bets.columns:
    filtered_bets = filtered_bets[
        (filtered_bets["Odds"] >= min_odds)
        & (filtered_bets["Odds"] < max_odds)
    ]

if selected_seasons and "TestSeason" in filtered_bets.columns:
    filtered_bets = filtered_bets[filtered_bets["TestSeason"].isin(selected_seasons)]


# =========================
# Main Tabs
# =========================

(
    tab_overview,
    tab_model,
    tab_calibration,
    tab_features,
    tab_strategy,
    tab_comparison,
    tab_simulator,
    tab_bankroll,
    tab_bets,
    tab_risk,
    tab_corners,
) = st.tabs([
    "🏠 Overview",
    "🤖 Model Performance",
    "📏 Model Calibration",
    "🧠 Feature Importance",
    "📊 Strategy Results",
    "🏁 Strategy Comparison",
    "🧪 Strategy Simulator",
    "💰 Bankroll Simulator",
    "🎯 Bet Explorer",
    "⚠️ Risk / Drawdown",
    "🟨 Corner Markets",
])


# =========================
# Tab 1: Overview
# =========================

with tab_overview:
    st.header("🏠 Overview")

    total_bets = len(filtered_bets)
    total_profit = filtered_bets["Profit"].sum() if "Profit" in filtered_bets.columns else 0
    roi = total_profit / total_bets if total_bets > 0 else 0

    if "Won" in filtered_bets.columns:
        won_series = make_bool_series(filtered_bets["Won"])
        win_rate = won_series.mean()
    else:
        win_rate = 0

    avg_odds = filtered_bets["Odds"].mean() if "Odds" in filtered_bets.columns and total_bets > 0 else 0
    avg_ev = filtered_bets["EV"].mean() if "EV" in filtered_bets.columns and total_bets > 0 else 0
    max_drawdown = calculate_max_drawdown(filtered_bets["Profit"]) if "Profit" in filtered_bets.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Filtered Bets", f"{total_bets:,}")
    col2.metric("Profit", f"{total_profit:.2f} units")
    col3.metric("ROI", f"{roi:.2%}")
    col4.metric("Win Rate", f"{win_rate:.2%}")

    col5, col6, col7 = st.columns(3)

    col5.metric("Average Odds", f"{avg_odds:.2f}")
    col6.metric("Average EV", f"{avg_ev:.2%}")
    col7.metric("Max Drawdown", f"{max_drawdown:.2f} units")

    st.subheader("Equity Curve")

    if "Profit" in filtered_bets.columns and len(filtered_bets) > 0:
        equity_df = filtered_bets.copy()
        equity_df = equity_df.sort_values("Date") if "Date" in equity_df.columns else equity_df.copy()
        equity_df["CumulativeProfit"] = equity_df["Profit"].cumsum()
        equity_df["BetNumber"] = range(1, len(equity_df) + 1)

        fig = px.line(
            equity_df,
            x="BetNumber",
            y="CumulativeProfit",
            title="Cumulative Profit",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bets match the current filters.")

    st.subheader("Current Best Strategy")

    st.code(
        """
EV >= 15%
Odds >= 2.00
Odds < 3.00
Flat stake = 1 unit
        """,
        language="text",
    )


# =========================
# Tab 2: Model Performance
# =========================

with tab_model:
    st.header("🤖 Model Performance")

    if len(model_summary) > 0:
        st.subheader("La Liga Model Summary")

        display_model = model_summary.copy()

        percent_cols = ["Accuracy", "WinRate", "AverageEV", "ROI"]
        decimal_cols = ["LogLoss", "AverageOdds", "Profit", "MaxDrawdown"]

        for col in percent_cols:
            if col in display_model.columns:
                display_model[col] = pd.to_numeric(display_model[col], errors="coerce").apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else ""
                )

        for col in decimal_cols:
            if col in display_model.columns:
                display_model[col] = pd.to_numeric(display_model[col], errors="coerce").apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ""
                )

        st.dataframe(display_model, use_container_width=True)

        chart_df = model_summary.copy()

        if "TestSeason" in chart_df.columns and "Accuracy" in chart_df.columns:
            fig = px.line(
                chart_df,
                x="TestSeason",
                y="Accuracy",
                markers=True,
                title="Model Accuracy by Season",
            )
            st.plotly_chart(fig, use_container_width=True)

        if "TestSeason" in chart_df.columns and "LogLoss" in chart_df.columns:
            fig = px.line(
                chart_df,
                x="TestSeason",
                y="LogLoss",
                markers=True,
                title="Log Loss by Season",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Model summary file not found.")

    st.subheader("Multi-League Model Summary")

    if len(multi_model) > 0:
        st.dataframe(multi_model, use_container_width=True)
    else:
        st.info("Multi-league model summary not found yet.")


# =========================
# Tab 3: Model Calibration
# =========================

with tab_calibration:
    st.header("📏 Model Calibration")

    if len(bets) > 0 and "ModelProb" in bets.columns and "Won" in bets.columns:
        calibration_df = bets.copy()
        calibration_df["ModelProb"] = pd.to_numeric(calibration_df["ModelProb"], errors="coerce")
        calibration_df["WonBool"] = make_bool_series(calibration_df["Won"])

        calibration_df = calibration_df.dropna(subset=["ModelProb"])

        calibration_df["ProbabilityBucket"] = pd.cut(
            calibration_df["ModelProb"],
            bins=[0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00],
        )

        bucket_summary = (
            calibration_df
            .groupby("ProbabilityBucket", observed=True)
            .agg(
                Bets=("WonBool", "count"),
                ActualWinRate=("WonBool", "mean"),
                AvgModelProb=("ModelProb", "mean"),
            )
            .reset_index()
        )

        st.dataframe(bucket_summary, use_container_width=True)

        fig = px.line(
            bucket_summary,
            x="AvgModelProb",
            y="ActualWinRate",
            markers=True,
            title="Model Probability vs Actual Win Rate",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Calibration requires ModelProb and Won columns.")


# =========================
# Tab 4: Feature Importance
# =========================

with tab_features:
    st.header("🧠 Feature Importance")

    feature_columns = [
        "HomeElo",
        "AwayElo",
        "EloDiff",
        "HomeTeamHomeElo",
        "AwayTeamAwayElo",
        "HomeAwayEloDiff",
        "HomeFormPoints",
        "AwayFormPoints",
        "HomeFormGoalsFor",
        "AwayFormGoalsFor",
        "HomeFormGoalsAgainst",
        "AwayFormGoalsAgainst",
        "HomeAdvantage",
        "FormPointDiff",
        "FormGoalsForDiff",
        "FormGoalsAgainstDiff",
        "HomeAttackRating",
        "AwayAttackRating",
        "HomeDefenseRating",
        "AwayDefenseRating",
        "HomeGoalDiffRating",
        "AwayGoalDiffRating",
        "AttackRatingDiff",
        "DefenseRatingDiff",
        "GoalDiffRatingDiff",
    ]

    if len(feature_store) > 0 and "Result" in feature_store.columns:
        available_features = [col for col in feature_columns if col in feature_store.columns]

        if len(available_features) > 0:
            data = feature_store.dropna(subset=available_features + ["Result"]).copy()

            X = data[available_features]
            y = data["Result"]

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=5000)),
            ])

            model.fit(X, y)

            classifier = model.named_steps["classifier"]

            importance_rows = []

            for class_index, class_label in enumerate(classifier.classes_):
                for feature, coefficient in zip(available_features, classifier.coef_[class_index]):
                    importance_rows.append({
                        "Class": class_label,
                        "Feature": feature,
                        "Coefficient": coefficient,
                        "AbsCoefficient": abs(coefficient),
                    })

            importance_df = pd.DataFrame(importance_rows)

            overall_importance = (
                importance_df
                .groupby("Feature")
                .agg(Importance=("AbsCoefficient", "mean"))
                .reset_index()
                .sort_values("Importance", ascending=False)
            )

            st.subheader("Overall Feature Importance")
            st.dataframe(overall_importance, use_container_width=True)

            fig = px.bar(
                overall_importance.head(20).sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 20 Feature Importance Scores",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Class-Specific Coefficients")
            st.dataframe(importance_df.sort_values("AbsCoefficient", ascending=False), use_container_width=True)

        else:
            st.info("No matching feature columns found.")
    else:
        st.info("Feature store file not found.")


# =========================
# Tab 5: Strategy Results
# =========================

with tab_strategy:
    st.header("📊 Strategy Results")

    if len(season_summary) > 0:
        st.subheader("La Liga Season Results")
        st.dataframe(season_summary, use_container_width=True)

        if "TestSeason" in season_summary.columns and "Profit" in season_summary.columns:
            fig = px.bar(
                season_summary,
                x="TestSeason",
                y="Profit",
                title="Profit by Season",
            )
            st.plotly_chart(fig, use_container_width=True)

        if "TestSeason" in season_summary.columns and "ROI" in season_summary.columns:
            fig = px.bar(
                season_summary,
                x="TestSeason",
                y="ROI",
                title="ROI by Season",
            )
            st.plotly_chart(fig, use_container_width=True)

        create_download_button(
            season_summary,
            "best_strategy_season_summary_v1.csv",
            "Download season summary",
        )
    else:
        st.info("Season summary file not found.")

    st.subheader("Multi-League Results")

    if len(multi_league) > 0:
        st.dataframe(multi_league, use_container_width=True)

        if "League" in multi_league.columns and "Profit" in multi_league.columns:
            fig = px.bar(
                multi_league.sort_values("Profit"),
                x="Profit",
                y="League",
                orientation="h",
                title="Multi-League Profit by League",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Multi-league strategy files not found yet.")


# =========================
# Tab 6: Strategy Comparison
# =========================

with tab_comparison:
    st.header("🏁 Strategy Comparison")

    if len(grid_search) > 0:
        st.subheader("Walk-Forward Grid Search")

        st.dataframe(grid_search, use_container_width=True)

        if "ROI" in grid_search.columns and "Profit" in grid_search.columns:
            chart_df = grid_search.copy()
            chart_df["ROI"] = pd.to_numeric(chart_df["ROI"], errors="coerce")
            chart_df["Profit"] = pd.to_numeric(chart_df["Profit"], errors="coerce")

            fig = px.scatter(
                chart_df,
                x="ROI",
                y="Profit",
                size="Total Bets" if "Total Bets" in chart_df.columns else None,
                hover_data=chart_df.columns,
                title="Strategy ROI vs Profit",
            )

            st.plotly_chart(fig, use_container_width=True)

        create_download_button(
            grid_search,
            "walk_forward_grid_search_v1.csv",
            "Download grid search results",
        )
    else:
        st.info("Grid search file not found.")


# =========================
# Tab 7: Strategy Simulator
# =========================

with tab_simulator:
    st.header("🧪 Strategy Simulator")

    st.write("Use the sidebar filters to simulate different EV and odds rules.")

    total_bets = len(filtered_bets)
    total_profit = filtered_bets["Profit"].sum() if "Profit" in filtered_bets.columns else 0
    roi = total_profit / total_bets if total_bets > 0 else 0

    if "Won" in filtered_bets.columns:
        win_rate = make_bool_series(filtered_bets["Won"]).mean()
    else:
        win_rate = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Bets", f"{total_bets:,}")
    col2.metric("Profit", f"{total_profit:.2f}")
    col3.metric("ROI", f"{roi:.2%}")
    col4.metric("Win Rate", f"{win_rate:.2%}")

    st.dataframe(filtered_bets, use_container_width=True)

    create_download_button(
        filtered_bets,
        "filtered_strategy_bets.csv",
        "Download filtered bets",
    )


# =========================
# Tab 8: Bankroll Simulator
# =========================

with tab_bankroll:
    st.header("💰 Bankroll Simulator")

    starting_bankroll = st.number_input(
        "Starting bankroll",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0,
    )

    stake_size = st.number_input(
        "Flat stake per bet",
        min_value=1.0,
        max_value=1000.0,
        value=10.0,
        step=5.0,
    )

    if len(filtered_bets) > 0 and "Profit" in filtered_bets.columns:
        bankroll_df = filtered_bets.copy()
        bankroll_df = bankroll_df.sort_values("Date") if "Date" in bankroll_df.columns else bankroll_df.copy()

        bankroll_df["StakeAdjustedProfit"] = bankroll_df["Profit"] * stake_size
        bankroll_df["Bankroll"] = starting_bankroll + bankroll_df["StakeAdjustedProfit"].cumsum()
        bankroll_df["BetNumber"] = range(1, len(bankroll_df) + 1)

        ending_bankroll = bankroll_df["Bankroll"].iloc[-1]
        bankroll_profit = ending_bankroll - starting_bankroll
        bankroll_roi = bankroll_profit / starting_bankroll

        col1, col2, col3 = st.columns(3)

        col1.metric("Ending Bankroll", f"${ending_bankroll:,.2f}")
        col2.metric("Profit", f"${bankroll_profit:,.2f}")
        col3.metric("Bankroll ROI", f"{bankroll_roi:.2%}")

        fig = px.line(
            bankroll_df,
            x="BetNumber",
            y="Bankroll",
            title="Bankroll Simulation",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bets available for bankroll simulation.")


# =========================
# Tab 9: Bet Explorer
# =========================

with tab_bets:
    st.header("🎯 Bet Explorer")

    st.dataframe(filtered_bets, use_container_width=True)

    create_download_button(
        filtered_bets,
        "filtered_bets.csv",
        "Download filtered bets",
    )

    if len(filtered_bets) > 0:
        st.subheader("Last 20 Bets")
        st.dataframe(filtered_bets.tail(20), use_container_width=True)


# =========================
# Tab 10: Risk / Drawdown
# =========================

with tab_risk:
    st.header("⚠️ Risk / Drawdown")

    if len(filtered_bets) > 0 and "Profit" in filtered_bets.columns:
        risk_df = filtered_bets.copy()
        risk_df = risk_df.sort_values("Date") if "Date" in risk_df.columns else risk_df.copy()

        risk_df["CumulativeProfit"] = risk_df["Profit"].cumsum()
        risk_df["RunningPeak"] = risk_df["CumulativeProfit"].cummax()
        risk_df["Drawdown"] = risk_df["RunningPeak"] - risk_df["CumulativeProfit"]
        risk_df["BetNumber"] = range(1, len(risk_df) + 1)

        max_dd = risk_df["Drawdown"].max()

        col1, col2, col3 = st.columns(3)

        col1.metric("Max Drawdown", f"{max_dd:.2f} units")
        col2.metric("Worst Bet", f"{risk_df['Profit'].min():.2f} units")
        col3.metric("Best Bet", f"{risk_df['Profit'].max():.2f} units")

        fig = px.line(
            risk_df,
            x="BetNumber",
            y="Drawdown",
            title="Drawdown Over Time",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drawdown Table")
        st.dataframe(risk_df, use_container_width=True)
    else:
        st.info("No risk data available.")


# =========================
# Tab 11: Corner Markets
# =========================

with tab_corners:
    render_corner_markets_tab()