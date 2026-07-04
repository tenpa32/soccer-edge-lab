import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
    "expected value filtering, odds bands, walk-forward backtesting, bankroll simulation, "
    "model calibration, and feature explainability."
)

# =========================
# File Paths
# =========================

DATA_DIR = Path("data/processed")

BETS_FILE = DATA_DIR / "best_strategy_bets_v1.csv"
SEASON_FILE = DATA_DIR / "best_strategy_season_summary_v1.csv"
MODEL_FILE = DATA_DIR / "best_strategy_model_summary_v1.csv"
GRID_FILE = DATA_DIR / "walk_forward_grid_search_v1.csv"
FEATURE_STORE_FILE = DATA_DIR / "feature_store_v1.csv"

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

if GRID_FILE.exists():
    grid_results = load_csv(GRID_FILE)
else:
    grid_results = None

if FEATURE_STORE_FILE.exists():
    feature_store = load_csv(FEATURE_STORE_FILE)
else:
    feature_store = None

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


def parse_percent_column(series):
    if series.dtype == object:
        return (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .astype(float) / 100
        )
    return series


def make_bool_series(series):
    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "won", "win"])
    )


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
    win_rate = make_bool_series(df[result_col]).mean()
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

tab_overview, tab_model, tab_calibration, tab_features, tab_strategy, tab_comparison, tab_simulator, tab_bankroll, tab_bets, tab_risk = st.tabs([
    "🏠 Overview",
    "🤖 Model Performance",
    "📏 Model Calibration",
    "🧠 Feature Importance",
    "📊 Strategy Results",
    "🏁 Strategy Comparison",
    "🧪 Strategy Simulator",
    "💰 Bankroll Simulator",
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
# Tab 3: Model Calibration
# =========================

with tab_calibration:
    st.header("📏 Model Calibration")

    st.write(
        "Calibration checks whether the model's predicted probabilities match real outcomes. "
        "Example: when the model says 60%, the bet should win close to 60% of the time."
    )

    calibration_source = st.radio(
        "Calibration Data",
        options=["All saved best-strategy bets", "Current sidebar-filtered bets"],
        horizontal=True
    )

    if calibration_source == "All saved best-strategy bets":
        calibration_data = bets.copy()
    else:
        calibration_data = filtered_bets.copy()

    calibration_data = calibration_data.dropna(subset=[prob_col, result_col]).copy()
    calibration_data[prob_col] = pd.to_numeric(calibration_data[prob_col], errors="coerce")
    calibration_data = calibration_data.dropna(subset=[prob_col]).copy()
    calibration_data["WonBool"] = make_bool_series(calibration_data[result_col]).astype(int)

    if len(calibration_data) == 0:
        st.warning("No calibration data available for the selected filters.")
    else:
        calibration_data["ProbabilityBin"] = pd.cut(
            calibration_data[prob_col],
            bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=[
                "0-10%",
                "10-20%",
                "20-30%",
                "30-40%",
                "40-50%",
                "50-60%",
                "60-70%",
                "70-80%",
                "80-90%",
                "90-100%",
            ],
            include_lowest=True
        )

        calibration_summary = (
            calibration_data
            .groupby("ProbabilityBin", observed=False)
            .agg(
                Bets=("WonBool", "count"),
                AvgPredictedProbability=(prob_col, "mean"),
                ActualWinRate=("WonBool", "mean")
            )
            .reset_index()
        )

        calibration_summary = calibration_summary[calibration_summary["Bets"] > 0].copy()
        calibration_summary["CalibrationGap"] = (
            calibration_summary["ActualWinRate"] -
            calibration_summary["AvgPredictedProbability"]
        )
        calibration_summary["AbsoluteGap"] = calibration_summary["CalibrationGap"].abs()

        brier_score = ((calibration_data[prob_col] - calibration_data["WonBool"]) ** 2).mean()
        average_predicted_probability = calibration_data[prob_col].mean()
        actual_win_rate = calibration_data["WonBool"].mean()

        if len(calibration_summary) > 0:
            expected_calibration_error = (
                calibration_summary["AbsoluteGap"] * calibration_summary["Bets"]
            ).sum() / calibration_summary["Bets"].sum()
        else:
            expected_calibration_error = 0

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Calibration Bets", f"{len(calibration_data)}")

        with c2:
            st.metric("Avg Predicted Prob", format_percent(average_predicted_probability))

        with c3:
            st.metric("Actual Win Rate", format_percent(actual_win_rate))

        with c4:
            st.metric("Brier Score", f"{brier_score:.4f}")

        c5, c6 = st.columns(2)

        with c5:
            st.metric("Calibration Error", format_percent(expected_calibration_error))

        with c6:
            st.metric("Overall Gap", format_percent(actual_win_rate - average_predicted_probability))

        st.divider()

        st.subheader("Calibration Table")

        st.dataframe(
            calibration_summary,
            use_container_width=True
        )

        st.subheader("Predicted Probability vs Actual Win Rate")

        fig_calibration = px.line(
            calibration_summary,
            x="AvgPredictedProbability",
            y="ActualWinRate",
            markers=True,
            hover_data=["ProbabilityBin", "Bets", "CalibrationGap"],
            title="Calibration Curve"
        )

        fig_calibration.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration"
        )

        st.plotly_chart(fig_calibration, use_container_width=True)

        st.subheader("Calibration Gap by Probability Band")

        fig_gap = px.bar(
            calibration_summary,
            x="ProbabilityBin",
            y="CalibrationGap",
            title="Actual Win Rate minus Predicted Probability"
        )

        st.plotly_chart(fig_gap, use_container_width=True)

        st.subheader("Prediction Confidence Distribution")

        fig_prob_dist = px.histogram(
            calibration_data,
            x=prob_col,
            nbins=20,
            title="Distribution of Model Probabilities"
        )

        st.plotly_chart(fig_prob_dist, use_container_width=True)

        st.info(
            "Lower Brier Score is better. A positive calibration gap means the model was too conservative in that band; "
            "a negative gap means the model was too optimistic."
        )

# =========================
# Tab 4: Feature Importance
# =========================

with tab_features:
    st.header("🧠 Feature Importance")

    st.write(
        "This tab retrains the Logistic Regression model on the saved feature store and shows which features have the strongest coefficients. "
        "This is a model explainability view, not proof of causality."
    )

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

    class_name_map = {
        0: "Away Win",
        1: "Draw",
        2: "Home Win",
        "0": "Away Win",
        "1": "Draw",
        "2": "Home Win",
    }

    if feature_store is None:
        st.warning(
            "Feature store file not found. Create this file first: data/processed/feature_store_v1.csv"
        )
        st.code(
            "python scripts/build_feature_store.py",
            language="powershell"
        )

    else:
        missing_feature_cols = [
            col for col in feature_columns
            if col not in feature_store.columns
        ]

        if "Result" not in feature_store.columns:
            st.error("Result column was not found in feature_store_v1.csv.")

        elif missing_feature_cols:
            st.error("Some feature columns were not found in feature_store_v1.csv.")
            st.write("Missing columns:")
            st.write(missing_feature_cols)

            st.write("Available columns:")
            st.write(list(feature_store.columns))

        else:
            model_data = feature_store.dropna(
                subset=feature_columns + ["Result"]
            ).copy()

            if len(model_data) == 0:
                st.warning("No usable rows available after dropping missing feature values.")

            else:
                X = model_data[feature_columns]
                y = model_data["Result"]

                explain_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=5000))
                ])

                explain_model.fit(X, y)

                classifier = explain_model.named_steps["classifier"]
                classes = classifier.classes_
                coefficients = classifier.coef_

                if coefficients.ndim == 1:
                    coefficient_df = pd.DataFrame({
                        "Feature": feature_columns,
                        "Class": "Model",
                        "Coefficient": coefficients
                    })
                else:
                    coefficient_rows = []

                    for class_index, class_value in enumerate(classes):
                        class_name = class_name_map.get(class_value, str(class_value))

                        for feature, coefficient in zip(feature_columns, coefficients[class_index]):
                            coefficient_rows.append({
                                "Feature": feature,
                                "Class": class_name,
                                "Coefficient": coefficient
                            })

                    coefficient_df = pd.DataFrame(coefficient_rows)

                coefficient_df["AbsoluteCoefficient"] = coefficient_df["Coefficient"].abs()

                overall_importance = (
                    coefficient_df
                    .groupby("Feature")
                    .agg(
                        MeanAbsoluteCoefficient=("AbsoluteCoefficient", "mean"),
                        MaxAbsoluteCoefficient=("AbsoluteCoefficient", "max")
                    )
                    .reset_index()
                    .sort_values("MeanAbsoluteCoefficient", ascending=False)
                )

                top_n = st.slider(
                    "Number of features to show",
                    min_value=5,
                    max_value=len(feature_columns),
                    value=15,
                    step=1
                )

                top_features = overall_importance.head(top_n)

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric("Training Rows", f"{len(model_data)}")

                with c2:
                    st.metric("Features", f"{len(feature_columns)}")

                with c3:
                    st.metric("Classes", f"{len(classes)}")

                with c4:
                    st.metric("Top Feature", top_features.iloc[0]["Feature"])

                st.divider()

                st.subheader("Overall Feature Importance")

                fig_feature_importance = px.bar(
                    top_features.sort_values("MeanAbsoluteCoefficient", ascending=True),
                    x="MeanAbsoluteCoefficient",
                    y="Feature",
                    orientation="h",
                    title="Top Features by Mean Absolute Logistic Regression Coefficient"
                )

                st.plotly_chart(fig_feature_importance, use_container_width=True)

                st.subheader("Overall Importance Table")

                st.dataframe(
                    overall_importance,
                    use_container_width=True
                )

                st.subheader("Class-Specific Coefficients")

                selected_class = st.selectbox(
                    "Select outcome class",
                    options=sorted(coefficient_df["Class"].unique())
                )

                class_coefficients = (
                    coefficient_df[coefficient_df["Class"] == selected_class]
                    .copy()
                    .sort_values("AbsoluteCoefficient", ascending=False)
                    .head(top_n)
                )

                fig_class_coefficients = px.bar(
                    class_coefficients.sort_values("Coefficient", ascending=True),
                    x="Coefficient",
                    y="Feature",
                    orientation="h",
                    title=f"Feature Coefficients for {selected_class}"
                )

                st.plotly_chart(fig_class_coefficients, use_container_width=True)

                st.dataframe(
                    class_coefficients,
                    use_container_width=True
                )

                st.subheader("Coefficient Heatmap")

                heatmap_data = coefficient_df.pivot(
                    index="Feature",
                    columns="Class",
                    values="Coefficient"
                ).reset_index()

                heatmap_long = coefficient_df.copy()

                fig_heatmap = px.density_heatmap(
                    heatmap_long,
                    x="Class",
                    y="Feature",
                    z="Coefficient",
                    title="Feature Coefficient Heatmap by Outcome Class"
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

                st.info(
                    "Positive coefficients increase the model's tendency toward that outcome class after scaling. "
                    "Negative coefficients push away from that outcome class. Larger absolute values mean the feature has more influence."
                )

# =========================
# Tab 5: Strategy Results
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
# Tab 6: Strategy Comparison
# =========================

with tab_comparison:
    st.header("🏁 Strategy Comparison")

    st.write(
        "Compare grid-search strategies across EV thresholds and odds bands. "
        "This helps identify which strategy is strongest and whether the current best strategy is robust."
    )

    if grid_results is None:
        st.warning(
            "Grid-search results file not found yet. "
            "Create this file first: data/processed/walk_forward_grid_search_v1.csv"
        )

        st.code(
            "python scripts/walk_forward_grid_search.py",
            language="powershell"
        )

    else:
        st.subheader("Raw Grid Search Results")

        st.dataframe(
            grid_results,
            use_container_width=True
        )

        grid_ev_col = find_column(grid_results, [
            "EV Threshold", "EVThreshold", "ev_threshold", "MinimumEV", "MinEV"
        ])

        grid_min_odds_col = find_column(grid_results, [
            "Min Odds", "MinOdds", "min_odds", "MinimumOdds", "Minimum Odds"
        ])

        grid_max_odds_col = find_column(grid_results, [
            "Max Odds", "MaxOdds", "max_odds", "MaximumOdds", "Maximum Odds"
        ])

        grid_bets_col = find_column(grid_results, [
            "Total Bets", "TotalBets", "Bets", "bets"
        ])

        grid_profit_col = find_column(grid_results, [
            "Profit", "profit", "TotalProfit", "Total Profit"
        ])

        grid_roi_col = find_column(grid_results, [
            "ROI", "roi"
        ])

        grid_winrate_col = find_column(grid_results, [
            "Win Rate", "WinRate", "win_rate"
        ])

        grid_drawdown_col = find_column(grid_results, [
            "Max Drawdown", "MaxDrawdown", "max_drawdown"
        ])

        comparison = grid_results.copy()

        if grid_roi_col:
            comparison[grid_roi_col] = parse_percent_column(comparison[grid_roi_col])

        if grid_winrate_col:
            comparison[grid_winrate_col] = parse_percent_column(comparison[grid_winrate_col])

        if grid_bets_col:
            max_bets = int(comparison[grid_bets_col].max())
        else:
            max_bets = 500

        minimum_bets_filter = st.slider(
            "Minimum Bets Required",
            min_value=0,
            max_value=max_bets,
            value=min(100, max_bets),
            step=10
        )

        if grid_bets_col:
            comparison = comparison[comparison[grid_bets_col] >= minimum_bets_filter]

        if len(comparison) == 0:
            st.warning("No strategies match the minimum bets filter.")

        else:
            if grid_roi_col and grid_profit_col:
                comparison = comparison.sort_values(
                    by=[grid_roi_col, grid_profit_col],
                    ascending=False
                )

            elif grid_profit_col:
                comparison = comparison.sort_values(
                    by=grid_profit_col,
                    ascending=False
                )

            top_strategy = comparison.iloc[0]

            st.subheader("Best Strategy From Grid")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                if grid_bets_col:
                    st.metric("Bets", f"{int(top_strategy[grid_bets_col])}")
                else:
                    st.metric("Bets", "N/A")

            with c2:
                if grid_profit_col:
                    st.metric("Profit", format_units(top_strategy[grid_profit_col]))
                else:
                    st.metric("Profit", "N/A")

            with c3:
                if grid_roi_col:
                    st.metric("ROI", format_percent(top_strategy[grid_roi_col]))
                else:
                    st.metric("ROI", "N/A")

            with c4:
                if grid_drawdown_col:
                    st.metric("Max Drawdown", f"{top_strategy[grid_drawdown_col]:.2f} units")
                else:
                    st.metric("Max Drawdown", "N/A")

            c5, c6, c7 = st.columns(3)

            with c5:
                if grid_ev_col:
                    st.metric("EV Threshold", f"{top_strategy[grid_ev_col]}")

            with c6:
                if grid_min_odds_col:
                    st.metric("Min Odds", f"{top_strategy[grid_min_odds_col]}")

            with c7:
                if grid_max_odds_col:
                    st.metric("Max Odds", f"{top_strategy[grid_max_odds_col]}")

            st.subheader("Top Strategies")

            st.dataframe(
                comparison.head(20),
                use_container_width=True
            )

            if grid_roi_col and grid_profit_col:
                st.subheader("ROI vs Profit")

                hover_cols = []

                for col in [
                    grid_ev_col,
                    grid_min_odds_col,
                    grid_max_odds_col,
                    grid_bets_col,
                    grid_winrate_col,
                    grid_drawdown_col
                ]:
                    if col and col in comparison.columns:
                        hover_cols.append(col)

                fig_roi_profit = px.scatter(
                    comparison,
                    x=grid_roi_col,
                    y=grid_profit_col,
                    size=grid_bets_col if grid_bets_col else None,
                    hover_data=hover_cols,
                    title="Strategy ROI vs Profit"
                )

                st.plotly_chart(fig_roi_profit, use_container_width=True)

            if grid_ev_col and grid_roi_col:
                st.subheader("EV Threshold vs ROI")

                fig_ev_roi = px.box(
                    comparison,
                    x=grid_ev_col,
                    y=grid_roi_col,
                    title="ROI Distribution by EV Threshold"
                )

                st.plotly_chart(fig_ev_roi, use_container_width=True)

            if grid_min_odds_col and grid_max_odds_col and grid_profit_col:
                st.subheader("Profit by Odds Band")

                comparison["OddsBand"] = (
                    comparison[grid_min_odds_col].astype(str)
                    + " - "
                    + comparison[grid_max_odds_col].astype(str)
                )

                odds_band_summary = (
                    comparison
                    .groupby("OddsBand")
                    .agg(
                        AverageProfit=(grid_profit_col, "mean"),
                        Strategies=(grid_profit_col, "count")
                    )
                    .reset_index()
                    .sort_values("AverageProfit", ascending=False)
                )

                fig_odds_band = px.bar(
                    odds_band_summary,
                    x="OddsBand",
                    y="AverageProfit",
                    title="Average Profit by Odds Band"
                )

                st.plotly_chart(fig_odds_band, use_container_width=True)

# =========================
# Tab 7: Strategy Simulator
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
# Tab 8: Bankroll Simulator
# =========================

with tab_bankroll:
    st.header("💰 Bankroll Simulator")

    st.write(
        "Simulate how the validated strategy performs under different staking plans. "
        "This helps compare flat staking versus percentage bankroll staking."
    )

    bankroll_col1, bankroll_col2, bankroll_col3 = st.columns(3)

    with bankroll_col1:
        starting_bankroll = st.number_input(
            "Starting Bankroll",
            min_value=10.0,
            max_value=100000.0,
            value=100.0,
            step=10.0
        )

    with bankroll_col2:
        flat_stake = st.number_input(
            "Flat Stake per Bet",
            min_value=0.1,
            max_value=1000.0,
            value=1.0,
            step=0.1
        )

    with bankroll_col3:
        selected_pct_stake = st.slider(
            "Percentage Stake",
            min_value=0.01,
            max_value=0.10,
            value=0.02,
            step=0.01,
            format="%.0%%"
        )

    st.divider()

    bankroll_bets = filtered_bets.copy()

    if len(bankroll_bets) == 0:
        st.warning("No bets match the current sidebar filters.")

    else:
        if date_col:
            bankroll_bets = bankroll_bets.sort_values(date_col).reset_index(drop=True)
        else:
            bankroll_bets = bankroll_bets.reset_index(drop=True)

        bankroll_bets["WonBool"] = make_bool_series(bankroll_bets[result_col])

        flat_rows = []
        flat_bankroll = starting_bankroll
        flat_peak = starting_bankroll

        for i, row in bankroll_bets.iterrows():
            odds = row[odds_col]
            won = row["WonBool"]

            stake = min(flat_stake, flat_bankroll)

            if stake <= 0:
                profit = 0
            elif won:
                profit = stake * (odds - 1)
            else:
                profit = -stake

            flat_bankroll += profit
            flat_peak = max(flat_peak, flat_bankroll)
            drawdown = flat_peak - flat_bankroll

            flat_rows.append({
                "BetNumber": i + 1,
                "Stake": stake,
                "Profit": profit,
                "Bankroll": flat_bankroll,
                "Drawdown": drawdown,
                "StakingPlan": "Flat Stake"
            })

        flat_sim = pd.DataFrame(flat_rows)

        pct_plan_rows = []

        for pct in [0.01, 0.02, 0.03, 0.05, selected_pct_stake]:
            bankroll = starting_bankroll
            peak = starting_bankroll

            for i, row in bankroll_bets.iterrows():
                odds = row[odds_col]
                won = row["WonBool"]

                stake = bankroll * pct

                if stake <= 0:
                    profit = 0
                elif won:
                    profit = stake * (odds - 1)
                else:
                    profit = -stake

                bankroll += profit
                peak = max(peak, bankroll)
                drawdown = peak - bankroll

                pct_plan_rows.append({
                    "BetNumber": i + 1,
                    "Stake": stake,
                    "Profit": profit,
                    "Bankroll": bankroll,
                    "Drawdown": drawdown,
                    "StakingPlan": f"{pct:.0%} Bankroll Stake"
                })

        pct_plans_sim = pd.DataFrame(pct_plan_rows)
        combined_sim = pd.concat([flat_sim, pct_plans_sim], ignore_index=True)

        staking_summary = (
            combined_sim
            .groupby("StakingPlan")
            .agg(
                FinalBankroll=("Bankroll", "last"),
                MaxDrawdown=("Drawdown", "max"),
                AverageStake=("Stake", "mean"),
                MaxStake=("Stake", "max")
            )
            .reset_index()
        )

        staking_summary["Profit"] = staking_summary["FinalBankroll"] - starting_bankroll
        staking_summary["ROI"] = staking_summary["Profit"] / starting_bankroll

        staking_summary = staking_summary[
            [
                "StakingPlan",
                "FinalBankroll",
                "Profit",
                "ROI",
                "MaxDrawdown",
                "AverageStake",
                "MaxStake"
            ]
        ]

        selected_plan_name = f"{selected_pct_stake:.0%} Bankroll Stake"

        flat_summary = staking_summary[staking_summary["StakingPlan"] == "Flat Stake"].iloc[0]
        pct_summary = staking_summary[staking_summary["StakingPlan"] == selected_plan_name].iloc[0]

        st.subheader("Selected Staking Plan Comparison")

        bm1, bm2, bm3, bm4 = st.columns(4)

        with bm1:
            st.metric("Flat Final Bankroll", f"{flat_summary['FinalBankroll']:.2f}")

        with bm2:
            st.metric("Flat Profit", f"{flat_summary['Profit']:+.2f}")

        with bm3:
            st.metric("Flat ROI", f"{flat_summary['ROI']:.2%}")

        with bm4:
            st.metric("Flat Max Drawdown", f"{flat_summary['MaxDrawdown']:.2f}")

        bm5, bm6, bm7, bm8 = st.columns(4)

        with bm5:
            st.metric("Percent Final Bankroll", f"{pct_summary['FinalBankroll']:.2f}")

        with bm6:
            st.metric("Percent Profit", f"{pct_summary['Profit']:+.2f}")

        with bm7:
            st.metric("Percent ROI", f"{pct_summary['ROI']:.2%}")

        with bm8:
            st.metric("Percent Max Drawdown", f"{pct_summary['MaxDrawdown']:.2f}")

        st.divider()

        st.subheader("Bankroll Growth by Staking Plan")

        fig_bankroll = px.line(
            combined_sim,
            x="BetNumber",
            y="Bankroll",
            color="StakingPlan",
            title="Bankroll Growth Comparison"
        )

        st.plotly_chart(fig_bankroll, use_container_width=True)

        st.subheader("Drawdown by Staking Plan")

        fig_bankroll_drawdown = px.line(
            combined_sim,
            x="BetNumber",
            y="Drawdown",
            color="StakingPlan",
            title="Drawdown Comparison"
        )

        st.plotly_chart(fig_bankroll_drawdown, use_container_width=True)

        st.subheader("Staking Plan Summary")

        st.dataframe(
            staking_summary,
            use_container_width=True
        )

        st.info(
            "Higher percentage staking can grow faster, but it also increases volatility and drawdown. "
            "Flat staking is usually safer for testing whether the betting edge is real."
        )

# =========================
# Tab 9: Bet Explorer
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
# Tab 10: Risk / Drawdown
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

    if grid_results is not None:
        st.write("Grid search columns:")
        st.write(list(grid_results.columns))

    if feature_store is not None:
        st.write("Feature store columns:")
        st.write(list(feature_store.columns))