import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


# =========================
# File paths
# =========================

MARKET_SUMMARY_FILE = Path("data/processed/corner_edge_summary_by_market_v1.csv")
LEAGUE_SUMMARY_FILE = Path("data/processed/corner_edge_summary_by_league_v1.csv")
MARKET_LEAGUE_SUMMARY_FILE = Path("data/processed/corner_edge_summary_by_market_league_v1.csv")
WATCHLIST_FILE = Path("data/processed/corner_edge_watchlist_v1.csv")


# =========================
# Helpers
# =========================

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def format_percent_column(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")

    return df


def format_decimal_column(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    return df


def display_download_button(df: pd.DataFrame, file_name: str, label: str):
    if len(df) == 0:
        return

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv,
        file_name=file_name,
        mime="text/csv",
    )


# =========================
# Main render function
# =========================

def render_corner_markets_tab():
    st.header("🟨 Corner Markets")

    st.write(
        """
        This tab analyzes corner betting markets using historical team corner patterns.
        It does **not** use actual corner odds yet. Instead, it shows model probability,
        fair odds, and the minimum sportsbook odds needed for a positive expected value bet.
        """
    )

    market_summary = load_csv(MARKET_SUMMARY_FILE)
    league_summary = load_csv(LEAGUE_SUMMARY_FILE)
    market_league_summary = load_csv(MARKET_LEAGUE_SUMMARY_FILE)
    watchlist = load_csv(WATCHLIST_FILE)

    missing_files = []

    for path in [
        MARKET_SUMMARY_FILE,
        LEAGUE_SUMMARY_FILE,
        MARKET_LEAGUE_SUMMARY_FILE,
        WATCHLIST_FILE,
    ]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        st.warning("Corner market files are missing. Run this first:")
        st.code("python scripts/analyze_corner_edges.py", language="powershell")
        st.write("Missing files:")
        for file in missing_files:
            st.write(f"- {file}")
        return

    if len(watchlist) > 0 and "Date" in watchlist.columns:
        watchlist["Date"] = pd.to_datetime(watchlist["Date"], errors="coerce")

    st.subheader("Key idea")

    st.info(
        """
        Fair odds = 1 / model probability.

        Example: if the model gives a corner market a 60% chance, fair odds are 1.67.
        With a 5% edge requirement, the sportsbook odds should be about 1.75 or higher.
        """
    )

    # =========================
    # KPI cards
    # =========================

    st.subheader("Corner Model Snapshot")

    total_markets = market_summary["Market"].nunique() if "Market" in market_summary.columns else 0
    total_leagues = league_summary["League"].nunique() if "League" in league_summary.columns else 0
    total_watchlist = len(watchlist)

    elite_count = 0
    if "ConfidenceTier" in watchlist.columns:
        elite_count = len(watchlist[watchlist["ConfidenceTier"] == "Elite"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Markets", total_markets)
    col2.metric("Leagues", total_leagues)
    col3.metric("Watchlist Picks", f"{total_watchlist:,}")
    col4.metric("Elite Picks", f"{elite_count:,}")

    # =========================
    # Best markets
    # =========================

    st.subheader("Best Corner Markets")

    if len(market_summary) > 0:
        market_display = market_summary.copy()

        market_percent_columns = [
            "HitRate",
            "AverageProbability",
            "HighConfidenceHitRate",
            "HighConfidenceAverageProbability",
            "StrongConfidenceHitRate",
            "StrongConfidenceAverageProbability",
            "EliteConfidenceHitRate",
            "EliteConfidenceAverageProbability",
        ]

        market_decimal_columns = [
            "AverageFairOdds",
            "HighConfidenceAverageFairOdds",
            "StrongConfidenceAverageFairOdds",
            "EliteConfidenceAverageFairOdds",
        ]

        market_display = format_percent_column(market_display, market_percent_columns)
        market_display = format_decimal_column(market_display, market_decimal_columns)

        st.dataframe(market_display, use_container_width=True)

        chart_df = market_summary.copy()

        if "HighConfidenceHitRate" in chart_df.columns:
            chart_df["HighConfidenceHitRate"] = pd.to_numeric(
                chart_df["HighConfidenceHitRate"],
                errors="coerce",
            )

            fig = px.bar(
                chart_df.sort_values("HighConfidenceHitRate", ascending=True),
                x="HighConfidenceHitRate",
                y="Market",
                orientation="h",
                title="High-Confidence Hit Rate by Corner Market",
            )

            st.plotly_chart(fig, use_container_width=True)

        display_download_button(
            market_summary,
            "corner_edge_summary_by_market_v1.csv",
            "Download market summary",
        )

    # =========================
    # Best leagues
    # =========================

    st.subheader("Best Leagues for Corner Markets")

    if len(league_summary) > 0:
        league_display = league_summary.copy()

        league_percent_columns = [
            "HitRate",
            "AverageProbability",
            "HighConfidenceHitRate",
            "HighConfidenceAverageProbability",
            "StrongConfidenceHitRate",
            "StrongConfidenceAverageProbability",
            "EliteConfidenceHitRate",
            "EliteConfidenceAverageProbability",
        ]

        league_decimal_columns = [
            "AverageFairOdds",
            "HighConfidenceAverageFairOdds",
            "StrongConfidenceAverageFairOdds",
            "EliteConfidenceAverageFairOdds",
        ]

        league_display = format_percent_column(league_display, league_percent_columns)
        league_display = format_decimal_column(league_display, league_decimal_columns)

        st.dataframe(league_display, use_container_width=True)

        chart_df = league_summary.copy()

        if "HighConfidenceHitRate" in chart_df.columns:
            chart_df["HighConfidenceHitRate"] = pd.to_numeric(
                chart_df["HighConfidenceHitRate"],
                errors="coerce",
            )

            fig = px.bar(
                chart_df.sort_values("HighConfidenceHitRate", ascending=True),
                x="HighConfidenceHitRate",
                y="League",
                orientation="h",
                title="High-Confidence Hit Rate by League",
            )

            st.plotly_chart(fig, use_container_width=True)

        display_download_button(
            league_summary,
            "corner_edge_summary_by_league_v1.csv",
            "Download league summary",
        )

    # =========================
    # Market + league combinations
    # =========================

    st.subheader("Best Market + League Combinations")

    if len(market_league_summary) > 0:
        min_picks = st.slider(
            "Minimum high-confidence picks",
            min_value=0,
            max_value=1000,
            value=100,
            step=25,
            key="corner_min_high_confidence_picks",
        )

        combo_filtered = market_league_summary.copy()

        if "HighConfidencePicks" in combo_filtered.columns:
            combo_filtered = combo_filtered[
                combo_filtered["HighConfidencePicks"] >= min_picks
            ].copy()

        combo_filtered = combo_filtered.sort_values(
            by=["HighConfidenceHitRate", "HighConfidencePicks"],
            ascending=[False, False],
        )

        combo_display = combo_filtered.copy()

        combo_percent_columns = [
            "HitRate",
            "AverageProbability",
            "HighConfidenceHitRate",
            "HighConfidenceAverageProbability",
            "StrongConfidenceHitRate",
            "StrongConfidenceAverageProbability",
            "EliteConfidenceHitRate",
            "EliteConfidenceAverageProbability",
        ]

        combo_decimal_columns = [
            "AverageFairOdds",
            "HighConfidenceAverageFairOdds",
            "StrongConfidenceAverageFairOdds",
            "EliteConfidenceAverageFairOdds",
        ]

        combo_display = format_percent_column(combo_display, combo_percent_columns)
        combo_display = format_decimal_column(combo_display, combo_decimal_columns)

        st.dataframe(combo_display.head(50), use_container_width=True)

        chart_df = combo_filtered.head(20).copy()

        if len(chart_df) > 0:
            chart_df["Combo"] = chart_df["Market"] + " | " + chart_df["League"]

            fig = px.bar(
                chart_df.sort_values("HighConfidenceHitRate", ascending=True),
                x="HighConfidenceHitRate",
                y="Combo",
                orientation="h",
                title="Top Market + League Corner Combinations",
            )

            st.plotly_chart(fig, use_container_width=True)

        display_download_button(
            market_league_summary,
            "corner_edge_summary_by_market_league_v1.csv",
            "Download market + league summary",
        )

    # =========================
    # Watchlist
    # =========================

    st.subheader("Corner Edge Watchlist")

    if len(watchlist) > 0:
        st.write(
            """
            The watchlist shows historical corner spots where the model probability was high.
            In live use, compare the sportsbook odds to the minimum bettable odds.
            """
        )

        available_leagues = sorted(watchlist["League"].dropna().unique()) if "League" in watchlist.columns else []
        available_markets = sorted(watchlist["Market"].dropna().unique()) if "Market" in watchlist.columns else []
        available_tiers = sorted(watchlist["ConfidenceTier"].dropna().unique()) if "ConfidenceTier" in watchlist.columns else []

        col1, col2, col3 = st.columns(3)

        selected_leagues = col1.multiselect(
            "Leagues",
            options=available_leagues,
            default=available_leagues,
            key="corner_watchlist_leagues",
        )

        selected_markets = col2.multiselect(
            "Markets",
            options=available_markets,
            default=available_markets,
            key="corner_watchlist_markets",
        )

        selected_tiers = col3.multiselect(
            "Confidence tiers",
            options=available_tiers,
            default=available_tiers,
            key="corner_watchlist_tiers",
        )

        min_probability = st.slider(
            "Minimum model probability",
            min_value=0.50,
            max_value=0.95,
            value=0.60,
            step=0.01,
            key="corner_watchlist_min_probability",
        )

        filtered_watchlist = watchlist.copy()

        if selected_leagues:
            filtered_watchlist = filtered_watchlist[
                filtered_watchlist["League"].isin(selected_leagues)
            ]

        if selected_markets:
            filtered_watchlist = filtered_watchlist[
                filtered_watchlist["Market"].isin(selected_markets)
            ]

        if selected_tiers:
            filtered_watchlist = filtered_watchlist[
                filtered_watchlist["ConfidenceTier"].isin(selected_tiers)
            ]

        if "Probability" in filtered_watchlist.columns:
            filtered_watchlist["Probability"] = pd.to_numeric(
                filtered_watchlist["Probability"],
                errors="coerce",
            )

            filtered_watchlist = filtered_watchlist[
                filtered_watchlist["Probability"] >= min_probability
            ]

        filtered_watchlist = filtered_watchlist.sort_values(
            by="Probability",
            ascending=False,
        )

        st.metric("Filtered Watchlist Picks", f"{len(filtered_watchlist):,}")

        display_df = filtered_watchlist.copy()

        display_df = format_percent_column(display_df, ["Probability"])
        display_df = format_decimal_column(
            display_df,
            [
                "FairOdds",
                "MinimumBettableOdds_5pctEdge",
            ],
        )

        st.dataframe(display_df.head(250), use_container_width=True)

        display_download_button(
            filtered_watchlist,
            "corner_edge_watchlist_filtered.csv",
            "Download filtered watchlist",
        )

    st.warning(
        """
        Corner results are not profit results yet because we do not have historical corner odds.
        Use fair odds and minimum bettable odds as decision thresholds until a live odds API is connected.
        """
    )