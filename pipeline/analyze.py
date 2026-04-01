"""
Stage 5: Plotly charts and a short written insights summary.

Reads ``sessions.csv`` and ``classifications.csv``, writes paired HTML/PNG
charts under ``CHARTS_DIR``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ANOMALY_THRESHOLD_SECONDS,
    CHARTS_DIR,
    CLASSIFICATIONS_CSV,
    MAX_VIDEO_MINUTES,
    ROLLING_WINDOW_MINUTES,
    SESSIONS_CSV,
)

from pipeline.build_sessions import (
    SKIP_SPECIES,
    format_duration_human,
    normalize_classification_rows,
    parse_timestamp_to_seconds,
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger(__name__)

# Canonical palette for known visitors; extended at runtime for new species.
SPECIES_COLORS: dict[str, str] = {
    "Eastern Chipmunk": "#2E86AB",
    "American Red Squirrel": "#A23B72",
    "Tufted Titmouse": "#F4A261",
    "Black-capped Chickadee": "#E76F51",
    "Northern Cardinal": "#C1121F",
    "Blue Jay": "#3A86FF",
    "Dark-eyed Junco": "#8338EC",
    "House Sparrow": "#8AC926",
    "American Goldfinch": "#FFBE0B",
    "Downy Woodpecker": "#FB5607",
    "White-breasted Nuthatch": "#264653",
    "Carolina Wren": "#2A9D8F",
}

EXTENDED_PALETTE: list[str] = [
    "#2E86AB",
    "#A23B72",
    "#F18F01",
    "#C73E1D",
    "#3B1F2B",
    "#44AF69",
    "#FCAB10",
    "#2C699A",
    "#048BA8",
    "#0DB39E",
    "#16DB93",
    "#B0E0E6",
    "#D4A373",
    "#6D597A",
    "#355070",
    "#6C757D",
]


def build_color_map(species_keys: list[str]) -> dict[str, str]:
    """
    Map each species string to a stable hex color.

    Args:
        species_keys: Distinct species labels appearing in the charts.

    Returns:
        Mapping species -> ``#RRGGBB``.
    """
    out = dict(SPECIES_COLORS)
    i = 0
    for s in sorted(set(species_keys), key=str.lower):
        if s not in out:
            out[s] = EXTENDED_PALETTE[i % len(EXTENDED_PALETTE)]
            i += 1
    return out


def ensure_charts_dir() -> None:
    """Create ``CHARTS_DIR`` if missing."""
    try:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create charts directory %s", CHARTS_DIR)
        raise


def apply_base_layout(fig: go.Figure, title: str) -> None:
    """
    Apply shared typography and margins.

    Args:
        fig: Plotly figure.
        title: Chart title text.
    """
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        template="plotly_white",
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=12),
        margin=dict(l=72, r=48, t=72, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def save_chart(fig: go.Figure, stem: str) -> None:
    """
    Write HTML and PNG for a figure.

    Args:
        fig: Figure to export.
        stem: Base filename without extension.
    """
    html_path = CHARTS_DIR / f"{stem}.html"
    png_path = CHARTS_DIR / f"{stem}.png"
    try:
        fig.write_html(str(html_path), include_plotlyjs="cdn")
    except OSError:
        logger.exception("Failed to write %s", html_path)
        raise
    try:
        fig.write_image(str(png_path), width=1200, height=700, scale=2)
    except Exception as exc:
        logger.warning("PNG export failed for %s (install kaleido?): %s", stem, exc)


def load_sessions() -> pd.DataFrame:
    """
    Load ``sessions.csv`` into a DataFrame.

    Returns:
        Parsed sessions or empty frame if missing.
    """
    if not SESSIONS_CSV.is_file():
        logger.error("Missing sessions file: %s", SESSIONS_CSV)
        return pd.DataFrame()
    try:
        return pd.read_csv(SESSIONS_CSV)
    except OSError:
        logger.exception("Failed to read sessions CSV")
        raise


def load_classifications() -> pd.DataFrame:
    """
    Load ``classifications.csv`` into a DataFrame.

    Returns:
        Parsed classifications or empty frame if missing.
    """
    if not CLASSIFICATIONS_CSV.is_file():
        logger.error("Missing classifications file: %s", CLASSIFICATIONS_CSV)
        return pd.DataFrame()
    try:
        return pd.read_csv(CLASSIFICATIONS_CSV)
    except OSError:
        logger.exception("Failed to read classifications CSV")
        raise


def normalize_classifications_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ``SPECIES_NORMALIZATION`` from config (same rules as Stage 4).

    Mutates row-wise equivalents so charts and timelines match merged species
    labels (e.g. scientific names → common labels).

    Args:
        df: Raw classifications from CSV.

    Returns:
        DataFrame with updated ``species`` and ``common_name`` columns.
    """
    if df.empty:
        return df
    records = df.to_dict("records")
    normalize_classification_rows(records)
    return pd.DataFrame.from_records(records)


def prep_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative time columns and numeric counts; drop sentinel species.

    Args:
        df: Raw classifications.

    Returns:
        Filtered copy with ``t_sec``, ``t_rel``, ``minute``, ``count``.
    """
    if df.empty:
        return df
    out = df.copy()
    out["species"] = out["species"].astype(str).str.strip()
    out = out[~out["species"].isin(SKIP_SPECIES)]
    out["t_sec"] = out["timestamp"].astype(str).map(parse_timestamp_to_seconds)
    t0 = int(out["t_sec"].min())
    out["t_rel"] = out["t_sec"] - t0
    out["minute"] = out["t_rel"] / 60.0
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0).astype(int)
    return out


def empty_figure_message(title: str, message: str = "No data") -> go.Figure:
    """
    Return a minimal figure with a centered title and annotation.

    Args:
        title: Chart title.
        message: Body text.

    Returns:
        Placeholder ``go.Figure``.
    """
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_white",
        margin=dict(l=48, r=48, t=56, b=48),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#6c757d"),
            )
        ],
    )
    return fig


def figure_species_frequency(sessions: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """
    Build the species frequency horizontal bar chart (no file I/O).

    Args:
        sessions: Sessions table.
        colors: Species -> hex color map.

    Returns:
        Plotly figure.
    """
    if sessions.empty:
        return empty_figure_message("Most Frequent Visitors", "No session data")
    g = sessions.groupby("common_name", as_index=False).size()
    g = g.rename(columns={"size": "sessions"})
    g = g.sort_values("sessions", ascending=True)
    cn_to_sp = (
        sessions.drop_duplicates(subset=["common_name"])
        .set_index("common_name")["species"]
        .astype(str)
        .to_dict()
    )
    bar_colors = [colors.get(str(cn_to_sp.get(cn, "")), "#888") for cn in g["common_name"]]
    fig = go.Figure(
        go.Bar(
            x=g["sessions"],
            y=g["common_name"],
            orientation="h",
            marker=dict(color=bar_colors),
            text=g["sessions"],
            textposition="outside",
        )
    )
    apply_base_layout(fig, "Most Frequent Visitors")
    fig.update_xaxes(title="Number of sessions", showgrid=True, gridcolor="#ECECEC")
    fig.update_yaxes(title="", showgrid=False, categoryorder="total ascending")
    return fig


def chart_species_frequency(sessions: pd.DataFrame, colors: dict[str, str]) -> None:
    """
    CHART 1 — horizontal bar of session counts by common name.

    Args:
        sessions: Sessions table.
        colors: Species -> hex color map (keyed by ``species``).
    """
    if sessions.empty:
        logger.warning("Skipping species frequency chart (no sessions).")
        return
    fig = figure_species_frequency(sessions, colors)
    save_chart(fig, "species_frequency")


def figure_temporal_heatmap(sessions: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """
    Build the temporal activity heatmap (no file I/O).

    Args:
        sessions: Sessions table.
        colors: Unused (heatmap uses ``YlOrRd``).

    Returns:
        Plotly figure.
    """
    del colors
    if sessions.empty:
        return empty_figure_message("Activity by Hour", "No session data")
    hours = list(range(8))
    species_list = sorted(sessions["species"].unique(), key=str.lower)
    z = []
    for sp in species_list:
        row = []
        sub = sessions[sessions["species"] == sp]
        for h in hours:
            row.append(int((sub["hour_of_video"] == h).sum()))
        z.append(row)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[str(h) for h in hours],
            y=species_list,
            colorscale="YlOrRd",
            colorbar=dict(title="Sessions"),
        )
    )
    apply_base_layout(fig, "Activity by Hour")
    fig.update_xaxes(title="Hour of video (0–7)", showgrid=False)
    fig.update_yaxes(title="Species", showgrid=False)
    return fig


def chart_temporal_heatmap(sessions: pd.DataFrame, colors: dict[str, str]) -> None:
    """
    CHART 2 — sessions per hour × species heatmap.

    Args:
        sessions: Sessions table.
        colors: Unused for heatmap fill; warm colorscale per spec.
    """
    if sessions.empty:
        logger.warning("Skipping temporal heatmap (no sessions).")
        return
    fig = figure_temporal_heatmap(sessions, colors)
    save_chart(fig, "temporal_activity_heatmap")


def window_species_label(
    df: pd.DataFrame,
    lo: float,
    hi: float,
    *,
    max_species: int = 3,
) -> str:
    """
    Summarize dominant species in a time window for annotations.

    Args:
        df: Prepared classifications with ``t_rel`` and ``count``.
        lo: Lower bound on ``t_rel`` (exclusive).
        hi: Upper bound on ``t_rel`` (inclusive).
        max_species: Maximum species names to include (comma-separated).

    Returns:
        Short comma-separated label.
    """
    win = df[(df["t_rel"] > lo) & (df["t_rel"] <= hi)]
    if win.empty:
        return ""
    agg = win.groupby("species")["count"].sum().sort_values(ascending=False).head(max_species)
    return ", ".join(f"{n} ({int(v)})" for n, v in agg.items())


def figure_activity_timeline(
    df: pd.DataFrame,
    *,
    primary_line_color: str = "#C1121F",
    secondary_line_color: str = "#2E86AB",
) -> go.Figure:
    """
    Build the rolling activity timeline (no file I/O).

    Args:
        df: Prepared classifications.
        primary_line_color: Color for the rolling-sum trace.
        secondary_line_color: Color for the unique-species trace.

    Returns:
        Plotly figure.
    """
    if df.empty:
        return empty_figure_message(
            "Feeding Station Activity Over 8 Hours",
            "No classification data",
        )
    win_sec = ROLLING_WINDOW_MINUTES * 60
    minutes = np.arange(0, MAX_VIDEO_MINUTES + 1, dtype=float)
    activity: list[float] = []
    uniq_sp: list[int] = []
    for m in minutes:
        hi = m * 60.0
        lo = max(0.0, hi - win_sec)
        win = df[(df["t_rel"] > lo) & (df["t_rel"] <= hi)]
        activity.append(float(win["count"].sum()))
        uniq_sp.append(int(win["species"].nunique(dropna=True)))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=activity,
            mode="lines",
            name="Animals (15m rolling sum)",
            line=dict(color=primary_line_color, width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=uniq_sp,
            mode="lines",
            name="Unique species visible",
            line=dict(color=secondary_line_color, width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    act_arr = np.array(activity)
    peak_idx = np.argsort(act_arr)[-3:][::-1]
    y_pad_px = 8
    for rank, idx in enumerate(peak_idx, start=1):
        m = float(minutes[idx])
        lo = max(0.0, m * 60.0 - win_sec)
        hi = m * 60.0
        label = window_species_label(df, lo, hi, max_species=2)
        if not label:
            label = "—"
        y_peak = float(act_arr[idx])
        ay_px = -(36 + rank * y_pad_px)
        fig.add_annotation(
            x=m,
            y=y_peak,
            xref="x",
            yref="y",
            text=f"Peak {rank}: {label}",
            showarrow=True,
            arrowhead=2,
            arrowwidth=1,
            ax=0,
            ay=ay_px,
            xanchor="center",
            yanchor="bottom",
            align="center",
            font=dict(size=10),
        )

    apply_base_layout(fig, "Feeding Station Activity Over 8 Hours")
    fig.update_xaxes(title="Minutes from start", showgrid=True, gridcolor="#ECECEC", range=[0, MAX_VIDEO_MINUTES])
    fig.update_yaxes(title_text="Rolling sum of counts", secondary_y=False, showgrid=True, gridcolor="#ECECEC")
    fig.update_yaxes(title_text="Distinct species", secondary_y=True, showgrid=False)
    return fig


def chart_activity_timeline(df: pd.DataFrame) -> None:
    """
    CHART 3 — rolling 15-minute activity and unique species (dual axis).

    Args:
        df: Prepared classifications.
    """
    if df.empty:
        logger.warning("Skipping activity timeline (no classifications).")
        return
    fig = figure_activity_timeline(df)
    save_chart(fig, "activity_timeline")


def figure_food_preferences(df: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """
    Build the food preference grouped bar chart (no file I/O).

    Args:
        df: Prepared classifications.
        colors: Species color map.

    Returns:
        Plotly figure.
    """
    if df.empty:
        return empty_figure_message("Food Preferences by Species", "No classification data")
    sub = df[(df["food_item"].notna()) & (df["food_item"].astype(str).str.strip() != "")]
    if sub.empty:
        return empty_figure_message("Food Preferences by Species", "No food_item values")
    sub = sub.copy()
    sub["food_item"] = sub["food_item"].astype(str).str.strip()
    totals = sub.groupby("food_item")["count"].sum().sort_values(ascending=False)
    top_foods = list(totals.head(5).index)
    species_list = sorted(sub["species"].unique(), key=str.lower)

    fig = go.Figure()
    for fi, food in enumerate(top_foods):
        ys = [
            int(sub[(sub["species"] == sp) & (sub["food_item"] == food)]["count"].sum())
            for sp in species_list
        ]
        bar_colors = [colors.get(sp, EXTENDED_PALETTE[fi % len(EXTENDED_PALETTE)]) for sp in species_list]
        fig.add_trace(
            go.Bar(
                name=food,
                x=species_list,
                y=ys,
                marker=dict(color=bar_colors, line=dict(width=0)),
                opacity=0.92,
            )
        )

    apply_base_layout(fig, "Food Preferences by Species")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Species", showgrid=False)
    fig.update_yaxes(title="Total count (weighted)", showgrid=True, gridcolor="#ECECEC")
    return fig


def chart_food_preferences(df: pd.DataFrame, colors: dict[str, str]) -> None:
    """
    CHART 4 — grouped bars of top-5 foods by species.

    Args:
        df: Prepared classifications.
        colors: Species color map.
    """
    if df.empty:
        logger.warning("Skipping food preference chart (no data).")
        return
    sub = df[(df["food_item"].notna()) & (df["food_item"].astype(str).str.strip() != "")]
    if sub.empty:
        logger.warning("Skipping food preference chart (no food_item values).")
        return
    fig = figure_food_preferences(df, colors)
    save_chart(fig, "food_preferences")


def build_cooccurrence_matrix(sessions: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Build a symmetric co-occurrence count matrix from ``concurrent_species``.

    Args:
        sessions: Sessions table.

    Returns:
        Ordered species labels and integer matrix ``counts[i][j]``.
    """
    species_list = sorted(sessions["species"].astype(str).unique(), key=str.lower)
    idx = {s: i for i, s in enumerate(species_list)}
    n = len(species_list)
    mat = np.zeros((n, n), dtype=int)
    for _, row in sessions.iterrows():
        s = str(row.get("species", "")).strip()
        if s not in idx:
            continue
        raw = str(row.get("concurrent_species", "") or "")
        others = [x.strip() for x in raw.split(",") if x.strip()]
        for o in others:
            if o not in idx:
                continue
            i, j = idx[s], idx[o]
            mat[i, j] += 1
            mat[j, i] += 1
    return species_list, mat


def figure_cooccurrence(sessions: pd.DataFrame) -> go.Figure:
    """
    Build the co-occurrence heatmap (no file I/O).

    Args:
        sessions: Sessions table.

    Returns:
        Plotly figure.
    """
    if sessions.empty:
        return empty_figure_message("Species Co-occurrence", "No session data")
    labels, mat = build_cooccurrence_matrix(sessions)
    if not labels:
        return empty_figure_message("Species Co-occurrence", "No co-occurrence data")
    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            colorscale="Blues",
            colorbar=dict(title="Shared sessions"),
        )
    )
    apply_base_layout(fig, "Species Co-occurrence")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def chart_cooccurrence(sessions: pd.DataFrame) -> None:
    """
    CHART 5 — species co-occurrence heatmap.

    Args:
        sessions: Sessions table.
    """
    if sessions.empty:
        logger.warning("Skipping co-occurrence chart (no sessions).")
        return
    labels, _ = build_cooccurrence_matrix(sessions)
    if not labels:
        return
    fig = figure_cooccurrence(sessions)
    save_chart(fig, "cooccurrence")


def figure_session_duration_box(sessions: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """
    Build the session duration box plot (no file I/O).

    Args:
        sessions: Sessions table.
        colors: Species color map.

    Returns:
        Plotly figure.
    """
    if sessions.empty:
        return empty_figure_message("Visit Duration by Species", "No session data")
    counts = sessions.groupby("species").size()
    keep = counts[counts >= 3].index.tolist()
    if not keep:
        return empty_figure_message(
            "Visit Duration by Species",
            "No species with ≥3 sessions (try clearing the species filter)",
        )
    sub = sessions[sessions["species"].isin(keep)]
    fig = go.Figure()
    for sp in sorted(keep, key=str.lower):
        fig.add_trace(
            go.Box(
                y=sub.loc[sub["species"] == sp, "duration_seconds"],
                name=sp,
                marker_color=colors.get(sp, "#888"),
                boxmean=True,
            )
        )
    apply_base_layout(fig, "Visit Duration by Species")
    fig.update_xaxes(title="Species", showgrid=False)
    fig.update_yaxes(title="Duration (seconds)", showgrid=True, gridcolor="#ECECEC")
    return fig


def chart_session_duration_box(sessions: pd.DataFrame, colors: dict[str, str]) -> None:
    """
    CHART 6 — box plot of session duration for species with ≥3 sessions.

    Args:
        sessions: Sessions table.
        colors: Species color map.
    """
    if sessions.empty:
        logger.warning("Skipping duration box chart (no sessions).")
        return
    counts = sessions.groupby("species").size()
    keep = counts[counts >= 3].index.tolist()
    if not keep:
        logger.warning("No species with ≥3 sessions; skipping duration box chart.")
        return
    fig = figure_session_duration_box(sessions, colors)
    save_chart(fig, "session_duration_box")


def print_insights_summary(
    sessions: pd.DataFrame,
    class_df: pd.DataFrame,
    co_labels: list[str],
    co_mat: np.ndarray,
) -> None:
    """
    Print a concise narrative summary to the console and log.

    Args:
        sessions: Sessions table.
        class_df: Prepared classifications.
        co_labels: Co-occurrence axis labels.
        co_mat: Co-occurrence matrix.
    """
    lines: list[str] = ["--- Insights summary ---"]

    if not sessions.empty:
        vc = sessions.groupby("common_name").size().sort_values(ascending=False)
        top = vc.index[0]
        lines.append(f"Most visits (by session count): {top} ({int(vc.iloc[0])} sessions).")

        by_hour = sessions.groupby("hour_of_video").size()
        peak_h = int(by_hour.idxmax()) if not by_hour.empty else 0
        lines.append(
            f"Peak activity hour (session starts): hour {peak_h} "
            f"({int(by_hour.max())} sessions starting in that hour)."
        )

        avg_by_sp = sessions.groupby("species")["duration_seconds"].mean().sort_values(ascending=False)
        if not avg_by_sp.empty:
            sp = avg_by_sp.index[0]
            lines.append(
                f"Longest average visit duration: {sp} "
                f"({avg_by_sp.iloc[0]:.0f}s avg, ~{format_duration_human(int(round(avg_by_sp.iloc[0])))})."
            )

        longest = sessions.loc[sessions["duration_seconds"].idxmax()]
        lines.append(
            f"Single longest visit: {longest.get('species', '')} "
            f"({int(longest.get('duration_seconds', 0))}s / {longest.get('duration_formatted', '')})."
        )

    if not class_df.empty:
        subf = class_df[class_df["food_item"].astype(str).str.strip() != ""].copy()
        if not subf.empty:
            subf["_food"] = subf["food_item"].astype(str).str.strip()
            totals = subf.groupby("_food")["count"].sum().sort_values(ascending=False)
            top_food = totals.index[0]
            lines.append(
                f"Most popular food (weighted by count column): {top_food} ({int(totals.iloc[0])})."
            )

    best_pair = ("", 0)
    if co_labels and co_mat.size:
        n = len(co_labels)
        for i in range(n):
            for j in range(i + 1, n):
                v = int(co_mat[i, j])
                if v > best_pair[1]:
                    best_pair = (f"{co_labels[i]} + {co_labels[j]}", v)
        if best_pair[1] > 0:
            lines.append(f"Most common co-occurring pair: {best_pair[0]} ({best_pair[1]} overlaps).")

    # Session-shape heuristics (median duration is often 0 when visits are single-frame)
    if not sessions.empty:
        n_total = len(sessions)
        fc = pd.to_numeric(sessions["frame_count"], errors="coerce").fillna(0).astype(int)
        n_single = int((fc == 1).sum())
        pct_single = (100.0 * n_single / n_total) if n_total else 0.0
        lines.append(
            f"Single-frame sessions: {n_single} of {n_total} ({pct_single:.1f}% of all sessions)."
        )

        dur = pd.to_numeric(sessions["duration_seconds"], errors="coerce").fillna(0)
        thresh = float(ANOMALY_THRESHOLD_SECONDS)
        n_long = int((dur > thresh).sum())
        if n_long > 0:
            lines.append(
                f"Unusually long sessions (>{int(thresh)}s / {int(thresh // 60)}m): {n_long} — "
                "possible extended feeding, loitering, or sparse frame sampling."
            )

        night = sessions[sessions["hour_of_video"].isin([0, 1, 2])]
        if len(night) <= max(1, len(sessions) // 20):
            lines.append("Few sessions start in early-night hours (0–2); activity may be diurnal.")

    for line in lines:
        logger.info("%s", line)
    print()
    for line in lines:
        print(line)
    print()


def run_analyze() -> None:
    """
    Generate all charts and print the insights summary.
    """
    ensure_charts_dir()
    sessions = load_sessions()
    raw_cls = load_classifications()
    raw_cls = normalize_classifications_dataframe(raw_cls)
    class_df = prep_classifications(raw_cls)

    species_keys: list[str] = []
    if not sessions.empty:
        species_keys.extend(sessions["species"].astype(str).tolist())
    if not class_df.empty:
        species_keys.extend(class_df["species"].astype(str).tolist())
    colors = build_color_map(species_keys)

    chart_species_frequency(sessions, colors)
    chart_temporal_heatmap(sessions, colors)
    chart_activity_timeline(class_df)
    chart_food_preferences(class_df, colors)
    chart_cooccurrence(sessions)
    chart_session_duration_box(sessions, colors)

    co_labels, co_mat = (
        build_cooccurrence_matrix(sessions) if not sessions.empty else ([], np.zeros((0, 0)))
    )
    print_insights_summary(sessions, class_df, co_labels, co_mat)
    logger.info("Charts written to %s", CHARTS_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for Stage 5.

    Args:
        argv: Optional argv.

    Returns:
        Parsed namespace.
    """
    return argparse.ArgumentParser(description="Generate analytics charts from session data.").parse_args(
        argv
    )


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry for Stage 5.

    Args:
        argv: Optional argv.

    Returns:
        Exit code.
    """
    parse_args(argv)
    try:
        run_analyze()
    except Exception:
        logger.exception("Stage 5 failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
