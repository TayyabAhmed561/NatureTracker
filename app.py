"""
Stage 6: Dash dashboard for wildlife feeding-station analytics.

Run from the repository root:
``python app.py``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, Input, Output, callback, dash_table, dcc, html

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.analyze import (
    build_color_map,
    figure_activity_timeline,
    figure_cooccurrence,
    figure_food_preferences,
    figure_session_duration_box,
    figure_species_frequency,
    figure_temporal_heatmap,
    load_classifications,
    load_sessions,
    normalize_classifications_dataframe,
    prep_classifications,
)
from pipeline.build_sessions import parse_timestamp_to_seconds, seconds_to_hhmmss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

ACCENT = "#2d6a4f"
ACCENT_LIGHT = "#40916c"
HEADER_BG = "#1b4332"
ALL_SPECIES_VALUE = "__ALL__"

TABLE_COLUMNS = [
    "arrival_time",
    "departure_time",
    "duration_formatted",
    "species",
    "preferred_food",
    "concurrent_species",
]

FEED_OPTIMIZATION_INSIGHTS: tuple[str, ...] = (
    "Sunflower seeds attract the highest interaction across all species.",
    "Peanuts are the #1 preference for chipmunks and squirrels.",
    "Hour 5 is peak activity — ensure the station is fully stocked by then.",
    "A domestic cat was detected 8 times — consider station elevation to protect smaller visitors.",
    "Eastern Chipmunk + Chickadee co-occur 300 times — the station successfully supports multi-species simultaneous feeding.",
)


def _video_clock_range(sess: pd.DataFrame) -> str:
    """
    Format min arrival to max departure as ``HH:MM:SS – HH:MM:SS``.

    Args:
        sess: Sessions dataframe.

    Returns:
        Range string or em dash if empty.
    """
    if sess.empty:
        return "—"
    arr = sess["arrival_time"].astype(str).map(parse_timestamp_to_seconds)
    dep = sess["departure_time"].astype(str).map(parse_timestamp_to_seconds)
    lo = int(arr.min())
    hi = int(dep.max())
    return f"{seconds_to_hhmmss(lo)} – {seconds_to_hhmmss(hi)}"


def _filter_by_species(
    species_value: str | None,
    sessions_df: pd.DataFrame,
    class_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return subsets for the selected species, or full data for ``ALL``.

    Args:
        species_value: Dropdown value or ``ALL_SPECIES_VALUE``.
        sessions_df: Full sessions.
        class_df: Prepared classifications.

    Returns:
        ``(filtered_sessions, filtered_classifications)``.
    """
    if not species_value or species_value == ALL_SPECIES_VALUE:
        return sessions_df.copy(), class_df.copy()
    s = sessions_df[sessions_df["species"].astype(str) == str(species_value)]
    c = class_df[class_df["species"].astype(str) == str(species_value)]
    return s, c


def _load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[dict]]:
    """
    Load CSVs, normalize classifications, build color map and dropdown options.

    Returns:
        ``(sessions, class_prep, colors, dropdown_options)``.
    """
    sessions_df = load_sessions()
    raw_cls = load_classifications()
    raw_cls = normalize_classifications_dataframe(raw_cls)
    class_prep = prep_classifications(raw_cls)

    species_keys: list[str] = []
    if not sessions_df.empty:
        species_keys.extend(sessions_df["species"].astype(str).tolist())
    if not class_prep.empty:
        species_keys.extend(class_prep["species"].astype(str).tolist())
    colors = build_color_map(species_keys)

    options: list[dict] = [{"label": "All species", "value": ALL_SPECIES_VALUE}]
    if not sessions_df.empty:
        for sp in sorted(sessions_df["species"].dropna().astype(str).unique(), key=str.lower):
            options.append({"label": sp, "value": sp})

    if sessions_df.empty:
        logger.warning("sessions.csv missing or empty — dashboard will show placeholders.")
    if class_prep.empty:
        logger.warning("classifications.csv missing or empty — some charts will be empty.")

    return sessions_df, class_prep, colors, options


SESSIONS_FULL, CLASS_PREP_FULL, COLOR_MAP, SPECIES_OPTIONS = _load_dashboard_data()

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
)
app.title = "Wildlife Feeding Station Analytics"

CARD_STYLE = {
    "borderLeft": f"4px solid {ACCENT}",
    "borderRadius": "8px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
    "height": "100%",
    "backgroundColor": "#fff",
}

GRAPH_STYLE = {"height": "420px"}

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "🦫 Wildlife Feeding Station Analytics",
                            className="mb-2",
                            style={"color": "#fff", "fontWeight": 600, "fontSize": "1.75rem"},
                        ),
                        html.Div(id="header-subtitle", style={"color": "#d8f3dc", "fontSize": "0.95rem"}),
                    ],
                    width=12,
                ),
            ],
            className="mb-3 py-3 px-4",
            style={
                "backgroundColor": HEADER_BG,
                "borderRadius": "10px",
                "borderBottom": f"4px solid {ACCENT}",
            },
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Filter by species", className="fw-semibold small text-muted"),
                        dcc.Dropdown(
                            id="species-filter",
                            options=SPECIES_OPTIONS,
                            value=ALL_SPECIES_VALUE,
                            clearable=False,
                            className="mt-1",
                        ),
                    ],
                    md=6,
                    lg=4,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P("Total animal detections", className="text-muted small mb-1"),
                                html.H3(id="kpi-detections", className="mb-0", style={"color": ACCENT}),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P("Unique species identified", className="text-muted small mb-1"),
                                html.H3(id="kpi-species", className="mb-0", style={"color": ACCENT}),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P("Most frequent visitor", className="text-muted small mb-1"),
                                html.H4(id="kpi-visitor", className="mb-0 fs-5", style={"color": ACCENT}),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P("Peak activity hour", className="text-muted small mb-1"),
                                html.H3(id="kpi-peak-hour", className="mb-0", style={"color": ACCENT}),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    md=3,
                    className="mb-3",
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="graph-frequency", style=GRAPH_STYLE), md=6, className="mb-3"),
                dbc.Col(dcc.Graph(id="graph-heatmap", style=GRAPH_STYLE), md=6, className="mb-3"),
            ],
        ),
        dbc.Row(
            [dbc.Col(dcc.Graph(id="graph-timeline", style={"height": "480px"}), width=12, className="mb-3")],
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="graph-food", style=GRAPH_STYLE), md=6, className="mb-3"),
                dbc.Col(dcc.Graph(id="graph-cooc", style=GRAPH_STYLE), md=6, className="mb-3"),
            ],
        ),
        dbc.Row(
            [dbc.Col(dcc.Graph(id="graph-duration", style={"height": "440px"}), width=12, className="mb-3")],
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Feed optimization recommendations",
                                    className="mb-3 fw-semibold",
                                    style={"color": HEADER_BG},
                                ),
                                html.Ul(
                                    [html.Li(t, className="mb-2") for t in FEED_OPTIMIZATION_INSIGHTS],
                                    className="mb-0 ps-3",
                                    style={"fontSize": "0.95rem", "lineHeight": 1.5},
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, "borderLeft": f"4px solid {ACCENT}"},
                    ),
                    width=12,
                    className="mb-4",
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("All sessions", className="mb-2 fw-semibold", style={"color": HEADER_BG}),
                        dash_table.DataTable(
                            id="sessions-table",
                            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in TABLE_COLUMNS],
                            data=[],
                            filter_action="native",
                            sort_action="native",
                            page_size=12,
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
                            style_header={
                                "backgroundColor": ACCENT,
                                "color": "#fff",
                                "fontWeight": "600",
                            },
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
                            ],
                        ),
                    ],
                    width=12,
                    className="mb-4",
                ),
            ],
        ),
    ],
    fluid=True,
    className="py-4",
    style={"backgroundColor": "#ffffff", "minHeight": "100vh"},
)


@callback(
    Output("header-subtitle", "children"),
    Output("kpi-detections", "children"),
    Output("kpi-species", "children"),
    Output("kpi-visitor", "children"),
    Output("kpi-peak-hour", "children"),
    Output("graph-frequency", "figure"),
    Output("graph-heatmap", "figure"),
    Output("graph-timeline", "figure"),
    Output("graph-food", "figure"),
    Output("graph-cooc", "figure"),
    Output("graph-duration", "figure"),
    Output("sessions-table", "data"),
    Input("species-filter", "value"),
)
def update_dashboard(species_value: str | None):
    """
    Recompute KPIs, figures, and table rows for the selected species filter.

    Args:
        species_value: Dropdown value.

    Returns:
        Tuple of outputs for the callback.
    """
    sess, cls_df = _filter_by_species(species_value, SESSIONS_FULL, CLASS_PREP_FULL)

    n_sess = len(sess)
    n_species_sess = int(sess["species"].nunique()) if not sess.empty else 0
    subtitle = (
        f"{n_sess} sessions · {n_species_sess} species in view · "
        f"Video span: {_video_clock_range(sess)}"
    )

    total_det = int(cls_df["count"].sum()) if not cls_df.empty else 0
    n_species_cls = int(cls_df["species"].nunique()) if not cls_df.empty else 0

    if not sess.empty:
        vc = sess.groupby("common_name").size().sort_values(ascending=False)
        visitor = str(vc.index[0])
        visitor_n = int(vc.iloc[0])
        visitor_str = f"{visitor} ({visitor_n} sessions)"
    else:
        visitor_str = "—"

    if not sess.empty:
        by_h = sess.groupby("hour_of_video").size()
        peak_h = int(by_h.idxmax())
        peak_str = f"Hour {peak_h} ({int(by_h.max())} sessions)"
    else:
        peak_str = "—"

    fig_freq = figure_species_frequency(sess, COLOR_MAP)
    fig_heat = figure_temporal_heatmap(sess, COLOR_MAP)
    fig_time = figure_activity_timeline(
        cls_df,
        primary_line_color=ACCENT,
        secondary_line_color=ACCENT_LIGHT,
    )
    fig_food = figure_food_preferences(cls_df, COLOR_MAP)
    fig_cooc = figure_cooccurrence(sess)
    fig_box = figure_session_duration_box(sess, COLOR_MAP)

    if sess.empty:
        table_data = []
    else:
        table_data = sess[TABLE_COLUMNS].to_dict("records")

    return (
        subtitle,
        f"{total_det:,}",
        str(n_species_cls),
        visitor_str,
        peak_str,
        fig_freq,
        fig_heat,
        fig_time,
        fig_food,
        fig_cooc,
        fig_box,
        table_data,
    )


if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )
