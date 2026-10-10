"""
projections.py
==============
Projection explorer for 4D and 5D random walks.

Since we cannot directly visualize 4D or 5D geometry, we show
2D projections (pairs of coordinates). The user selects which two
coordinates to display on the X and Y axes, and optionally a third
coordinate to encode as color.

These are always clearly labeled as PROJECTIONS, not direct visualizations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go


BG_COLOR    = "#0d1117"
GRID_COLOR  = "#1e2a38"
PAPER_COLOR = "#0d1117"

COORD_NAMES = ["x₁", "x₂", "x₃", "x₄", "x₅"]
COORD_LABELS = ["x1", "x2", "x3", "x4", "x5"]


def plot_projection(
    trajectory: NDArray[np.int32],
    x_axis: int = 0,
    y_axis: int = 1,
    color_axis: int | None = None,
    title: str | None = None,
    dimension: int | None = None,
    max_points: int = 3000,
) -> go.Figure:
    """
    Plot a 2D projection of a high-dimensional random walk trajectory.

    Parameters
    ----------
    trajectory : ndarray of shape (T, d)
    x_axis : int   — coordinate index for X axis (0-based)
    y_axis : int   — coordinate index for Y axis (0-based)
    color_axis : int or None — coordinate index to encode as color
    title : str or None
    dimension : int or None  — for labeling purposes
    max_points : int  — downsample for performance

    Returns
    -------
    fig : plotly Figure
    """
    d   = trajectory.shape[1]
    dim = dimension or d

    n = len(trajectory)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        traj_plot = trajectory[idx]
    else:
        traj_plot = trajectory

    x = traj_plot[:, x_axis].astype(np.float64)
    y = traj_plot[:, y_axis].astype(np.float64)

    # Color values
    if color_axis is not None and color_axis < d and color_axis != x_axis and color_axis != y_axis:
        c = traj_plot[:, color_axis].astype(np.float64)
        color_label = f"← {COORD_NAMES[color_axis]} →"
        marker_kwargs = dict(
            color=c,
            colorscale="Plasma",
            size=3,
            showscale=True,
            colorbar=dict(title=color_label, thickness=12),
        )
    else:
        t_color = np.linspace(0, 1, len(traj_plot))
        marker_kwargs = dict(
            color=t_color,
            colorscale="Blues",
            size=3,
            showscale=True,
            colorbar=dict(title="Time →", thickness=12),
        )

    x_label = COORD_NAMES[x_axis]
    y_label = COORD_NAMES[y_axis]

    auto_title = (
        f"2D Projection of {dim}D Random Walk — "
        f"{x_label} vs {y_label}"
    )
    plot_title = title or auto_title

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        marker=marker_kwargs,
        line=dict(color="#4fc3f7", width=0.8),
        name="Trajectory",
        hovertemplate=f"{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>",
    ))

    # Origin
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(color="#ffd740", size=12, symbol="star"),
        name="Origin",
    ))

    # Start
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[y[0]],
        mode="markers",
        marker=dict(color="#00e676", size=10, symbol="circle"),
        name="Start",
    ))

    # End
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[y[-1]],
        mode="markers",
        marker=dict(color="#ff5252", size=10, symbol="square"),
        name="End",
    ))

    annotation_text = (
        f"⚠️ This is a 2D projection of a {dim}D random walk. "
        f"Showing coordinates {x_label} and {y_label}."
    )

    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=15, color="#e0e0e0")),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc", family="Inter, sans-serif"),
        xaxis=dict(
            title=x_label,
            gridcolor=GRID_COLOR,
            zerolinecolor="#37474f",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title=y_label,
            gridcolor=GRID_COLOR,
            zerolinecolor="#37474f",
        ),
        annotations=[
            dict(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0, y=-0.12,
                showarrow=False,
                font=dict(size=11, color="#90a4ae"),
                align="left",
            )
        ],
        margin=dict(l=60, r=30, t=60, b=80),
    )
    return fig


def plot_projection_grid(
    trajectory: NDArray[np.int32],
    dimension: int,
    max_points: int = 2000,
) -> go.Figure:
    """
    Create a grid of all pairwise 2D projections for a d-dimensional walk.
    For d=4: 6 projections. For d=5: 10 projections.
    Shows up to a 3×3 grid.
    """
    from plotly.subplots import make_subplots
    import plotly.colors as pc

    d = dimension
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    n_pairs = len(pairs)

    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    subplot_titles = [
        f"{COORD_NAMES[i]} vs {COORD_NAMES[j]}"
        for i, j in pairs
    ]

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    n = len(trajectory)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        traj_plot = trajectory[idx]
    else:
        traj_plot = trajectory

    t_color = np.linspace(0, 1, len(traj_plot))

    for idx_pair, (i, j) in enumerate(pairs):
        row = idx_pair // ncols + 1
        col = idx_pair % ncols + 1

        x = traj_plot[:, i].astype(np.float64)
        y = traj_plot[:, j].astype(np.float64)

        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="lines+markers",
                marker=dict(color=t_color, colorscale="Viridis", size=2),
                line=dict(color="#4fc3f7", width=0.8),
                showlegend=False,
                name=f"{COORD_NAMES[i]}-{COORD_NAMES[j]}",
            ),
            row=row, col=col,
        )

        # Origin marker
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode="markers",
                marker=dict(color="#ffd740", size=8, symbol="star"),
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc"),
        title=dict(
            text=f"All 2D Projections of {d}D Random Walk",
            font=dict(size=16, color="#e0e0e0"),
        ),
        height=350 * nrows,
        margin=dict(l=50, r=30, t=80, b=50),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor="#37474f")
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor="#37474f")
    return fig
