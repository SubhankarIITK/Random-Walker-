"""
distributions.py (visualization)
=================================
Plotly plots for coordinate and radial distributions:
simulation histogram vs theoretical Gaussian overlay.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theory.distributions import (
    coordinate_gaussian_pdf,
    radial_distribution_nd,
)


BG_COLOR    = "#0d1117"
GRID_COLOR  = "#1e2a38"
PAPER_COLOR = "#0d1117"


def plot_coordinate_distribution(
    final_positions: NDArray[np.int32],
    n_steps: int,
    dimension: int,
    coord_idx: int = 0,
    title: str | None = None,
    n_bins: int = 60,
) -> go.Figure:
    """
    Histogram of final coordinate values vs theoretical Gaussian.

    Parameters
    ----------
    final_positions : (n_walkers, d)
    n_steps : int
    dimension : int
    coord_idx : int  — which coordinate to plot
    """
    coord_name = ["x₁", "x₂", "x₃", "x₄", "x₅"][coord_idx]
    values = final_positions[:, coord_idx].astype(np.float64)

    # Theoretical Gaussian
    x_range = np.linspace(values.min() - 1, values.max() + 1, 400)
    pdf_th  = coordinate_gaussian_pdf(x_range, n_steps, dimension)

    fig = go.Figure()

    # Empirical histogram (density=True equivalent: normalize to density)
    counts, bin_edges = np.histogram(values, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=(bin_edges[1] - bin_edges[0]) * 0.9,
        name="Simulation",
        marker=dict(color="#4fc3f7", opacity=0.7),
        hovertemplate=f"{coord_name}=%{{x:.2f}}<br>density=%{{y:.4f}}<extra></extra>",
    ))

    # Theoretical PDF
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf_th,
        mode="lines",
        line=dict(color="#ff7043", width=2.5),
        name=f"Theory: N(0, {n_steps}/{dimension})",
    ))

    auto_title = (
        f"Distribution of {coord_name} — {dimension}D walk, "
        f"n={n_steps} steps"
    )
    fig.update_layout(
        title=dict(text=title or auto_title, font=dict(size=15, color="#e0e0e0")),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc"),
        xaxis=dict(title=coord_name, gridcolor=GRID_COLOR, zerolinecolor="#37474f"),
        yaxis=dict(title="Probability density", gridcolor=GRID_COLOR),
        barmode="overlay",
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_radial_distribution(
    final_positions: NDArray[np.int32],
    n_steps: int,
    dimension: int,
    title: str | None = None,
    n_bins: int = 60,
) -> go.Figure:
    """
    Histogram of final |X| values vs theoretical radial PDF.
    """
    pos = final_positions.astype(np.float64)
    r_vals = np.sqrt(np.sum(pos ** 2, axis=1))

    r_range = np.linspace(0, r_vals.max() * 1.05, 400)
    pdf_th  = radial_distribution_nd(r_range, n_steps, dimension)

    fig = go.Figure()

    counts, bin_edges = np.histogram(r_vals, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=(bin_edges[1] - bin_edges[0]) * 0.9,
        name="Simulation |X|",
        marker=dict(color="#ce93d8", opacity=0.7),
    ))

    fig.add_trace(go.Scatter(
        x=r_range, y=pdf_th,
        mode="lines",
        line=dict(color="#ff7043", width=2.5),
        name="Theory (chi distribution)",
    ))

    auto_title = f"Radial Distribution |X| — {dimension}D walk, n={n_steps}"
    fig.update_layout(
        title=dict(text=title or auto_title, font=dict(size=15, color="#e0e0e0")),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc"),
        xaxis=dict(title="|X| (Euclidean distance from origin)", gridcolor=GRID_COLOR),
        yaxis=dict(title="Probability density", gridcolor=GRID_COLOR),
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_all_coord_distributions(
    final_positions: NDArray[np.int32],
    n_steps: int,
    dimension: int,
    n_bins: int = 50,
) -> go.Figure:
    """
    Grid of coordinate distribution plots for all d coordinates.
    """
    ncols = min(3, dimension)
    nrows = (dimension + ncols - 1) // ncols
    coord_names = ["x₁", "x₂", "x₃", "x₄", "x₅"]
    subplot_titles = [coord_names[i] for i in range(dimension)]

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    x_range_vals = np.linspace(
        final_positions.min() - 1, final_positions.max() + 1, 300
    )

    for ci in range(dimension):
        row = ci // ncols + 1
        col = ci % ncols + 1

        values = final_positions[:, ci].astype(np.float64)
        counts, bin_edges = np.histogram(values, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        pdf_th = coordinate_gaussian_pdf(x_range_vals, n_steps, dimension)

        fig.add_trace(
            go.Bar(
                x=bin_centers, y=counts,
                width=(bin_edges[1] - bin_edges[0]) * 0.9,
                marker=dict(color="#4fc3f7", opacity=0.7),
                showlegend=(ci == 0),
                name="Simulation",
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_range_vals, y=pdf_th,
                mode="lines",
                line=dict(color="#ff7043", width=2),
                showlegend=(ci == 0),
                name="Theory",
            ),
            row=row, col=col,
        )

    fig.update_layout(
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc"),
        title=dict(
            text=f"Coordinate Distributions — {dimension}D, n={n_steps}",
            font=dict(size=15, color="#e0e0e0"),
        ),
        height=350 * nrows,
        margin=dict(l=50, r=30, t=80, b=50),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor="#37474f")
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig
