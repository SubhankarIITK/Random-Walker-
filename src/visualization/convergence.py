"""
convergence.py (visualization)
================================
Plotly plots for Monte Carlo convergence analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BG_COLOR    = "#0d1117"
GRID_COLOR  = "#1e2a38"
PAPER_COLOR = "#0d1117"


def plot_convergence(
    conv_df: pd.DataFrame,
    theory_value: float,
    title: str = "Monte Carlo Convergence — MSD",
) -> go.Figure:
    """
    Plot MSD estimate vs number of walkers, showing convergence to theory.

    Parameters
    ----------
    conv_df : DataFrame with columns: n_walkers, msd_sim, ci95_low, ci95_high
    theory_value : float  — theoretical MSD
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "MSD Estimate vs Number of Walkers",
            "Relative Error (%) vs Number of Walkers",
        ],
        vertical_spacing=0.18,
    )

    nw = conv_df["n_walkers"].values
    msd = conv_df["msd_sim"].values
    pct = conv_df["pct_error"].values

    ci_low  = conv_df.get("ci95_low",  conv_df["msd_sim"] - conv_df["se"] * 1.96).values
    ci_high = conv_df.get("ci95_high", conv_df["msd_sim"] + conv_df["se"] * 1.96).values

    # --- Row 1: MSD estimate ---
    # CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([nw, nw[::-1]]),
        y=np.concatenate([ci_high, ci_low[::-1]]),
        fill="toself",
        fillcolor="rgba(79,195,247,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        showlegend=True,
    ), row=1, col=1)

    # Simulation estimate
    fig.add_trace(go.Scatter(
        x=nw, y=msd,
        mode="lines+markers",
        line=dict(color="#4fc3f7", width=2),
        marker=dict(size=6),
        name="Simulation MSD",
    ), row=1, col=1)

    # Theory line
    fig.add_trace(go.Scatter(
        x=nw, y=[theory_value] * len(nw),
        mode="lines",
        line=dict(color="#ff7043", width=2, dash="dash"),
        name=f"Theory = {theory_value:.1f}",
    ), row=1, col=1)

    # --- Row 2: Relative error ---
    fig.add_trace(go.Scatter(
        x=nw, y=pct,
        mode="lines+markers",
        line=dict(color="#ffd740", width=2),
        marker=dict(size=6),
        name="Relative Error (%)",
    ), row=2, col=1)

    # 1/sqrt(N) reference line
    ref_scale = pct[0] * np.sqrt(nw[0])
    ref_line  = ref_scale / np.sqrt(nw)
    fig.add_trace(go.Scatter(
        x=nw, y=ref_line,
        mode="lines",
        line=dict(color="#80cbc4", width=1.5, dash="dot"),
        name="∝ 1/√N reference",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc"),
        height=600,
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        type="log",
        title_text="Number of walkers (log scale)",
        gridcolor=GRID_COLOR,
        zerolinecolor="#37474f",
    )
    fig.update_yaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="⟨R²⟩", row=1, col=1)
    fig.update_yaxes(title_text="Relative Error (%)", type="log", row=2, col=1)

    return fig


def plot_msd_error_vs_walkers(
    conv_df: pd.DataFrame,
    title: str = "MSD Relative Error vs Ensemble Size",
) -> go.Figure:
    """Simpler single-panel plot of error vs walkers."""
    fig = go.Figure()

    nw  = conv_df["n_walkers"].values
    pct = conv_df["pct_error"].values

    fig.add_trace(go.Scatter(
        x=nw, y=pct,
        mode="lines+markers",
        line=dict(color="#ffd740", width=2),
        marker=dict(size=8, symbol="circle"),
        name="Relative error (%)",
    ))

    # 1/sqrt(N) reference
    if len(nw) > 1 and pct[0] > 0:
        ref_scale = pct[0] * np.sqrt(nw[0])
        ref_line  = ref_scale / np.sqrt(nw)
        fig.add_trace(go.Scatter(
            x=nw, y=ref_line,
            mode="lines",
            line=dict(color="#80cbc4", width=1.5, dash="dot"),
            name="1/√N reference",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e0e0e0")),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc"),
        xaxis=dict(type="log", title="Number of walkers", gridcolor=GRID_COLOR),
        yaxis=dict(type="log", title="Relative Error (%)", gridcolor=GRID_COLOR),
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_scaling_fit(
    time_steps: np.ndarray,
    rms_sim: np.ndarray,
    alpha_fit: float,
    A_fit: float,
    alpha_theory: float = 0.5,
    title: str = "Scaling Exponent: log(R) vs log(t)",
) -> go.Figure:
    """
    Log-log plot of RMS vs time with fitted and theoretical power laws.
    """
    mask = (time_steps > 0) & (rms_sim > 0)
    t_plot = time_steps[mask]
    r_plot = rms_sim[mask]

    fig = go.Figure()

    # Simulation points
    fig.add_trace(go.Scatter(
        x=t_plot, y=r_plot,
        mode="markers",
        marker=dict(color="#4fc3f7", size=4, opacity=0.7),
        name="Simulation RMS",
    ))

    # Fitted power law
    r_fit = A_fit * t_plot ** alpha_fit
    fig.add_trace(go.Scatter(
        x=t_plot, y=r_fit,
        mode="lines",
        line=dict(color="#ffd740", width=2),
        name=f"Fit: α = {alpha_fit:.4f}",
    ))

    # Theoretical t^0.5
    A_th = 1.0  # prefactor = 1 for standard model
    r_th = A_th * t_plot ** alpha_theory
    fig.add_trace(go.Scatter(
        x=t_plot, y=r_th,
        mode="lines",
        line=dict(color="#ff7043", width=2, dash="dash"),
        name=f"Theory: α = {alpha_theory}",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e0e0e0")),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc"),
        xaxis=dict(type="log", title="Step t (log)", gridcolor=GRID_COLOR),
        yaxis=dict(type="log", title="RMS displacement (log)", gridcolor=GRID_COLOR),
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig
