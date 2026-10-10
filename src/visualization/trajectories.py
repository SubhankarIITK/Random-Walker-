"""
trajectories.py
===============
Plotly-based interactive trajectory visualizations for 1D, 2D, and 3D random walks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
TRAJ_COLOR   = "#4fc3f7"   # light blue for trajectory
START_COLOR  = "#00e676"   # green for start
END_COLOR    = "#ff5252"   # red for end
ORIGIN_COLOR = "#ffd740"   # amber for origin marker
BG_COLOR     = "#0d1117"   # dark background
GRID_COLOR   = "#1e2a38"
PAPER_COLOR  = "#0d1117"


def _dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply dark scientific theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#cfd8dc", family="Inter, sans-serif"),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor="#37474f",
        tickfont=dict(color="#90a4ae"),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor="#37474f",
        tickfont=dict(color="#90a4ae"),
    )
    return fig


def plot_1d_trajectory(
    trajectory: NDArray[np.int32],
    title: str = "1D Random Walk — Position vs Time",
    max_points: int = 5000,
) -> go.Figure:
    """
    Plot 1D random walk: x-position vs time step.
    """
    n_steps = len(trajectory) - 1
    steps = np.arange(n_steps + 1)
    pos   = trajectory[:, 0].astype(np.float64)

    # Downsample for performance if very long
    if len(steps) > max_points:
        idx  = np.linspace(0, len(steps) - 1, max_points, dtype=int)
        steps = steps[idx]
        pos   = pos[idx]

    fig = go.Figure()

    # Trajectory
    fig.add_trace(go.Scatter(
        x=steps, y=pos,
        mode="lines",
        line=dict(color=TRAJ_COLOR, width=1.5),
        name="Position",
        hovertemplate="Step %{x}<br>x = %{y}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=ORIGIN_COLOR,
                  annotation_text="Origin", annotation_position="right",
                  line_width=1)

    # Start / end markers
    fig.add_trace(go.Scatter(
        x=[steps[0]], y=[pos[0]],
        mode="markers",
        marker=dict(color=START_COLOR, size=10, symbol="circle"),
        name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[steps[-1]], y=[pos[-1]],
        mode="markers",
        marker=dict(color=END_COLOR, size=10, symbol="square"),
        name="End",
    ))

    fig.update_xaxes(title_text="Time step t")
    fig.update_yaxes(title_text="Position x")
    return _dark_layout(fig, title)


def plot_2d_trajectory(
    trajectory: NDArray[np.int32],
    title: str = "2D Random Walk — Trajectory",
    max_points: int = 5000,
) -> go.Figure:
    """
    Plot 2D random walk x-y trajectory.
    """
    n_steps = len(trajectory) - 1
    x = trajectory[:, 0].astype(np.float64)
    y = trajectory[:, 1].astype(np.float64)

    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x, y = x[idx], y[idx]

    # Color by time
    t_color = np.linspace(0, 1, len(x))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        marker=dict(
            color=t_color,
            colorscale="Blues",
            size=2,
            showscale=True,
            colorbar=dict(title="Time →", thickness=12),
        ),
        line=dict(color=TRAJ_COLOR, width=1),
        name="Trajectory",
        hovertemplate="x=%{x}<br>y=%{y}<extra></extra>",
    ))

    # Origin
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(color=ORIGIN_COLOR, size=12, symbol="star"),
        name="Origin",
    ))

    # Start / End
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[y[0]],
        mode="markers",
        marker=dict(color=START_COLOR, size=10, symbol="circle"),
        name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[y[-1]],
        mode="markers",
        marker=dict(color=END_COLOR, size=10, symbol="square"),
        name="End",
    ))

    fig.update_xaxes(title_text="x₁", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="x₂")
    return _dark_layout(fig, title)


def plot_3d_trajectory(
    trajectory: NDArray[np.int32],
    title: str = "3D Random Walk — Trajectory",
    max_points: int = 3000,
) -> go.Figure:
    """
    Interactive 3D plot of a random walk trajectory.
    """
    # Ensure trajectory has at least 3 dimensions to avoid IndexError
    d = trajectory.shape[1] if len(trajectory.shape) > 1 else 1
    
    x = trajectory[:, 0].astype(np.float64) if d > 0 else np.zeros(len(trajectory))
    y = trajectory[:, 1].astype(np.float64) if d > 1 else np.zeros(len(trajectory))
    z = trajectory[:, 2].astype(np.float64) if d > 2 else np.zeros(len(trajectory))

    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x, y, z = x[idx], y[idx], z[idx]

    t_color = np.linspace(0, 1, len(x))

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        marker=dict(
            color=t_color,
            colorscale="Viridis",
            size=2,
            showscale=True,
            colorbar=dict(title="Time", thickness=12),
        ),
        line=dict(color=TRAJ_COLOR, width=2),
        name="Trajectory",
    ))

    # Origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(color=ORIGIN_COLOR, size=8, symbol="diamond"),
        name="Origin",
    ))

    # Start / End
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers",
        marker=dict(color=START_COLOR, size=8),
        name="Start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers",
        marker=dict(color=END_COLOR, size=8),
        name="End",
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR, title="x₁"),
            yaxis=dict(backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR, title="x₂"),
            zaxis=dict(backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR, title="x₃"),
            bgcolor=BG_COLOR,
        ),
        paper_bgcolor=PAPER_COLOR,
        font=dict(color="#cfd8dc"),
        title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def plot_multiple_walks_2d(
    trajectories: list[NDArray[np.int32]],
    title: str = "2D Random Walk — Multiple Walkers",
    max_walkers_display: int = 20,
    max_points_each: int = 1000,
) -> go.Figure:
    """
    Overlay multiple 2D trajectories on one plot (up to max_walkers_display).
    """
    import plotly.colors as pc

    fig = go.Figure()
    colors = pc.sample_colorscale("Viridis", max(len(trajectories), 2))

    for i, traj in enumerate(trajectories[:max_walkers_display]):
        x = traj[:, 0].astype(np.float64)
        y = traj[:, 1].astype(np.float64)
        if len(x) > max_points_each:
            idx = np.linspace(0, len(x) - 1, max_points_each, dtype=int)
            x, y = x[idx], y[idx]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=colors[i], width=1),
            opacity=0.6,
            showlegend=(i == 0),
            name="Walkers" if i == 0 else None,
        ))

    # Origin
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(color=ORIGIN_COLOR, size=12, symbol="star"),
        name="Origin",
    ))

    fig.update_xaxes(title_text="x₁", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="x₂")
    return _dark_layout(fig, title)


def plot_msd_vs_time(
    time_steps: NDArray,
    msd_sim: NDArray,
    msd_theory: NDArray,
    ci_low: NDArray | None = None,
    ci_high: NDArray | None = None,
    title: str = "Mean Square Displacement vs Time",
) -> go.Figure:
    """
    Plot simulation MSD and theoretical MSD on the same axes.
    """
    fig = go.Figure()

    # Simulation
    fig.add_trace(go.Scatter(
        x=time_steps, y=msd_sim,
        mode="lines",
        line=dict(color=TRAJ_COLOR, width=2),
        name="Simulation ⟨R²(t)⟩",
    ))

    # CI band
    if ci_low is not None and ci_high is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_steps, time_steps[::-1]]),
            y=np.concatenate([ci_high, ci_low[::-1]]),
            fill="toself",
            fillcolor="rgba(79,195,247,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            showlegend=True,
        ))

    # Theory
    fig.add_trace(go.Scatter(
        x=time_steps, y=msd_theory,
        mode="lines",
        line=dict(color="#ff7043", width=2, dash="dash"),
        name="Theory E[R²] = t",
    ))

    fig.update_xaxes(title_text="Step t")
    fig.update_yaxes(title_text="⟨R²(t)⟩")
    return _dark_layout(fig, title)


def plot_animated_2d_trajectory(
    trajectory: NDArray[np.int32],
    title: str = "2D Random Walk — Animated Live View",
    max_frames: int = 200,
) -> go.Figure:
    """
    Plot an animated 2D random walk with play/pause and speed controls.
    """
    n_steps = len(trajectory) - 1
    x = trajectory[:, 0].astype(np.float64)
    y = trajectory[:, 1].astype(np.float64)

    # Downsample frames if necessary to keep HTML size reasonable
    if n_steps > max_frames:
        idx = np.linspace(0, n_steps, max_frames, dtype=int)
    else:
        idx = np.arange(n_steps + 1)
        
    frames = []
    for k in idx:
        frames.append(go.Frame(
            data=[go.Scatter(x=x[:k+1], y=y[:k+1], mode="lines+markers",
                             marker=dict(size=4, color=TRAJ_COLOR),
                             line=dict(color=TRAJ_COLOR, width=2))],
            name=str(k)
        ))
        
    fig = go.Figure(
        data=[go.Scatter(x=[x[0]], y=[y[0]], mode="lines+markers",
                         marker=dict(size=4, color=TRAJ_COLOR),
                         line=dict(color=TRAJ_COLOR, width=2),
                         name="Trajectory")],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
            paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
            font=dict(color="#cfd8dc", family="Inter, sans-serif"),
            margin=dict(l=60, r=30, t=60, b=60),
            xaxis=dict(range=[np.min(x)-2, np.max(x)+2], title="x₁", gridcolor=GRID_COLOR, zerolinecolor="#37474f"),
            yaxis=dict(range=[np.min(y)-2, np.max(y)+2], title="x₂", gridcolor=GRID_COLOR, zerolinecolor="#37474f", scaleanchor="x", scaleratio=1),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15, x=0.0, xanchor="left", yanchor="top", direction="right",
                buttons=[
                    dict(label="▶ Play (Normal)", method="animate", args=[None, dict(frame=dict(duration=80, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏩ Play (Fast)", method="animate", args=[None, dict(frame=dict(duration=15, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
                ]
            )]
        ),
        frames=frames
    )
    # Add origin marker
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color=ORIGIN_COLOR, size=12, symbol="star"), name="Origin"))
    return fig


def plot_animated_1d_trajectory(
    trajectory: NDArray[np.int32],
    title: str = "1D Random Walk — Animated Live View",
    max_frames: int = 200,
) -> go.Figure:
    n_steps = len(trajectory) - 1
    steps = np.arange(n_steps + 1)
    pos = trajectory[:, 0].astype(np.float64)

    if n_steps > max_frames:
        idx = np.linspace(0, n_steps, max_frames, dtype=int)
    else:
        idx = np.arange(n_steps + 1)
        
    frames = []
    for k in idx:
        frames.append(go.Frame(
            data=[go.Scatter(x=steps[:k+1], y=pos[:k+1], mode="lines+markers",
                             marker=dict(size=4, color=TRAJ_COLOR),
                             line=dict(color=TRAJ_COLOR, width=1.5))],
            name=str(k)
        ))
        
    fig = go.Figure(
        data=[go.Scatter(x=[steps[0]], y=[pos[0]], mode="lines+markers",
                         marker=dict(size=4, color=TRAJ_COLOR),
                         line=dict(color=TRAJ_COLOR, width=1.5),
                         name="Position")],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
            paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
            font=dict(color="#cfd8dc", family="Inter, sans-serif"),
            margin=dict(l=60, r=30, t=60, b=60),
            xaxis=dict(range=[0, n_steps], title="Time step t", gridcolor=GRID_COLOR, zerolinecolor="#37474f"),
            yaxis=dict(range=[np.min(pos)-2, np.max(pos)+2], title="Position x", gridcolor=GRID_COLOR, zerolinecolor="#37474f"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15, x=0.0, xanchor="left", yanchor="top", direction="right",
                buttons=[
                    dict(label="▶ Play (Normal)", method="animate", args=[None, dict(frame=dict(duration=80, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏩ Play (Fast)", method="animate", args=[None, dict(frame=dict(duration=15, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
                ]
            )]
        ),
        frames=frames
    )
    fig.add_hline(y=0, line_dash="dash", line_color=ORIGIN_COLOR, annotation_text="Origin", annotation_position="right", line_width=1)
    return fig


def plot_animated_3d_trajectory(
    trajectory: NDArray[np.int32],
    title: str = "3D Random Walk — Animated Live View",
    max_frames: int = 150,
) -> go.Figure:
    n_steps = len(trajectory) - 1
    d = trajectory.shape[1] if len(trajectory.shape) > 1 else 1
    
    x = trajectory[:, 0].astype(np.float64) if d > 0 else np.zeros(len(trajectory))
    y = trajectory[:, 1].astype(np.float64) if d > 1 else np.zeros(len(trajectory))
    z = trajectory[:, 2].astype(np.float64) if d > 2 else np.zeros(len(trajectory))

    if n_steps > max_frames:
        idx = np.linspace(0, n_steps, max_frames, dtype=int)
    else:
        idx = np.arange(n_steps + 1)
        
    frames = []
    for k in idx:
        frames.append(go.Frame(
            data=[go.Scatter3d(x=x[:k+1], y=y[:k+1], z=z[:k+1], mode="lines+markers",
                               marker=dict(size=3, color=TRAJ_COLOR),
                               line=dict(color=TRAJ_COLOR, width=2))],
            name=str(k)
        ))
        
    fig = go.Figure(
        data=[go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode="lines+markers",
                           marker=dict(size=3, color=TRAJ_COLOR),
                           line=dict(color=TRAJ_COLOR, width=2),
                           name="Trajectory")],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
            paper_bgcolor=PAPER_COLOR,
            font=dict(color="#cfd8dc", family="Inter, sans-serif"),
            margin=dict(l=0, r=0, t=60, b=0),
            scene=dict(
                xaxis=dict(range=[np.min(x)-1, np.max(x)+1], backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR, title="x₁"),
                yaxis=dict(range=[np.min(y)-1, np.max(y)+1], backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR, title="x₂"),
                zaxis=dict(range=[np.min(z)-1, np.max(z)+1], backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR, title="x₃"),
                bgcolor=BG_COLOR
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.0, x=0.0, xanchor="left", yanchor="top", direction="right",
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=80, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏩ Fast", method="animate", args=[None, dict(frame=dict(duration=15, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
                ]
            )]
        ),
        frames=frames
    )
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(color=ORIGIN_COLOR, size=6, symbol="diamond"), name="Origin"))
    return fig
