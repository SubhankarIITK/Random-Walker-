"""
app.py
======
RANDOM WALK LABORATORY
Interactive simulation and statistical analysis of random walks from 1D to 5D.

IDC612 (2026) — IIT Kanpur
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import io

from src.simulation.random_walk import (
    simulate_walk,
    squared_displacement,
    displacement,
    count_distinct_sites,
    distinct_sites_over_time,
    returns_to_origin,
    first_passage_times_ensemble,
)
from src.simulation.ensemble import simulate_ensemble, ensemble_final_positions
from src.simulation.biased_walk import (
    simulate_biased_walk,
    biased_ensemble_final,
    theoretical_biased_mean,
)
from src.theory.moments import (
    msd_theory, coordinate_variance, rms_displacement_theory,
    coordinate_std_theory, msd_at_times,
)
from src.theory.distributions import coordinate_gaussian_pdf, radial_distribution_nd
from src.theory.scaling import (
    theoretical_scaling_exponent, log_log_fit_alpha,
)
from src.theory.recurrence import (
    is_recurrent, recurrence_classification,
    theoretical_return_probability, recurrence_note, finite_return_rate_caveat,
)
from src.analysis.statistics import (
    compute_walk_statistics, compute_ensemble_statistics,
    compute_return_statistics, unique_sites_statistics,
    build_comparison_dataframe,
)
from src.analysis.convergence import run_convergence_experiment
from src.analysis.fitting import fit_power_law, fit_msd_slope
from src.analysis.uncertainty import standard_error, confidence_interval_95
from src.visualization.trajectories import (
    plot_1d_trajectory, plot_2d_trajectory, plot_3d_trajectory,
    plot_multiple_walks_2d, plot_msd_vs_time, plot_animated_2d_trajectory,
    plot_animated_1d_trajectory, plot_animated_3d_trajectory
)
from src.visualization.projections import plot_projection, plot_projection_grid
from src.visualization.distributions import (
    plot_coordinate_distribution, plot_radial_distribution,
    plot_all_coord_distributions,
)
from src.visualization.convergence import (
    plot_convergence, plot_scaling_fit,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Random Walk Laboratory — IDC612",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-card:      #1c2128;
    --accent-blue:  #4fc3f7;
    --accent-orange:#ff7043;
    --accent-green: #00e676;
    --accent-amber: #ffd740;
    --accent-purple:#ce93d8;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       #30363d;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Main header */
.lab-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.lab-title {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #4fc3f7 0%, #ce93d8 50%, #ff7043 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.lab-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
    letter-spacing: 0.03em;
}
.lab-tagline {
    font-size: 0.95rem;
    color: #607d8b;
    margin-top: 0.5rem;
    font-style: italic;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.35rem 0;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent-blue); }
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-size: 1.3rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-blue);
}
.metric-theory {
    font-size: 0.8rem;
    color: var(--accent-orange);
    margin-top: 0.15rem;
}
.metric-error {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}

/* Colored badges */
.badge-recurrent {
    background: rgba(0,230,118,0.18);
    color: #00e676;
    border: 1px solid rgba(0,230,118,0.4);
    border-radius: 6px;
    padding: 0.1rem 0.6rem;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-transient {
    background: rgba(255,112,67,0.18);
    color: #ff7043;
    border: 1px solid rgba(255,112,67,0.4);
    border-radius: 6px;
    padding: 0.1rem 0.6rem;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    border-left: 3px solid var(--accent-blue);
    padding-left: 0.8rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Caveat box */
.caveat-box {
    background: rgba(255,193,7,0.08);
    border: 1px solid rgba(255,193,7,0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #ffe082;
    margin: 0.8rem 0;
}

/* Warning box */
.warning-box {
    background: rgba(255,112,67,0.08);
    border: 1px solid rgba(255,112,67,0.3);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #ff8a65;
    margin: 0.6rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
}

/* Streamlit default element overrides */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0);
    box-shadow: 0 4px 12px rgba(21,101,192,0.4);
}

/* Divider */
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="lab-header">
    <p class="lab-title">🎲 RANDOM WALK LABORATORY</p>
    <p class="lab-subtitle">Interactive computational study of random walks from 1D to 5D</p>
    <p class="lab-tagline">
        "Explore how simple random microscopic motion produces predictable macroscopic statistical behavior."
    </p>
    <p style="font-size:0.8rem;color:#455a64;margin-top:0.4rem;">
        IDC612 (2026) &nbsp;·&nbsp; IIT Kanpur &nbsp;·&nbsp; Standard nearest-neighbor unbiased lattice random walk
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — GLOBAL CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    dimension = st.selectbox(
        "Dimension (d)",
        options=[1, 2, 3, 4, 5],
        index=1,
        help="Spatial dimension of the random walk.",
    )

    n_steps = st.slider(
        "Number of Steps",
        min_value=10,
        max_value=50_000,
        value=1_000,
        step=10,
        help="Steps per walker.",
    )

    n_walkers = st.slider(
        "Number of Walkers (Ensemble)",
        min_value=1,
        max_value=20_000,
        value=500,
        step=50,
        help="Walkers for Monte Carlo statistics.",
    )

    seed_mode = st.radio(
        "Random Seed",
        options=["Fixed seed", "Random seed"],
        horizontal=True,
    )
    if seed_mode == "Fixed seed":
        seed = st.number_input("Seed value", value=42, min_value=0, step=1)
    else:
        seed = None

    st.markdown("---")

    sim_mode = st.selectbox(
        "Simulation Mode",
        options=["Single Walk Explorer", "Monte Carlo Statistical Lab"],
        help="Single Walk: trajectory visualization. Monte Carlo: ensemble statistics.",
    )

    st.markdown("---")
    run_btn   = st.button("▶ Run Simulation", use_container_width=True, type="primary")
    reset_btn = st.button("↺ Reset", use_container_width=True)

    if seed is not None:
        st.caption(f"🔑 Seed: `{int(seed)}`")
    else:
        st.caption("🔑 Seed: random (not fixed)")

    # Performance warning
    total_ops = n_walkers * n_steps
    if total_ops > 5_000_000:
        st.markdown(
            '<div class="warning-box">⚠️ Large simulation: may be slow. '
            'Consider reducing steps or walkers.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem;color:#455a64;line-height:1.6;">'
        '<b>Model:</b> Nearest-neighbor unbiased lattice walk.<br>'
        'Each step changes exactly one coordinate by ±1.<br>'
        'Theory: E[R²] = n for all dimensions.<br>'
        'Scaling exponent: α = 0.5</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "single_traj" not in st.session_state:
    st.session_state.single_traj = None
if "ensemble_result" not in st.session_state:
    st.session_state.ensemble_result = None
if "last_config" not in st.session_state:
    st.session_state.last_config = None

if reset_btn:
    st.session_state.sim_result     = None
    st.session_state.single_traj    = None
    st.session_state.ensemble_result = None
    st.session_state.last_config     = None
    st.rerun()

config_key = (dimension, n_steps, n_walkers, seed, sim_mode)

if run_btn or st.session_state.sim_result is None:
    with st.spinner("Running simulation…"):
        _seed = int(seed) if seed is not None else None
        _rng  = np.random.default_rng(_seed)

        # Single walker
        single_seed = int(_rng.integers(0, 2**31))
        traj = simulate_walk(dimension, n_steps, seed=single_seed)
        st.session_state.single_traj = traj

        # Ensemble
        ens_seed = int(_rng.integers(0, 2**31))

        # Determine thinning for performance
        if n_steps > 5000:
            thinning = max(1, n_steps // 500)
        else:
            thinning = 1

        ens = simulate_ensemble(
            dimension=dimension,
            n_steps=n_steps,
            n_walkers=n_walkers,
            seed=ens_seed,
            store_trajectories=False,
            compute_returns=True,
            thinning=thinning,
        )
        st.session_state.ensemble_result = ens
        st.session_state.last_config = {
            "dimension": dimension,
            "n_steps": n_steps,
            "n_walkers": n_walkers,
            "seed": seed,
            "sim_mode": sim_mode,
        }
    st.session_state.sim_result = "done"


# ============================================================
# Retrieve results
# ============================================================
traj = st.session_state.single_traj
ens  = st.session_state.ensemble_result
cfg  = st.session_state.last_config

if cfg is not None:
    dimension = cfg["dimension"]
    n_steps = cfg["n_steps"]
    n_walkers = cfg["n_walkers"]
    seed = cfg["seed"]

walk_stats = compute_walk_statistics(traj, dimension)
ens_stats  = compute_ensemble_statistics(ens.final_positions, n_steps, dimension)
ret_stats  = compute_return_statistics(ens.n_returns, ens.ever_returned, dimension)


# ============================================================
# HELPER: metric card HTML
# ============================================================
def metric_card(label: str, value: str, theory: str = "", error: str = "") -> str:
    theory_html = f'<div class="metric-theory">Theory: {theory}</div>' if theory else ""
    error_html  = f'<div class="metric-error">{error}</div>' if error else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {theory_html}
        {error_html}
    </div>"""


def fmt_vec(v: list, ndigits: int = 2) -> str:
    return "(" + ", ".join(f"{x:.{ndigits}f}" for x in v) + ")"


# ============================================================
# TABS
# ============================================================
tab_explorer, tab_stats, tab_theory, tab_dim_compare, tab_experiments, tab_about = st.tabs([
    "🔭 Explorer",
    "📊 Statistical Lab",
    "📐 Theory vs Simulation",
    "🔢 Dimension Comparison",
    "🧪 Experiments",
    "📖 About & Methodology",
])


# ============================================================
# TAB 1 — EXPLORER
# ============================================================
with tab_explorer:
    st.markdown("## Random Walk Explorer")
    st.markdown(
        "Visualize the trajectory of a single walker alongside real-time statistical quantities. "
        "The same simulation feeds both panels."
    )

    col_left, col_right = st.columns([3, 2], gap="medium")

    with col_left:
        st.markdown('<div class="section-header">Trajectory Visualization</div>', unsafe_allow_html=True)

        if dimension == 1:
            fig = plot_1d_trajectory(traj, title=f"1D Random Walk — {n_steps} steps")
        elif dimension == 2:
            fig = plot_2d_trajectory(traj, title=f"2D Random Walk — {n_steps} steps")
        elif dimension == 3:
            fig = plot_3d_trajectory(traj, title=f"3D Random Walk — {n_steps} steps")
        else:
            # Projection explorer for 4D / 5D
            coord_options = [f"x{i+1}" for i in range(dimension)]
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                x_proj = st.selectbox("X axis", coord_options, index=0, key="xp")
            with col_p2:
                y_options = [c for c in coord_options if c != x_proj]
                y_proj = st.selectbox("Y axis", y_options, index=0, key="yp")
            with col_p3:
                color_options = ["None"] + [c for c in coord_options if c not in [x_proj, y_proj]]
                c_proj = st.selectbox("Color axis", color_options, index=0, key="cp")

            x_idx = coord_options.index(x_proj)
            y_idx = coord_options.index(y_proj)
            c_idx = None if c_proj == "None" else coord_options.index(c_proj)

            st.info(
                f"⚠️ Showing a 2D projection of a **{dimension}D** random walk. "
                f"Displaying coordinates **{x_proj}** vs **{y_proj}**."
            )
            fig = plot_projection(
                traj, x_axis=x_idx, y_axis=y_idx,
                color_axis=c_idx, dimension=dimension,
            )

        st.plotly_chart(fig, use_container_width=True, key="chart_fig")

        # Animated Live View for 1D, 2D, and 3D
        if dimension <= 3:
            with st.expander("🎬 Animation / Live View"):
                st.markdown(f"Use the controls below to play the live {dimension}D animation:")
                if dimension == 1:
                    fig_live = plot_animated_1d_trajectory(traj)
                    st.plotly_chart(fig_live, use_container_width=True, key="chart_live_1d")
                elif dimension == 2:
                    fig_live = plot_animated_2d_trajectory(traj)
                    st.plotly_chart(fig_live, use_container_width=True, key="chart_live_2d")
                elif dimension == 3:
                    fig_live = plot_animated_3d_trajectory(traj)
                    st.plotly_chart(fig_live, use_container_width=True, key="chart_live_3d")

    with col_right:
        st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)

        pos_str   = fmt_vec(walk_stats["final_position"])
        r_val     = walk_stats["final_r"]
        r2_val    = walk_stats["final_r2"]
        msd_th    = walk_stats["msd_theory"]
        abs_err   = walk_stats["abs_error_msd"]
        rel_err   = walk_stats["rel_error_msd"]
        pct_err   = walk_stats["pct_error_msd"]

        # Ensemble MSD (more robust estimator)
        msd_sim_ens = ens_stats["msd_sim"]
        msd_se      = ens_stats["msd_se"]

        cards_html = (
            metric_card("Current Step (n)", f"{n_steps:,}")
            + metric_card("Final Position (single walk)", pos_str)
            + metric_card(
                "Distance from Origin |X|",
                f"{r_val:.4f}",
                theory=f"√n = {np.sqrt(n_steps):.4f}",
                error=f"Single-walk value (fluctuates)"
            )
            + metric_card(
                "Squared Distance |X|² (single walk)",
                f"{r2_val:.1f}",
                theory=f"n = {n_steps:.0f}",
                error=f"Abs error: {abs_err:.2f} | {pct_err:.2f}%"
            )
            + metric_card(
                f"Ensemble MSD ({n_walkers:,} walkers)",
                f"{msd_sim_ens:.4f}",
                theory=f"n = {msd_th:.1f}",
                error=f"SE: ±{msd_se:.4f} | Rel error: {ens_stats['pct_error_msd']:.3f}%"
            )
            + metric_card("RMS Displacement (sim)", f"{ens_stats['rms_sim']:.4f}",
                          theory=f"√n = {ens_stats['rms_theory']:.4f}")
            + metric_card(
                "Dimension",
                f"{dimension}D",
                theory=f"Moves: {2*dimension} directions"
            )
            + metric_card(
                "Return Classification",
                recurrence_classification(dimension),
                theory=f"Polya (1921)"
            )
        )

        st.markdown(cards_html, unsafe_allow_html=True)

        # Scaling exponent estimate
        ts = ens.time_steps
        rms_vals = ens.mean_displacement
        mask = ts > 5
        if np.sum(mask) > 4:
            alpha_fit, _, r2_fit = log_log_fit_alpha(ts[mask].astype(float), rms_vals[mask])
            st.markdown('<div class="section-header">Scaling Exponent</div>', unsafe_allow_html=True)
            st.markdown(
                metric_card(
                    "Fitted α (log-log slope)",
                    f"{alpha_fit:.4f}",
                    theory="α = 0.5000",
                    error=f"R² = {r2_fit:.4f}"
                ),
                unsafe_allow_html=True,
            )


# ============================================================
# TAB 2 — STATISTICAL LABORATORY
# ============================================================
with tab_stats:
    st.markdown("## Statistical Laboratory")
    st.markdown(
        "Quantitative analysis of the ensemble simulation. "
        "All values come from actual simulation — no fabricated results."
    )

    # ---- Metric row ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Ensemble MSD", f"{ens_stats['msd_sim']:.4f}",
                  delta=f"Theory: {ens_stats['msd_theory']:.1f}")
    with c2:
        st.metric("RMS Displacement", f"{ens_stats['rms_sim']:.4f}",
                  delta=f"Theory: {ens_stats['rms_theory']:.4f}")
    with c3:
        st.metric("Rel. Error MSD", f"{ens_stats['pct_error_msd']:.4f}%")
    with c4:
        st.metric("Frac. Returned to Origin", f"{ret_stats['fraction_returned_sim']:.4f}",
                  delta=f"Theory ∞: {ret_stats.get('theory_return_probability', 'N/A')}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Coordinate Statistics")
        coord_data = []
        for i in range(dimension):
            var_sim = ens_stats["coord_vars_sim"][i]
            var_th  = ens_stats["coord_var_theory"]
            err_pct = abs(var_sim - var_th) / var_th * 100 if var_th > 0 else 0
            coord_data.append({
                "Coord": f"x{i+1}",
                "Mean (sim)":   f"{ens_stats['coord_means'][i]:.4f}",
                "Var (sim)":    f"{var_sim:.4f}",
                "Var (theory)": f"{var_th:.4f}",
                "Error (%)":    f"{err_pct:.3f}%",
            })
        st.dataframe(pd.DataFrame(coord_data), use_container_width=True, hide_index=True)

        st.markdown("### Return-to-Origin Statistics")
        ret_data = {
            "Metric": [
                "Walkers returning (≥1 return)",
                "Fraction returned",
                "Mean return count",
                "Theoretical classification",
                "Theoretical return probability",
            ],
            "Value": [
                f"{ret_stats['n_walkers_returned']:,} / {n_walkers:,}",
                f"{ret_stats['fraction_returned_sim']:.4f}",
                f"{ret_stats['mean_return_count']:.3f}",
                ret_stats['recurrence_class'],
                str(ret_stats.get('theory_return_probability', 'N/A')),
            ],
        }
        st.dataframe(pd.DataFrame(ret_data), use_container_width=True, hide_index=True)

        # Caveat
        st.markdown(
            '<div class="caveat-box">' + finite_return_rate_caveat() + "</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("### Coordinate Distribution (x₁)")
        fig_dist = plot_coordinate_distribution(
            ens.final_positions, n_steps, dimension, coord_idx=0
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="chart_fig_dist")

        st.markdown("### Radial Distribution |X|")
        fig_rad = plot_radial_distribution(ens.final_positions, n_steps, dimension)
        st.plotly_chart(fig_rad, use_container_width=True, key="chart_fig_rad")

    st.markdown("---")
    st.markdown("### MSD vs Time")
    ts = ens.time_steps
    th = msd_at_times(ts)
    # Build CI from std
    se_band = ens.std_msd / np.sqrt(n_walkers)
    fig_msd = plot_msd_vs_time(
        ts, ens.mean_msd, th,
        ci_low=ens.mean_msd - 1.96 * se_band,
        ci_high=ens.mean_msd + 1.96 * se_band,
        title=f"MSD vs Time — {dimension}D, {n_walkers:,} walkers",
    )
    st.plotly_chart(fig_msd, use_container_width=True, key="chart_fig_msd")

    # Linear fit
    fit = fit_msd_slope(ts.astype(float), ens.mean_msd)
    st.markdown(
        f"**Linear fit:** slope = `{fit['slope']:.6f}` (theory = 1.000), "
        f"intercept = `{fit['intercept']:.4f}`, "
        f"R² = `{fit['r_squared']:.6f}`"
    )

    # All coordinate distributions
    if dimension > 1:
        with st.expander("All Coordinate Distributions"):
            fig_all = plot_all_coord_distributions(ens.final_positions, n_steps, dimension)
            st.plotly_chart(fig_all, use_container_width=True, key="chart_fig_all")

    # Export
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    export_df = pd.DataFrame({
        "Step": ens.time_steps,
        "MSD_sim": ens.mean_msd,
        "MSD_theory": msd_at_times(ens.time_steps),
        "std_MSD": ens.std_msd,
        "mean_disp": ens.mean_displacement,
        **{f"mean_x{i+1}": ens.mean_coord[:, i] for i in range(dimension)},
        **{f"var_x{i+1}": ens.var_coord[:, i] for i in range(dimension)},
    })
    csv_bytes = export_df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download MSD data (CSV)",
        data=csv_bytes,
        file_name=f"random_walk_d{dimension}_n{n_steps}_w{n_walkers}.csv",
        mime="text/csv",
    )
    config_json = json.dumps({
        "dimension": dimension,
        "n_steps": n_steps,
        "n_walkers": n_walkers,
        "seed": seed,
        "final_msd_sim": ens_stats["msd_sim"],
        "final_msd_theory": ens_stats["msd_theory"],
        "rel_error_pct": ens_stats["pct_error_msd"],
        "fraction_returned": ret_stats["fraction_returned_sim"],
    }, indent=2)
    st.download_button(
        "⬇ Download Config + Summary (JSON)",
        data=config_json.encode(),
        file_name=f"random_walk_config_d{dimension}.json",
        mime="application/json",
    )


# ============================================================
# TAB 3 — THEORY VS SIMULATION
# ============================================================
with tab_theory:
    st.markdown("## Theory vs Simulation")
    st.markdown(
        "Direct quantitative comparison between simulation output and analytical predictions. "
        "All simulation values come from actual computation."
    )

    # ---- Experiment A: MSD vs t ----
    st.markdown("### Experiment A — Mean Square Displacement E[R²(t)] vs t")

    col_ta, col_tb = st.columns([2, 1])
    with col_ta:
        ts   = ens.time_steps
        th   = msd_at_times(ts)
        se_b = ens.std_msd / np.sqrt(n_walkers)
        fig_A = plot_msd_vs_time(
            ts, ens.mean_msd, th,
            ci_low=ens.mean_msd - 1.96 * se_b,
            ci_high=ens.mean_msd + 1.96 * se_b,
        )
        st.plotly_chart(fig_A, use_container_width=True, key="chart_fig_A")

    with col_tb:
        fit_A = fit_msd_slope(ts.astype(float), ens.mean_msd)
        st.markdown("**Fit Results:**")
        rows_A = {
            "Metric": ["MSD Slope", "Theory slope", "Intercept", "R²",
                       "Final MSD (sim)", "Final MSD (theory)", "Error (%)"],
            "Value": [
                f"{fit_A['slope']:.6f}",
                "1.000000",
                f"{fit_A['intercept']:.4f}",
                f"{fit_A['r_squared']:.6f}",
                f"{ens_stats['msd_sim']:.4f}",
                f"{ens_stats['msd_theory']:.1f}",
                f"{ens_stats['pct_error_msd']:.4f}%",
            ],
        }
        st.dataframe(pd.DataFrame(rows_A), use_container_width=True, hide_index=True)

    st.markdown("---")
    # ---- Experiment B: RMS vs t ----
    st.markdown("### Experiment B — RMS Displacement vs t (diffusive scaling)")

    rms_sim = ens.mean_displacement
    rms_th  = np.sqrt(ts.astype(float))
    mask    = ts > 5

    col_bc, col_bd = st.columns([2, 1])
    with col_bc:
        fig_B = go.Figure()
        fig_B.add_trace(go.Scatter(
            x=ts, y=rms_sim, mode="lines",
            line=dict(color="#4fc3f7", width=2),
            name="Simulation ⟨|X|⟩"
        ))
        fig_B.add_trace(go.Scatter(
            x=ts, y=rms_th, mode="lines",
            line=dict(color="#ff7043", width=2, dash="dash"),
            name="Theory √t"
        ))
        fig_B.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#cfd8dc"),
            xaxis=dict(title="Step t", gridcolor="#1e2a38"),
            yaxis=dict(title="RMS Displacement", gridcolor="#1e2a38"),
            title=dict(text="RMS Displacement vs t", font=dict(size=15, color="#e0e0e0")),
            margin=dict(l=60, r=30, t=60, b=60),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_B, use_container_width=True, key="chart_fig_B")

    with col_bd:
        if np.sum(mask) > 4:
            alpha_fit, A_fit, r2_fit = log_log_fit_alpha(ts[mask].astype(float), rms_sim[mask])
        else:
            alpha_fit, A_fit, r2_fit = float("nan"), float("nan"), float("nan")
        st.markdown("**Scaling Exponent Fit:**")
        rows_B = {
            "Metric": ["Fitted α", "Theory α", "Prefactor A", "R² (log-log)", "Deviation from ½"],
            "Value": [
                f"{alpha_fit:.5f}",
                "0.50000",
                f"{A_fit:.5f}",
                f"{r2_fit:.6f}",
                f"{abs(alpha_fit - 0.5):.6f}",
            ],
        }
        st.dataframe(pd.DataFrame(rows_B), use_container_width=True, hide_index=True)

    st.markdown("---")
    # ---- Experiment C: Scaling (log-log) ----
    st.markdown("### Experiment C — Log-Log Scaling Plot")

    if np.sum(mask) > 4 and not np.isnan(alpha_fit):
        fig_C = plot_scaling_fit(
            ts.astype(float), rms_sim,
            alpha_fit=alpha_fit, A_fit=A_fit,
        )
        st.plotly_chart(fig_C, use_container_width=True, key="chart_fig_C")

    st.markdown("---")
    # ---- Experiment D: Coordinate distribution ----
    st.markdown("### Experiment D — Coordinate Distribution vs Gaussian")

    coord_sel = st.selectbox(
        "Select coordinate", [f"x{i+1}" for i in range(dimension)], key="coord_dist_sel"
    )
    ci_dist = int(coord_sel[1]) - 1
    fig_D = plot_coordinate_distribution(ens.final_positions, n_steps, dimension, ci_dist)
    st.plotly_chart(fig_D, use_container_width=True, key="chart_fig_D")

    # KS test
    from scipy import stats as scipy_stats
    sigma = np.sqrt(n_steps / dimension)
    vals  = ens.final_positions[:, ci_dist].astype(float)
    ks_stat, ks_pval = scipy_stats.kstest(vals, lambda x: scipy_stats.norm.cdf(x, 0, sigma))
    st.markdown(
        f"**Kolmogorov–Smirnov test** vs N(0, {sigma**2:.2f}): "
        f"statistic = `{ks_stat:.5f}`, p-value = `{ks_pval:.4f}` "
        + ("✅ (cannot reject Gaussian)" if ks_pval > 0.05 else "⚠️ (deviates from Gaussian)")
    )

    st.markdown("---")
    # ---- Experiment E: Radial distribution ----
    st.markdown("### Experiment E — Radial Distribution |X| vs Theory")
    fig_E = plot_radial_distribution(ens.final_positions, n_steps, dimension)
    st.plotly_chart(fig_E, use_container_width=True, key="chart_fig_E")


# ============================================================
# TAB 4 — DIMENSION COMPARISON
# ============================================================
with tab_dim_compare:
    st.markdown("## Dimension Comparison")
    st.markdown(
        "Run identical experiments across all five dimensions to compare statistical properties. "
        "**Key insight:** E[R²] = n for ALL dimensions under this model. "
        "Yet many other properties vary significantly with dimension."
    )

    dims_to_compare = st.multiselect(
        "Dimensions to compare", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5]
    )
    n_steps_cmp = st.slider("Steps (comparison)", 100, 10_000, 2_000, step=100, key="cmp_steps")
    n_walk_cmp  = st.slider("Walkers (comparison)", 50, 5_000, 500, step=50, key="cmp_walkers")

    if st.button("▶ Run Dimension Comparison", key="run_dim_cmp"):
        cmp_results = {}
        _rng_cmp = np.random.default_rng(int(seed) if seed is not None else None)

        with st.spinner("Running comparison across dimensions…"):
            for d in dims_to_compare:
                d_seed = int(_rng_cmp.integers(0, 2**31))
                _thinning = max(1, n_steps_cmp // 300)
                _ens = simulate_ensemble(
                    d, n_steps_cmp, n_walk_cmp,
                    seed=d_seed,
                    store_trajectories=False,
                    compute_returns=True,
                    thinning=_thinning,
                )
                _es  = compute_ensemble_statistics(_ens.final_positions, n_steps_cmp, d)
                _rs  = compute_return_statistics(_ens.n_returns, _ens.ever_returned, d)

                # Scaling exponent from this dimension
                ts_d = _ens.time_steps
                rms_d = _ens.mean_displacement
                mask_d = ts_d > 5
                if np.sum(mask_d) > 4:
                    _alpha, _, _ = log_log_fit_alpha(ts_d[mask_d].astype(float), rms_d[mask_d])
                else:
                    _alpha = float("nan")

                cmp_results[d] = {
                    **_es,
                    "fraction_returned": _rs["fraction_returned_sim"],
                    "scaling_alpha": _alpha,
                    "recurrence_class": _rs["recurrence_class"],
                    "n_steps": n_steps_cmp,
                    "n_walkers": n_walk_cmp,
                    "_ens": _ens,
                }

        st.session_state["cmp_results"] = cmp_results

    if "cmp_results" in st.session_state:
        cmp = st.session_state["cmp_results"]

        # Summary table
        rows = []
        for d, r in sorted(cmp.items()):
            rows.append({
                "Dim d": d,
                "Walkers": r["n_walkers"],
                "Steps": r["n_steps"],
                "MSD (sim)": f"{r['msd_sim']:.3f}",
                "MSD (theory)": f"{r['msd_theory']:.1f}",
                "Error (%)": f"{r['pct_error_msd']:.4f}%",
                "RMS (sim)": f"{r['rms_sim']:.4f}",
                "Scaling α": f"{r['scaling_alpha']:.4f}",
                "Frac. Returned": f"{r['fraction_returned']:.4f}",
                "Classification": r["recurrence_class"],
            })
        df_cmp = pd.DataFrame(rows)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        # Key observations
        st.markdown("""
        <div class="caveat-box">
        <b>Key observation:</b> E[R²] ≈ n for every dimension — this is an exact result of the model.
        The MSD does <em>not</em> depend on dimension.
        However, the fraction of walkers returning to origin decreases sharply with dimension
        (recurrence → transience transition at d=3).
        </div>
        """, unsafe_allow_html=True)

        # MSD comparison plot
        st.markdown("### MSD vs Time — All Dimensions")
        fig_cmp_msd = go.Figure()
        colors_dim = ["#4fc3f7", "#00e676", "#ffd740", "#ff7043", "#ce93d8"]
        ts_ref = None
        for i, (d, r) in enumerate(sorted(cmp.items())):
            _ens_d = r["_ens"]
            ts_d   = _ens_d.time_steps
            ts_ref = ts_d
            fig_cmp_msd.add_trace(go.Scatter(
                x=ts_d, y=_ens_d.mean_msd,
                mode="lines",
                line=dict(color=colors_dim[i % 5], width=2),
                name=f"{d}D (sim)",
            ))
        if ts_ref is not None:
            fig_cmp_msd.add_trace(go.Scatter(
                x=ts_ref, y=ts_ref.astype(float),
                mode="lines",
                line=dict(color="white", width=2, dash="dash"),
                name="Theory E[R²] = t",
            ))
        fig_cmp_msd.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#cfd8dc"),
            xaxis=dict(title="Step t", gridcolor="#1e2a38"),
            yaxis=dict(title="⟨R²(t)⟩", gridcolor="#1e2a38"),
            title=dict(text="MSD vs Time — Dimension Comparison", font=dict(size=15, color="#e0e0e0")),
            margin=dict(l=60, r=30, t=60, b=60),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_cmp_msd, use_container_width=True, key="chart_fig_cmp_msd")

        # Return fraction comparison
        st.markdown("### Return Fraction vs Dimension")
        dims_r = sorted(cmp.keys())
        fracs  = [cmp[d]["fraction_returned"] for d in dims_r]
        theory_probs = [theoretical_return_probability(d) for d in dims_r]

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Bar(
            x=[f"{d}D" for d in dims_r],
            y=fracs,
            name="Simulation (finite horizon)",
            marker_color="#4fc3f7",
            opacity=0.8,
        ))
        fig_ret.add_trace(go.Scatter(
            x=[f"{d}D" for d in dims_r],
            y=theory_probs,
            mode="markers+lines",
            marker=dict(color="#ff7043", size=10, symbol="diamond"),
            line=dict(color="#ff7043", dash="dash"),
            name="Theory (infinite time)",
        ))
        fig_ret.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#cfd8dc"),
            xaxis=dict(title="Dimension", gridcolor="#1e2a38"),
            yaxis=dict(title="Fraction/Probability of Return", gridcolor="#1e2a38"),
            title=dict(text="Return to Origin — Theory vs Simulation", font=dict(size=15, color="#e0e0e0")),
            margin=dict(l=60, r=30, t=60, b=60),
            barmode="group",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_ret, use_container_width=True, key="chart_fig_ret")

        # Export comparison table
        csv_cmp = df_cmp.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Comparison Table (CSV)",
            data=csv_cmp,
            file_name="dimension_comparison.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 5 — EXPERIMENTS
# ============================================================
with tab_experiments:
    st.markdown("## Experiments")

    exp_tabs = st.tabs([
        "📈 MC Convergence",
        "🔄 Recurrence/Transience",
        "🗺️ Distinct Sites",
        "🏁 First Passage",
        "⚡ Biased Walk",
        "🔲 Projection Explorer",
    ])

    # ---- Experiment: MC Convergence ----
    with exp_tabs[0]:
        st.markdown("### Monte Carlo Convergence Experiment")
        st.markdown(
            "Demonstrate that as the ensemble size N grows, the simulated MSD converges "
            "toward the theoretical value E[R²] = n. All values are from actual simulation."
        )

        col_cv1, col_cv2 = st.columns(2)
        with col_cv1:
            conv_dim   = st.selectbox("Dimension", [1, 2, 3, 4, 5], index=1, key="conv_dim")
            conv_steps = st.slider("Steps", 100, 10_000, 1_000, step=100, key="conv_steps")
        with col_cv2:
            max_walkers_conv = st.selectbox(
                "Max walkers",
                [1_000, 5_000, 10_000, 50_000],
                index=1,
                key="conv_max_w",
            )

        if st.button("▶ Run Convergence Experiment", key="run_conv"):
            walker_counts = []
            w = 10
            while w <= max_walkers_conv:
                walker_counts.append(w)
                w = w * 10 if w * 10 <= max_walkers_conv else max_walkers_conv
                if walker_counts[-1] == max_walkers_conv:
                    break
            # Ensure we always have at least a few points
            walker_counts = sorted(set(
                [10, 100, 1000] + [max_walkers_conv]
            ))
            walker_counts = [w for w in walker_counts if w <= max_walkers_conv]

            with st.spinner("Running convergence experiment…"):
                conv_df = run_convergence_experiment(
                    conv_dim, conv_steps, walker_counts,
                    seed=int(seed) if seed is not None else None,
                )
            st.session_state["conv_df"] = conv_df
            st.session_state["conv_theory"] = float(msd_theory(conv_steps))

        if "conv_df" in st.session_state:
            conv_df  = st.session_state["conv_df"]
            th_val   = st.session_state["conv_theory"]
            fig_conv = plot_convergence(conv_df, th_val)
            st.plotly_chart(fig_conv, use_container_width=True, key="chart_fig_conv")

            st.dataframe(
                conv_df[["n_walkers", "msd_sim", "msd_theory", "abs_error", "pct_error", "se"]].round(5),
                use_container_width=True, hide_index=True,
            )
            csv_conv = conv_df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Convergence Data (CSV)",
                data=csv_conv,
                file_name="convergence.csv",
                mime="text/csv",
            )

    # ---- Experiment: Recurrence ----
    with exp_tabs[1]:
        st.markdown("### Recurrence / Transience Experiment")
        st.markdown(
            "Investigate the return-to-origin probability across dimensions. "
            "Compare finite-simulation estimates with theoretical predictions."
        )

        rec_steps   = st.slider("Steps", 100, 20_000, 5_000, step=100, key="rec_steps")
        rec_walkers = st.slider("Walkers", 100, 10_000, 2_000, step=100, key="rec_walk")

        if st.button("▶ Run Recurrence Experiment", key="run_rec"):
            rec_results = {}
            _rng_rec = np.random.default_rng(int(seed) if seed is not None else None)
            with st.spinner("Running recurrence experiment…"):
                for d in [1, 2, 3, 4, 5]:
                    d_seed = int(_rng_rec.integers(0, 2**31))
                    _ens_r = simulate_ensemble(
                        d, rec_steps, rec_walkers, seed=d_seed,
                        store_trajectories=False, compute_returns=True,
                        thinning=max(1, rec_steps // 200),
                    )
                    frac = float(np.mean(_ens_r.ever_returned))
                    mean_ret = float(np.mean(_ens_r.n_returns))
                    rec_results[d] = {
                        "frac": frac,
                        "mean_ret": mean_ret,
                        "n_walkers": rec_walkers,
                        "n_steps": rec_steps,
                    }
            st.session_state["rec_results"] = rec_results

        if "rec_results" in st.session_state:
            rec = st.session_state["rec_results"]

            rec_rows = []
            for d in [1, 2, 3, 4, 5]:
                if d not in rec:
                    continue
                r = rec[d]
                tp = theoretical_return_probability(d)
                rec_rows.append({
                    "Dim": f"{d}D",
                    "Classification": recurrence_classification(d),
                    "Frac. Returned (sim)": f"{r['frac']:.4f}",
                    "Theory Prob (∞ time)": str(tp),
                    "Mean Returns": f"{r['mean_ret']:.3f}",
                    "Steps": r["n_steps"],
                    "Walkers": r["n_walkers"],
                })
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

            # Bar chart
            dims_rec  = sorted(rec.keys())
            fracs_rec = [rec[d]["frac"] for d in dims_rec]
            th_probs  = [theoretical_return_probability(d) for d in dims_rec]

            fig_rec = go.Figure()
            fig_rec.add_trace(go.Bar(
                x=[f"{d}D" for d in dims_rec],
                y=fracs_rec,
                name="Sim (finite horizon)",
                marker_color=["#00e676" if d <= 2 else "#ff7043" for d in dims_rec],
                opacity=0.8,
            ))
            fig_rec.add_trace(go.Scatter(
                x=[f"{d}D" for d in dims_rec],
                y=th_probs,
                mode="markers",
                marker=dict(size=12, color="white", symbol="diamond"),
                name="Theory (∞ time)",
            ))
            fig_rec.add_hline(y=1.0, line_dash="dot", line_color="#ffd740",
                              annotation_text="Recurrent (p=1)", annotation_position="right")
            fig_rec.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#cfd8dc"),
                xaxis=dict(title="Dimension", gridcolor="#1e2a38"),
                yaxis=dict(title="Return Probability", gridcolor="#1e2a38", range=[0, 1.1]),
                title=dict(text="Recurrence: Return Probability by Dimension", font=dict(size=15, color="#e0e0e0")),
                margin=dict(l=60, r=30, t=60, b=60),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_rec, use_container_width=True, key="chart_fig_rec")

            st.markdown('<div class="caveat-box">' + finite_return_rate_caveat() + "</div>",
                        unsafe_allow_html=True)

            for d in [1, 2, 3, 4, 5]:
                if d in rec:
                    badge = (
                        '<span class="badge-recurrent">Recurrent</span>'
                        if is_recurrent(d)
                        else '<span class="badge-transient">Transient</span>'
                    )
                    st.markdown(f"**{d}D:** {badge} — {recurrence_note(d)}", unsafe_allow_html=True)
                    st.markdown("")

    # ---- Experiment: Distinct Sites ----
    with exp_tabs[2]:
        st.markdown("### Distinct Lattice Sites Visited")
        st.markdown(
            "Count S_n = number of unique lattice sites visited after n steps. "
            "Convention: the starting site (origin) is counted as site 1 at t=0."
        )

        site_dim   = st.selectbox("Dimension", [1, 2, 3, 4, 5], index=1, key="site_dim")
        site_steps = st.slider("Steps", 100, 5_000, 1_000, key="site_steps")

        if st.button("▶ Run Distinct Sites Experiment", key="run_sites"):
            _seed_s = int(seed) if seed is not None else None
            traj_s  = simulate_walk(site_dim, site_steps, seed=_seed_s)
            counts_s = distinct_sites_over_time(traj_s)
            st.session_state["sites_counts"] = counts_s
            st.session_state["sites_n"]      = site_steps

        if "sites_counts" in st.session_state:
            counts_s = st.session_state["sites_counts"]
            n_s      = st.session_state["sites_n"]
            ts_s     = np.arange(len(counts_s))

            st.metric("Total distinct sites visited", f"{counts_s[-1]:,}")
            st.metric("Total steps", f"{n_s:,}")
            st.metric("Revisit count (steps − distinct)", f"{n_s + 1 - int(counts_s[-1]):,}")
            st.metric("Coverage fraction", f"{counts_s[-1]/(n_s+1):.4f}")

            fig_sites = go.Figure()
            fig_sites.add_trace(go.Scatter(
                x=ts_s, y=counts_s,
                mode="lines",
                line=dict(color="#ce93d8", width=2),
                name="Distinct sites",
            ))
            fig_sites.add_trace(go.Scatter(
                x=ts_s, y=ts_s + 1,
                mode="lines",
                line=dict(color="#ff7043", dash="dash", width=1.5),
                name="Steps + 1 (all new)",
            ))
            fig_sites.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#cfd8dc"),
                xaxis=dict(title="Step t", gridcolor="#1e2a38"),
                yaxis=dict(title="Distinct sites visited", gridcolor="#1e2a38"),
                title=dict(text=f"Distinct Sites vs Steps — {site_dim}D", font=dict(size=15, color="#e0e0e0")),
                margin=dict(l=60, r=30, t=60, b=60),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_sites, use_container_width=True, key="chart_fig_sites")

    # ---- Experiment: First Passage ----
    with exp_tabs[3]:
        st.markdown("### First Passage Time Experiment")
        st.markdown(
            "Measure the first time the walker reaches |X| ≥ target distance from origin."
        )

        fp_dim     = st.selectbox("Dimension", [1, 2, 3, 4, 5], index=0, key="fp_dim")
        fp_target  = st.slider("Target distance L", 1.0, 50.0, 10.0, 0.5, key="fp_target")
        fp_steps   = st.slider("Max steps", 100, 50_000, 10_000, step=100, key="fp_steps")
        fp_walkers = st.slider("Walkers", 10, 2_000, 200, step=10, key="fp_walk")

        if st.button("▶ Run First Passage Experiment", key="run_fp"):
            _seed_fp = int(seed) if seed is not None else None
            with st.spinner("Running first passage experiment…"):
                fp_times = first_passage_times_ensemble(
                    fp_dim, fp_steps, fp_target, fp_walkers, seed=_seed_fp
                )
            st.session_state["fp_times"] = fp_times

        if "fp_times" in st.session_state:
            fp_times = st.session_state["fp_times"]
            reached  = [t for t in fp_times if t is not None]
            not_reached = len(fp_times) - len(reached)

            col_fp1, col_fp2, col_fp3, col_fp4 = st.columns(4)
            with col_fp1:
                st.metric("Walkers reaching target", f"{len(reached)}/{len(fp_times)}")
            with col_fp2:
                if reached:
                    st.metric("Mean FPT", f"{np.mean(reached):.1f}")
            with col_fp3:
                if reached:
                    st.metric("Median FPT", f"{np.median(reached):.1f}")
            with col_fp4:
                if reached:
                    st.metric("Min FPT", f"{min(reached)}")

            if reached:
                fig_fp = go.Figure()
                fig_fp.add_trace(go.Histogram(
                    x=reached,
                    nbinsx=40,
                    marker_color="#ffd740",
                    opacity=0.8,
                    name="First Passage Time",
                ))
                fig_fp.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#cfd8dc"),
                    xaxis=dict(title="First Passage Time (steps)", gridcolor="#1e2a38"),
                    yaxis=dict(title="Count", gridcolor="#1e2a38"),
                    title=dict(
                        text=f"First Passage Time — {fp_dim}D, target |X| ≥ {fp_target}",
                        font=dict(size=15, color="#e0e0e0")
                    ),
                    margin=dict(l=60, r=30, t=60, b=60),
                )
                st.plotly_chart(fig_fp, use_container_width=True, key="chart_fig_fp")

                st.markdown(
                    f"**{not_reached}** walkers did not reach the target within {fp_steps} steps. "
                    f"These are right-censored observations (not shown in histogram)."
                )

    # ---- Experiment: Biased Walk ----
    with exp_tabs[4]:
        st.markdown("### Biased Random Walk Experiment")
        st.markdown(
            "Extension: p ≠ 0.5 introduces a drift. Compare the biased walk's statistics "
            "with the unbiased baseline."
        )

        bias_dim    = st.selectbox("Dimension", [1, 2, 3, 4, 5], index=0, key="bias_dim")
        bias_p      = st.slider("Bias probability p", 0.5, 1.0, 0.55, 0.01, key="bias_p",
                                help="p=0.5 is unbiased. p>0.5 adds positive drift along x₁.")
        bias_steps  = st.slider("Steps", 100, 20_000, 2_000, step=100, key="bias_steps")
        bias_walkers = st.slider("Walkers", 50, 5_000, 500, step=50, key="bias_walkers")

        if st.button("▶ Run Biased Walk Experiment", key="run_bias"):
            _seed_b = int(seed) if seed is not None else None
            with st.spinner("Running biased walk experiment…"):
                # Unbiased
                unbiased_pos = ensemble_final_positions(
                    bias_dim, bias_steps, bias_walkers, seed=_seed_b
                )
                # Biased
                biased_pos = biased_ensemble_final(
                    bias_dim, bias_steps, bias_walkers, p=bias_p,
                    seed=_seed_b
                )
            st.session_state["bias_results"] = {
                "unbiased": unbiased_pos,
                "biased": biased_pos,
                "p": bias_p,
                "dim": bias_dim,
                "steps": bias_steps,
            }

        if "bias_results" in st.session_state:
            br = st.session_state["bias_results"]
            ub_pos = br["unbiased"].astype(float)
            b_pos  = br["biased"].astype(float)
            bp     = br["p"]
            bd     = br["dim"]
            bst    = br["steps"]

            # Compare x1 distributions
            th_mean = theoretical_biased_mean(bst, bp, bd)

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.metric("Unbiased mean x₁", f"{np.mean(ub_pos[:, 0]):.4f}", delta="Theory: 0")
                st.metric("Biased mean x₁", f"{np.mean(b_pos[:, 0]):.4f}",
                          delta=f"Theory: {th_mean:.4f}")
            with col_b2:
                st.metric("Unbiased MSD", f"{np.mean(np.sum(ub_pos**2, axis=1)):.4f}",
                          delta=f"Theory: {bst}")
                msd_biased = float(np.mean(np.sum(b_pos**2, axis=1)))
                st.metric("Biased MSD", f"{msd_biased:.4f}",
                          delta=f"Drift adds {msd_biased - bst:.2f}")

            fig_bias = go.Figure()
            fig_bias.add_trace(go.Histogram(
                x=ub_pos[:, 0], nbinsx=50,
                name=f"Unbiased (p=0.5)", marker_color="#4fc3f7", opacity=0.7,
            ))
            fig_bias.add_trace(go.Histogram(
                x=b_pos[:, 0], nbinsx=50,
                name=f"Biased (p={bp})", marker_color="#ff7043", opacity=0.7,
            ))
            fig_bias.update_layout(
                barmode="overlay",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#cfd8dc"),
                xaxis=dict(title="x₁ (biased axis)", gridcolor="#1e2a38"),
                yaxis=dict(title="Count", gridcolor="#1e2a38"),
                title=dict(text=f"Biased vs Unbiased Walk — {bd}D, {bst} steps", font=dict(size=15, color="#e0e0e0")),
                margin=dict(l=60, r=30, t=60, b=60),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_bias, use_container_width=True, key="chart_fig_bias")

            st.info(
                f"Biased walk (p={bp}): drift in x₁ coordinate. "
                f"Mean displacement of biased walk ≈ n × (1/d) × (2p−1) = "
                f"{bst} × {1/bd:.3f} × {2*bp-1:.3f} = {th_mean:.4f}."
            )

    # ---- Experiment: Projection Explorer ----
    with exp_tabs[5]:
        st.markdown("### High-Dimension Projection Explorer")
        proj_dim = st.selectbox("Dimension", [4, 5], key="proj_dim")
        proj_steps = st.slider("Steps", 100, 10_000, 2_000, key="proj_steps")

        if st.button("▶ Generate Projection Walk", key="run_proj"):
            _seed_p = int(seed) if seed is not None else None
            traj_proj = simulate_walk(proj_dim, proj_steps, seed=_seed_p)
            st.session_state["proj_traj"] = traj_proj
            st.session_state["proj_dim"]  = proj_dim

        if "proj_traj" in st.session_state:
            t_proj = st.session_state["proj_traj"]
            d_proj = st.session_state["proj_dim"]

            st.info(
                f"Showing **2D projections** of a **{d_proj}D** random walk. "
                "These are NOT direct visualizations of {d_proj}D geometry."
            )

            coord_names = [f"x{i+1}" for i in range(d_proj)]
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                px = st.selectbox("X axis", coord_names, index=0, key="proj_x")
            with pc2:
                py_opts = [c for c in coord_names if c != px]
                py = st.selectbox("Y axis", py_opts, index=0, key="proj_y")
            with pc3:
                pc_opts = ["None"] + [c for c in coord_names if c not in [px, py]]
                pc_sel = st.selectbox("Color axis", pc_opts, index=0, key="proj_c")

            px_i = coord_names.index(px)
            py_i = coord_names.index(py)
            pc_i = None if pc_sel == "None" else coord_names.index(pc_sel)

            fig_proj = plot_projection(t_proj, x_axis=px_i, y_axis=py_i, color_axis=pc_i, dimension=d_proj)
            st.plotly_chart(fig_proj, use_container_width=True, key="chart_fig_proj")

            with st.expander("View all 2D projections"):
                fig_grid = plot_projection_grid(t_proj, d_proj)
                st.plotly_chart(fig_grid, use_container_width=True, key="chart_fig_grid")


# ============================================================
# TAB 6 — ABOUT / METHODOLOGY
# ============================================================
with tab_about:
    st.markdown("## About & Methodology")

    col_about1, col_about2 = st.columns([3, 2])

    with col_about1:
        st.markdown("""
### Mathematical Model

**Standard nearest-neighbor unbiased lattice random walk** in dimension $d$.

At every step, the walker selects **exactly one** coordinate uniformly at random
from $\\{1, \\ldots, d\\}$ and moves ±1 in that coordinate.

$$\\Delta X_t \\in \\{+e_1, -e_1, +e_2, -e_2, \\ldots, +e_d, -e_d\\}$$

with probability $\\frac{1}{2d}$ each. There are $2d$ equally likely moves.

> ⚠️ **Important:** Only one coordinate changes per step. This is **not** the
> model where all coordinates change simultaneously.

---

### Theoretical Results

**A. Expected position:**
$$E[X_n] = \\mathbf{0}$$

**B. Coordinate variance:**
$$\\text{Var}(X_i^{(n)}) = \\frac{n}{d}$$

**C. Mean-square displacement (MSD):**
$$E[|X_n|^2] = \\sum_{i=1}^d \\text{Var}(X_i^{(n)}) = d \\cdot \\frac{n}{d} = n$$

This result holds for **all dimensions** $d \\in \\{1,2,3,4,5\\}$.

**D. RMS displacement:**
$$\\text{RMS} = \\sqrt{E[|X_n|^2]} = \\sqrt{n}$$

**E. Diffusive scaling exponent:**
$$R_{\\text{typical}} \\sim n^{\\alpha}, \\quad \\alpha = \\frac{1}{2}$$

**F. Central-Limit behavior (large n):**
$$X_i^{(n)} \\xrightarrow{\\text{d}} \\mathcal{N}\\!\\left(0, \\frac{n}{d}\\right)$$

---

### Recurrence / Transience (Pólya, 1921)

| Dimension | Classification | Infinite-time return prob. |
|-----------|---------------|--------------------------|
| 1D | Recurrent | 1.0 |
| 2D | Recurrent | 1.0 |
| 3D | Transient | ≈ 0.3405 |
| 4D | Transient | ≈ 0.1932 |
| 5D | Transient | ≈ 0.1351 |

---

### Scientific Limitations

1. **Finite simulations** do not prove infinite-time mathematical theorems.
2. **Empirical MSD** fluctuates due to Monte Carlo noise; more walkers → smaller error.
3. **4D/5D visualization** uses 2D projections — not direct geometric visualization.
4. **Return statistics** from finite runs underestimate recurrent walk return probability.
5. **Scaling exponent** estimate improves with more steps and walkers.
6. **CLT approximation** requires sufficiently large n.

---

### Technology Stack

| Component | Library |
|-----------|---------|
| Numerical computation | NumPy |
| Scientific analysis | SciPy |
| Data analysis | Pandas |
| Interactive visualization | Plotly |
| Dashboard | Streamlit |
| Statistical testing | SciPy stats |
        """)

    with col_about2:
        st.markdown("### Current Simulation Configuration")
        cfg_html = (
            metric_card("Dimension", f"{dimension}D")
            + metric_card("Steps", f"{n_steps:,}")
            + metric_card("Walkers", f"{n_walkers:,}")
            + metric_card("Seed", str(seed) if seed is not None else "Random")
            + metric_card("Theory MSD", f"{msd_theory(n_steps):.1f}")
            + metric_card("Theory RMS", f"{np.sqrt(n_steps):.4f}")
            + metric_card("Theory α", "0.5000")
            + metric_card("Coord. Variance (theory)", f"{n_steps/dimension:.4f}")
        )
        st.markdown(cfg_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Demonstration Flow")
        st.markdown("""
1. Start with **1D** in Explorer tab
2. Observe position vs time plot
3. Switch to **Monte Carlo** mode
4. View **MSD vs time** in Theory tab
5. Note: simulation ≈ theory line (E[R²] = n)
6. Increase walkers → smaller error
7. Run **Convergence Experiment**
8. Compare **1D → 5D** in Dimension tab
9. Observe MSD unchanged, returns decrease
10. View **4D/5D Projection Explorer**
11. Run **Recurrence Experiment**
12. Discuss theory vs simulation agreement
        """)

        st.markdown("---")
        st.markdown("### IDC612 (2026) — IIT Kanpur")
        st.markdown(
            "This laboratory implements the **simulation** sub-project for the assigned "
            "topic: *Random Walker in 1D, 2D, 3D, 4D and 5D*."
        )
