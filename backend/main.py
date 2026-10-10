"""
FastAPI backend for the Random Walk Laboratory.
Wraps the existing Python simulation engine (src/) and exposes JSON
endpoints that the React frontend consumes.

Run from the project root with:
    python -m uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from scipy import stats as scipy_stats
from scipy.special import gammaln

from src.analysis.fitting import fit_msd_slope
from src.analysis.statistics import (
    compute_ensemble_statistics,
    compute_return_statistics,
    compute_walk_statistics,
    unique_sites_statistics,
)
from src.simulation.biased_walk import biased_ensemble_final, theoretical_biased_mean
from src.simulation.ensemble import ensemble_final_positions, simulate_ensemble, replay_ensemble_walker
from src.simulation.random_walk import (
    displacement,
    distinct_sites_over_time,
    returns_to_origin,
    simulate_walk,
)
from src.theory.distributions import coordinate_gaussian_pdf, radial_distribution_nd
from src.theory.moments import msd_theory, rms_displacement_theory
from src.theory.recurrence import (
    finite_return_rate_caveat,
    is_recurrent,
    recurrence_classification,
    recurrence_note,
    theoretical_return_probability,
)
from src.theory.scaling import log_log_fit_alpha

app = FastAPI(title="Random Walk Laboratory API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(obj):
    """Recursively convert numpy types to JSON-safe Python types (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def caveat_text() -> str:
    """Scientific caveat without the leading warning glyph."""
    txt = finite_return_rate_caveat()
    return txt.replace("⚠️", "").strip()


def downsample_idx(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def lattice_histogram(values: np.ndarray, parity_locked: bool, max_bins: int = 40):
    """
    Histogram for integer lattice data.

    Bin edges sit on half-integers so no value falls on an edge.
    When `parity_locked` is True (1D walk: every value has the parity of n),
    the bin width is forced to an even number so each bin covers the same
    number of reachable lattice points. Without this, a 1D histogram shows
    alternating empty bins and does not match the Gaussian density.
    """
    v = np.asarray(values).astype(np.int64)
    vmin, vmax = int(v.min()), int(v.max())
    span = vmax - vmin + 1
    w = max(1, int(math.ceil(span / max_bins)))
    if parity_locked:
        if w % 2:
            w += 1
        start = vmin - 1.0
    else:
        start = vmin - 0.5
    n_bins = int(math.ceil((vmax - start) / w))
    edges = start + w * np.arange(n_bins + 1)
    while edges[-1] <= vmax:
        edges = np.append(edges, edges[-1] + w)
    counts, edges = np.histogram(v, bins=edges)
    density = counts / (len(v) * w)
    mids = (edges[:-1] + edges[1:]) / 2.0
    return mids, density, float(w), edges


def continuous_histogram(values: np.ndarray, bins: int = 40):
    counts, edges = np.histogram(values, bins=bins, range=(0.0, float(values.max()) * 1.0001 + 1e-9))
    width = float(edges[1] - edges[0])
    density = counts / (len(values) * width)
    mids = (edges[:-1] + edges[1:]) / 2.0
    return mids, density, width, edges


def chi_mean(n_steps: int, d: int) -> float:
    """E|X| for X ~ N(0, (n/d) I_d): sigma * sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)."""
    sigma = math.sqrt(n_steps / d)
    return float(sigma * math.sqrt(2.0) * math.exp(gammaln((d + 1) / 2) - gammaln(d / 2)))


def theory_prob_value(d: int):
    p = theoretical_return_probability(d)
    return float(p) if isinstance(p, (int, float)) else None


def scaling_fit(ts: np.ndarray, values: np.ndarray):
    mask = ts > 5
    if np.sum(mask) > 4:
        alpha, log_a, r2 = log_log_fit_alpha(ts[mask].astype(float), values[mask])
        return float(alpha), float(math.exp(log_a)), float(r2)
    return float("nan"), float("nan"), float("nan")


ASYMPTOTIC_ESCAPE = {3: 1 - 0.3405, 4: 1 - 0.1932, 5: 1 - 0.1351}


def distinct_sites_asymptotic(t: np.ndarray, d: int) -> np.ndarray:
    """Large-n asymptotics for the expected number of distinct sites S_n."""
    t = np.asarray(t, dtype=float)
    out = np.full_like(t, np.nan)
    pos = t > 1
    if d == 1:
        out[pos] = np.sqrt(8 * t[pos] / np.pi)
    elif d == 2:
        out[pos] = np.pi * t[pos] / np.log(t[pos])
    else:
        out[pos] = ASYMPTOTIC_ESCAPE[d] * t[pos]
    return out


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    dimension: int = Field(3, ge=1, le=5)
    n_steps: int = Field(2000, ge=1, le=100_000)
    n_visual_walkers: int = Field(5, ge=1, le=20)
    seed: int | None = None


class EnsembleRequest(BaseModel):
    dimension: int = Field(3, ge=1, le=5)
    n_steps: int = Field(2000, ge=1, le=50_000)
    n_walkers: int = Field(500, ge=1, le=20_000)
    seed: int | None = None


class InspectWalkerRequest(EnsembleRequest):
    walker_id: int = Field(1, ge=1)


class CompareRequest(BaseModel):
    dimensions: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    n_steps: int = Field(2000, ge=50, le=20_000)
    n_walkers: int = Field(500, ge=10, le=5_000)
    seed: int | None = None


class ConvergenceRequest(BaseModel):
    dimension: int = Field(2, ge=1, le=5)
    n_steps: int = Field(1000, ge=50, le=10_000)
    max_walkers: int = Field(5000, ge=100, le=50_000)
    seed: int | None = None


class RecurrenceRequest(BaseModel):
    n_steps: int = Field(5000, ge=100, le=20_000)
    n_walkers: int = Field(2000, ge=100, le=10_000)
    seed: int | None = None


class SitesRequest(BaseModel):
    dimension: int = Field(2, ge=1, le=5)
    n_steps: int = Field(2000, ge=50, le=20_000)
    n_walks: int = Field(20, ge=1, le=200)
    seed: int | None = None


class FirstPassageRequest(BaseModel):
    dimension: int = Field(1, ge=1, le=5)
    target: float = Field(10.0, ge=1.0, le=100.0)
    max_steps: int = Field(10_000, ge=100, le=100_000)
    n_walkers: int = Field(500, ge=10, le=5_000)
    seed: int | None = None


class BiasedRequest(BaseModel):
    dimension: int = Field(1, ge=1, le=5)
    p: float = Field(0.55, ge=0.5, le=1.0)
    n_steps: int = Field(2000, ge=50, le=20_000)
    n_walkers: int = Field(500, ge=10, le=5_000)
    seed: int | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/simulate")
def simulate(req: SimulateRequest):
    """Multiple visual walkers: trajectories for the 3D viewer plus single-walk stats for the first one."""
    d, n = req.dimension, req.n_steps
    try:
        traj = simulate_walk(d, n, seed=req.seed)
        extra_trajs = [simulate_walk(d, n) for _ in range(req.n_visual_walkers - 1)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    idx = downsample_idx(len(traj), 8_000)
    trajectories = [traj[idx]] + [et[idx] for et in extra_trajs]
    walk = compute_walk_statistics(traj, d)
    walk.pop("r2_series", None)
    walk.pop("r_series", None)

    sites = unique_sites_statistics(traj)
    counts = sites.pop("distinct_over_time")

    r = displacement(traj)
    ridx = downsample_idx(len(traj), 600)
    ret = returns_to_origin(traj)

    return clean({
        "config": {"dimension": d, "n_steps": n, "seed": req.seed},
        "trajectory": traj[idx],
        "trajectories": trajectories,
        "steps": idx,
        "walk": {
            **walk,
            "rms_theory": float(rms_displacement_theory(n)),
            "coord_var_theory": n / d,
        },
        "sites": sites,
        "returns": {
            "n_returns": ret["n_returns"],
            "first_return_time": ret["first_return_time"],
        },
        "series": {
            "t": ridx,
            "r": r[ridx],
            "rms_theory": np.sqrt(ridx.astype(float)),
            "distinct": counts[ridx],
        },
    })


@app.post("/ensemble")
def ensemble(req: EnsembleRequest):
    """Monte Carlo ensemble: every statistic shown in the Statistics and Theory views."""
    d, n, N = req.dimension, req.n_steps, req.n_walkers
    actual_seed = req.seed if req.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    thinning = max(1, n // 400)

    try:
        ens = simulate_ensemble(
            dimension=d, n_steps=n, n_walkers=N, seed=actual_seed,
            store_trajectories=False, compute_returns=True, thinning=thinning,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    st = compute_ensemble_statistics(ens.final_positions, n, d)
    ret = compute_return_statistics(ens.n_returns, ens.ever_returned, d)

    ts = ens.time_steps.astype(float)
    msd = ens.mean_msd
    se_band = ens.std_msd / np.sqrt(N)
    rms = np.sqrt(msd)
    fit = fit_msd_slope(ts, msd) if n > 1 else {"slope": None, "r_squared": None}
    alpha_abs, A_abs, r2_abs = scaling_fit(ts, ens.mean_displacement)
    alpha_rms, A_rms, r2_rms = scaling_fit(ts, rms)

    # Per-coordinate distributions and statistics
    var_th = n / d
    sigma = math.sqrt(var_th)
    coords = []
    for i in range(d):
        vals = ens.final_positions[:, i]
        mids, dens, width, edges = lattice_histogram(vals, parity_locked=(d == 1))
        xs = np.linspace(edges[0], edges[-1], 160)
        ks_stat, ks_p = scipy_stats.kstest(vals.astype(float), lambda x: scipy_stats.norm.cdf(x, 0, sigma))
        var_sim = float(st["coord_vars_sim"][i])
        coords.append({
            "index": i,
            "mean": float(st["coord_means"][i]),
            "var_sim": var_sim,
            "var_theory": var_th,
            "err_pct": abs(var_sim - var_th) / var_th * 100 if var_th > 0 else 0.0,
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "hist": {"x": mids, "y": dens, "width": width, "edges": edges},
            "theory": {"x": xs, "y": coordinate_gaussian_pdf(xs, n, d)},
        })

    # Radial distribution |X|
    rvals = st["r_values"]
    if d == 1:
        mids, dens, width, edges = lattice_histogram(np.abs(ens.final_positions[:, 0]), parity_locked=True)
        lo = max(0.0, edges[0])
    else:
        mids, dens, width, edges = continuous_histogram(rvals)
        lo = 0.0
    rx = np.linspace(lo, edges[-1], 160)
    radial = {
        "hist": {"x": mids, "y": dens, "width": width, "edges": edges},
        "theory": {"x": rx, "y": radial_distribution_nd(rx, n, d)},
        "mean_sim": float(np.mean(rvals)),
        "mean_theory": chi_mean(n, d),
    }

    # Distribution of the number of returns per walker
    max_show = int(min(max(ret["max_return_count"], 1), 40))
    rc = np.clip(ens.n_returns, 0, max_show)
    rc_counts = np.bincount(rc, minlength=max_show + 1)

    return clean({
        "config": {"dimension": d, "n_steps": n, "n_walkers": N, "seed": actual_seed, "thinning": thinning},
        "walkers": {"final_positions": ens.final_positions, "n_returns": ens.n_returns},
        "summary": {
            "msd_sim": st["msd_sim"],
            "msd_se": st["msd_se"],
            "msd_std": st["msd_std"],
            "msd_theory": st["msd_theory"],
            "abs_error_msd": st["abs_error_msd"],
            "pct_error_msd": st["pct_error_msd"],
            "rms_sim": st["rms_sim"],
            "rms_theory": st["rms_theory"],
            "mean_r": st["mean_r"],
            "max_r": st["max_r"],
            "min_r": st["min_r"],
        },
        "fit_msd": fit,
        "scaling": {
            "alpha_abs": alpha_abs, "A_abs": A_abs, "r2_abs": r2_abs,
            "alpha_rms": alpha_rms, "A_rms": A_rms, "r2_rms": r2_rms,
            "theory_alpha": 0.5,
        },
        "series": {
            "t": ts,
            "msd": msd,
            "msd_lo": msd - 1.96 * se_band,
            "msd_hi": msd + 1.96 * se_band,
            "msd_theory": ts,
            "std_msd": ens.std_msd,
            "mean_abs": ens.mean_displacement,
            "mean_abs_theory": np.array([chi_mean(int(t), d) if t > 0 else 0.0 for t in ts]),
            "rms": rms,
            "rms_theory": np.sqrt(ts),
            "mean_coord": ens.mean_coord.T,
            "var_coord": ens.var_coord.T,
            "var_theory": ts / d,
        },
        "coords": coords,
        "radial": radial,
        "returns": {
            "n_walkers_returned": ret["n_walkers_returned"],
            "fraction_returned": ret["fraction_returned_sim"],
            "mean_return_count": ret["mean_return_count"],
            "max_return_count": ret["max_return_count"],
            "classification": ret["recurrence_class"],
            "is_recurrent": is_recurrent(d),
            "theory_prob": theory_prob_value(d),
            "note": recurrence_note(d),
            "caveat": caveat_text(),
            "count_hist": {"x": np.arange(max_show + 1), "y": rc_counts / N},
        },
    })


@app.post("/ensemble/walker")
def inspect_ensemble_walker(req: InspectWalkerRequest):
    if req.seed is None or req.walker_id > req.n_walkers:
        raise HTTPException(status_code=400, detail="Use the completed run seed and a walker ID within the ensemble.")
    path = replay_ensemble_walker(req.dimension, req.n_steps, req.n_walkers, req.seed, req.walker_id)
    stats = compute_walk_statistics(path, req.dimension)
    stats.pop("r2_series", None)
    stats.pop("r_series", None)
    stats["rms_theory"] = float(rms_displacement_theory(req.n_steps))
    return clean({"config": req.model_dump(), "trajectory": path,
        "steps": np.arange(req.n_steps + 1), "walk": stats,
        "sites": {k: v for k, v in unique_sites_statistics(path).items() if k != "distinct_over_time"},
        "returns": returns_to_origin(path), "source": "Recorded ensemble seed replay"})


@app.post("/ensemble/export")
def export_ensemble(req: EnsembleRequest):
    """Replay the recorded seed through the same engine, exporting every step."""
    if req.seed is None:
        raise HTTPException(status_code=400, detail="Export requires the seed from the completed ensemble run.")
    archive = tempfile.TemporaryFile()
    try:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            with z.open("ensemble_paths.csv", "w", force_zip64=True) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as output:
                    writer = csv.writer(output)
                    writer.writerow(["Walker", "Step"] + [f"x{i + 1}" for i in range(req.dimension)])
                    def record(t, positions):
                        writer.writerows([i + 1, t, *p.tolist()] for i, p in enumerate(positions))
                    ens = simulate_ensemble(req.dimension, req.n_steps, req.n_walkers,
                        seed=req.seed, thinning=max(1, req.n_steps // 400), on_step=record)
            z.writestr("run.json", json.dumps({
                **req.model_dump(), "rows": req.n_walkers * (req.n_steps + 1),
                "msd_from_endpoints": float(np.mean(np.sum(ens.final_positions.astype(float) ** 2, axis=1))),
                "note": "Exact deterministic replay using the completed run seed and the same ensemble engine. Visual walkers are a separate simulation."
            }, indent=2))
        archive.seek(0)
    except Exception:
        archive.close()
        raise
    def chunks():
        try:
            while block := archive.read(1024 * 1024):
                yield block
        finally:
            archive.close()
    return StreamingResponse(chunks(), media_type="application/zip")


@app.post("/compare")
def compare(req: CompareRequest):
    """Identical ensemble experiment across several dimensions."""
    dims = sorted({d for d in req.dimensions if 1 <= d <= 5}) or [1, 2, 3, 4, 5]
    rng = np.random.default_rng(req.seed)
    thinning = max(1, req.n_steps // 300)
    rows = []
    for d in dims:
        d_seed = int(rng.integers(0, 2**31))
        ens = simulate_ensemble(d, req.n_steps, req.n_walkers, seed=d_seed,
                                store_trajectories=False, compute_returns=True, thinning=thinning)
        es = compute_ensemble_statistics(ens.final_positions, req.n_steps, d)
        rs = compute_return_statistics(ens.n_returns, ens.ever_returned, d)
        ts = ens.time_steps.astype(float)
        alpha, A, r2 = scaling_fit(ts, ens.mean_displacement)
        rows.append({
            "dimension": d,
            "msd_sim": es["msd_sim"],
            "msd_se": es["msd_se"],
            "msd_theory": es["msd_theory"],
            "pct_error_msd": es["pct_error_msd"],
            "rms_sim": es["rms_sim"],
            "mean_r": es["mean_r"],
            "mean_r_theory": chi_mean(req.n_steps, d),
            "coord_var_sim": float(np.mean(es["coord_vars_sim"])),
            "coord_var_theory": es["coord_var_theory"],
            "alpha": alpha,
            "alpha_r2": r2,
            "fraction_returned": rs["fraction_returned_sim"],
            "mean_return_count": rs["mean_return_count"],
            "theory_prob": theory_prob_value(d),
            "classification": rs["recurrence_class"],
            "series": {"t": ts, "msd": ens.mean_msd, "mean_abs": ens.mean_displacement},
        })
    return clean({"n_steps": req.n_steps, "n_walkers": req.n_walkers, "rows": rows})


@app.post("/convergence")
def convergence(req: ConvergenceRequest):
    """MSD estimate vs ensemble size N: error should shrink like 1/sqrt(N)."""
    base = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    counts = [w for w in base if w <= req.max_walkers]
    if counts[-1] != req.max_walkers:
        counts.append(req.max_walkers)
    rng = np.random.default_rng(req.seed)
    theory = float(msd_theory(req.n_steps))
    rows = []
    for w in counts:
        fp = ensemble_final_positions(req.dimension, req.n_steps, w, seed=int(rng.integers(0, 2**31)))
        r2 = np.sum(fp.astype(float) ** 2, axis=1)
        m = float(np.mean(r2))
        se = float(np.std(r2, ddof=1) / np.sqrt(w)) if w > 1 else 0.0
        rows.append({
            "n_walkers": w, "msd_sim": m, "msd_theory": theory,
            "abs_error": abs(m - theory), "pct_error": abs(m - theory) / theory * 100,
            "se": se, "rel_se_pct": se / theory * 100,
            "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se,
        })
    return clean({"dimension": req.dimension, "n_steps": req.n_steps, "theory": theory, "rows": rows})


def _return_experiment(d: int, n_steps: int, n_walkers: int, seed: int | None, n_points: int = 200):
    """Track first-return times so we can plot P(returned by t)."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_walkers, d), dtype=np.int32)
    first = np.full(n_walkers, -1, dtype=np.int64)
    n_ret = np.zeros(n_walkers, dtype=np.int64)
    ar = np.arange(n_walkers)
    chunk = 1000
    t = 0
    while t < n_steps:
        c = min(chunk, n_steps - t)
        ci = rng.integers(0, d, size=(c, n_walkers))
        dr = rng.integers(0, 2, size=(c, n_walkers)) * 2 - 1
        for k in range(c):
            t += 1
            pos[ar, ci[k]] += dr[k]
            at0 = ~pos.any(axis=1)
            if at0.any():
                n_ret[at0] += 1
                new = at0 & (first < 0)
                first[new] = t
    grid = np.unique(np.linspace(1, n_steps, n_points).astype(int))
    reached = np.sort(first[first > 0])
    cum = np.searchsorted(reached, grid, side="right") / n_walkers
    return {
        "dimension": d,
        "fraction_returned": float(np.mean(first > 0)),
        "mean_returns": float(np.mean(n_ret)),
        "median_first_return": float(np.median(reached)) if len(reached) else None,
        "theory_prob": theory_prob_value(d),
        "classification": recurrence_classification(d),
        "is_recurrent": is_recurrent(d),
        "note": recurrence_note(d),
        "curve": {"t": grid, "p": cum},
    }


@app.post("/recurrence")
def recurrence(req: RecurrenceRequest):
    rng = np.random.default_rng(req.seed)
    rows = [
        _return_experiment(d, req.n_steps, req.n_walkers, int(rng.integers(0, 2**31)))
        for d in [1, 2, 3, 4, 5]
    ]
    return clean({"n_steps": req.n_steps, "n_walkers": req.n_walkers, "rows": rows, "caveat": caveat_text()})


@app.post("/sites")
def sites(req: SitesRequest):
    """Distinct lattice sites S_n: one sample walk plus an average over n_walks."""
    rng = np.random.default_rng(req.seed)
    all_counts = []
    for _ in range(req.n_walks):
        traj = simulate_walk(req.dimension, req.n_steps, seed=int(rng.integers(0, 2**31)))
        all_counts.append(distinct_sites_over_time(traj))
    arr = np.vstack(all_counts).astype(float)
    mean = arr.mean(axis=0)
    idx = downsample_idx(arr.shape[1], 500)
    t = idx.astype(float)
    final = arr[:, -1]
    n = req.n_steps
    return clean({
        "dimension": req.dimension,
        "n_steps": n,
        "n_walks": req.n_walks,
        "single_final": float(arr[0, -1]),
        "mean_final": float(final.mean()),
        "se_final": float(final.std(ddof=1) / np.sqrt(len(final))) if len(final) > 1 else None,
        "revisits_mean": float(n + 1 - final.mean()),
        "coverage_mean": float(final.mean() / (n + 1)),
        "asymptotic_final": float(distinct_sites_asymptotic(np.array([n]), req.dimension)[0]),
        "series": {
            "t": t,
            "single": arr[0, idx],
            "mean": mean[idx],
            "upper": t + 1,
            "asymptotic": distinct_sites_asymptotic(t, req.dimension),
        },
    })


@app.post("/first_passage")
def first_passage(req: FirstPassageRequest):
    """Vectorised first-passage time to |X| >= L."""
    d, N, L = req.dimension, req.n_walkers, req.target
    rng = np.random.default_rng(req.seed)
    pos = np.zeros((N, d), dtype=np.int64)
    fpt = np.full(N, -1, dtype=np.int64)
    active = np.arange(N)
    L2 = L * L
    t = 0
    chunk = 1000
    while t < req.max_steps and active.size:
        c = min(chunk, req.max_steps - t)
        ci = rng.integers(0, d, size=(c, N))
        dr = rng.integers(0, 2, size=(c, N)) * 2 - 1
        for k in range(c):
            t += 1
            pos[active, ci[k, active]] += dr[k, active]
            r2 = np.sum(pos[active] ** 2, axis=1)
            hit = r2 >= L2
            if hit.any():
                fpt[active[hit]] = t
                active = active[~hit]
                if not active.size:
                    break
    reached = fpt[fpt > 0].astype(float)
    hist = None
    if reached.size:
        counts, edges = np.histogram(reached, bins=40)
        width = float(edges[1] - edges[0])
        hist = {"x": (edges[:-1] + edges[1:]) / 2, "y": counts, "width": width}
    return clean({
        "dimension": d, "target": L, "max_steps": req.max_steps, "n_walkers": N,
        "n_reached": int(reached.size),
        "n_censored": int(N - reached.size),
        "mean": float(reached.mean()) if reached.size else None,
        "median": float(np.median(reached)) if reached.size else None,
        "min": float(reached.min()) if reached.size else None,
        "max": float(reached.max()) if reached.size else None,
        "theory_mean": L2,
        "hist": hist,
    })


@app.post("/biased")
def biased(req: BiasedRequest):
    d, p, n, N = req.dimension, req.p, req.n_steps, req.n_walkers
    ub = ensemble_final_positions(d, n, N, seed=req.seed).astype(float)
    bp = biased_ensemble_final(d, n, N, p=p, seed=req.seed).astype(float)

    mu = 2 * p - 1
    mean_th = theoretical_biased_mean(n, p, d)
    # Every axis carries the bias in biased_ensemble_final.
    # k_i ~ Bin(n, 1/d) steps on axis i, each with mean mu and variance 1 - mu^2.
    ex2_axis = (n / d) * (1 - mu ** 2) + mu ** 2 * ((n / d) * (1 - 1 / d) + (n / d) ** 2)
    msd_biased_th = d * ex2_axis
    var_x1_biased_th = ex2_axis - mean_th ** 2

    allv = np.concatenate([ub[:, 0], bp[:, 0]])
    mids, _, width, edges = lattice_histogram(allv, parity_locked=(d == 1), max_bins=60)
    h_ub, _ = np.histogram(ub[:, 0], bins=edges)
    h_b, _ = np.histogram(bp[:, 0], bins=edges)
    xs = np.linspace(edges[0], edges[-1], 200)
    pdf_ub = scipy_stats.norm.pdf(xs, 0, math.sqrt(n / d))
    pdf_b = scipy_stats.norm.pdf(xs, mean_th, math.sqrt(max(var_x1_biased_th, 1e-9)))

    return clean({
        "dimension": d, "p": p, "n_steps": n, "n_walkers": N,
        "unbiased": {
            "mean_x1": float(ub[:, 0].mean()), "mean_x1_theory": 0.0,
            "var_x1": float(ub[:, 0].var(ddof=1)), "var_x1_theory": n / d,
            "msd": float(np.mean(np.sum(ub ** 2, axis=1))), "msd_theory": float(n),
        },
        "biased": {
            "mean_x1": float(bp[:, 0].mean()), "mean_x1_theory": mean_th,
            "var_x1": float(bp[:, 0].var(ddof=1)), "var_x1_theory": var_x1_biased_th,
            "msd": float(np.mean(np.sum(bp ** 2, axis=1))), "msd_theory": msd_biased_th,
        },
        "hist": {
            "x": mids, "width": width,
            "unbiased": h_ub / (N * width), "biased": h_b / (N * width),
        },
        "theory": {"x": xs, "unbiased": pdf_ub, "biased": pdf_b},
    })
