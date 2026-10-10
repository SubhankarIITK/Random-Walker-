"""Verify downloadable ensemble evidence against the statistics engine."""
import asyncio
import csv
import io
import zipfile

import numpy as np
import pytest

from backend.main import EnsembleRequest, ensemble, export_ensemble


@pytest.mark.parametrize("d,n,w", [(1, 1, 1), (4, 17, 7), (5, 1011, 3)])
def test_export_reproduces_ensemble(d, n, w):
    req = EnsembleRequest(dimension=d, n_steps=n, n_walkers=w, seed=42)
    result = ensemble(req)
    endpoints = np.array(result["walkers"]["final_positions"])
    assert len(endpoints) == w
    assert np.mean(np.sum(endpoints ** 2, axis=1)) == result["summary"]["msd_sim"]
    response = export_ensemble(req)

    async def consume():
        return b"".join([chunk async for chunk in response.body_iterator])

    with zipfile.ZipFile(io.BytesIO(asyncio.run(consume()))) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("ensemble_paths.csv").decode())))
    assert len(rows) == w * (n + 1)
    final = [r for r in rows if int(r["Step"]) == n]
    assert [[int(r[f"x{i + 1}"]) for i in range(d)] for r in final] == endpoints.tolist()
    for walker in range(1, w + 1):
        path = np.array([[int(r[f"x{i + 1}"]) for i in range(d)]
                         for r in rows if int(r["Walker"]) == walker])
        assert np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) == 1)


def test_random_seed_is_recorded_for_replay():
    result = ensemble(EnsembleRequest(dimension=3, n_steps=13, n_walkers=6))
    replay = ensemble(EnsembleRequest(**result["config"]))
    assert replay["walkers"] == result["walkers"]
    assert replay["summary"] == result["summary"]


@pytest.mark.parametrize("d,n,w", [(1, 1, 1), (2, 27, 8), (3, 1000, 4), (4, 1013, 9), (5, 2017, 7)])
def test_inspected_walker_matches_every_original_step(d, n, w):
    from src.simulation.ensemble import simulate_ensemble, replay_ensemble_walker
    from backend.main import InspectWalkerRequest, inspect_ensemble_walker
    original = simulate_ensemble(d, n, w, seed=1729, store_trajectories=True)
    for walker in {1, (w + 1) // 2, w}:
        replay = replay_ensemble_walker(d, n, w, 1729, walker)
        np.testing.assert_array_equal(replay, original.trajectories[walker - 1])
        response = inspect_ensemble_walker(InspectWalkerRequest(
            dimension=d, n_steps=n, n_walkers=w, seed=1729, walker_id=walker))
        assert response["trajectory"] == replay.tolist()
        assert response["walk"]["final_r2"] == int(np.sum(replay[-1] ** 2))
        assert response["returns"]["n_returns"] == int(original.n_returns[walker - 1])


def test_inspection_rejects_unknown_seed_or_id():
    from backend.main import InspectWalkerRequest, inspect_ensemble_walker
    from fastapi import HTTPException
    for seed, walker in [(None, 1), (42, 4)]:
        with pytest.raises(HTTPException) as exc:
            inspect_ensemble_walker(InspectWalkerRequest(seed=seed, n_walkers=3, walker_id=walker))
        assert exc.value.status_code == 400
