# Random Walk Laboratory

IDC612 course project, IIT Kanpur. A React/Three.js frontend and FastAPI backend for exploring random walks in 1D through 5D.

## What the app does

- Animates sample paths with playback, scrubbing, camera views and zoom.
- Lets you choose any three coordinates for a 4D or 5D projection.
- Calculates ensemble MSD, coordinate and radial distributions, and returns to the origin.
- Inspects a numbered ensemble member and verifies its endpoint against the completed run.
- Shows an endpoint cloud and builds MSD estimates and histograms as walkers are included.
- Exports endpoint evidence, convergence data, inspected paths and complete ensemble paths.
- Provides a resizable statistics panel and layouts for desktop, tablet and phone.

Every step changes exactly one coordinate by +1 or -1. All walkers start at the origin. For this model, the expected mean squared displacement (MSD) equals the number of steps.

## Install

Use Python 3.11 or later and Node.js 20 or later. Git is needed to clone the repository.

```powershell
git clone https://github.com/SubhankarIITK/Random-Walker-.git
cd Random-Walker-
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
cd frontend
npm.cmd ci
cd ..
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`, and `npm` instead of `npm.cmd`.

## Run the showcase on Windows

From the project folder:

```powershell
.\showcase.cmd
```

You can also double-click `showcase.cmd`. It starts the backend and frontend in the background and opens http://localhost:5173. It uses the project's `.venv` when present and otherwise uses Python from PATH. Local startup logs are written to `.showcase-logs/`, which is excluded from Git.

The launcher reuses services already listening on ports 8000 and 5173. If you change backend code, restart its process to load the changes. Use the two-terminal method below when developing or when you want to stop the services with Ctrl+C.

## Run manually

In one terminal at the project root:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm.cmd run dev
```

Open http://localhost:5173. Backend API documentation is at http://localhost:8000/docs. Keep both terminals running; Ctrl+C stops each service.

## How the ensemble and displayed paths relate

**Ensemble walkers** are all simulated to calculate the statistics. The normal run retains their final positions, return counts and statistics at recorded times; it does not retain every full trajectory.

**Displayed sample walkers** are a separate simulation. Selecting a numbered ensemble member replays that member's path using the completed run's recorded seed. Its endpoint is checked before the path is shown. Complete ensemble-path export also replays the same seed and writes every step to a compressed CSV archive.

Convergence and distribution playback include walkers in ID order at a fixed final time. They share the included count with the endpoint cloud. They do not advance walker trajectories through time. Histogram bins remain fixed; density is normalized using the included walker count and bin width.

More walkers usually stabilize an estimate, but the measured error need not decrease after every addition. The displayed 95% band uses an approximate normal interval based on sample standard error; one walker cannot provide a sample standard error.

## Verify the data

Export the evidence ZIP and open `ensemble_endpoints.csv`. There is one row per ensemble member. Averaging its `R_squared` column reproduces the final ensemble MSD. The ZIP also contains `ensemble_verification.json` and `ensemble_convergence.csv`; a loaded inspected member is exported with its full path.

The Distributions tab also exports current bin counts and densities. Complete path exports can be large: N walkers with T steps produce N x (T + 1) rows, including the starting positions.

## Tests and build

From the root:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Inside `frontend`:

```powershell
npm.cmd test
npm.cmd run build
```

GitHub Actions runs the Python tests, frontend tests and frontend production build for pushes and pull requests. The Vite build writes to `frontend/dist/`, which is excluded from Git. A build alone does not deploy the application; the FastAPI backend must also be running.

## Project files

| Location | Purpose |
| --- | --- |
| `frontend/` | React UI and Three.js viewer |
| `backend/main.py` | FastAPI endpoints |
| `src/simulation/` | Random walk and ensemble engines |
| `src/analysis/`, `src/theory/` | Statistics and mathematical comparisons |
| `tests/` | Python regression tests |
| `docs/ensemble-inspection.md` | Ensemble inspection and playback details |
| `presentation/output/Random_Walk_Laboratory_Simple_Final.pptx` | Simplified presentation |
| `presentation/assets/` | Saved result images and data |
| `notebooks/` | Earlier exploration notebook |
| `app.py` | Earlier Streamlit implementation, retained as an archive |

The primary project is the frontend/backend application. The original root `requirements.txt` belongs to the earlier Streamlit and notebook workflow; use it only when you need those archived components.
