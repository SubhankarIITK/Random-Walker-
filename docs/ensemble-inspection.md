# Inspecting ensemble members and convergence

Start the backend and frontend, run a simulation, then use **Inspect the ensemble** in the left sidebar.

- **Samples** shows separately generated visual walkers.
- **Walker** shows the full path of the selected numbered ensemble member, alongside ensemble endpoint dots. Its final position is checked against the recorded ensemble endpoint before it is displayed.
- **Endpoints** shows the endpoint cloud without animated trajectories. Click a blue dot or enter a walker ID to select a member. Several walkers may occupy the same endpoint; use the ID input to inspect overlapping members individually.
- The inspector displays the selected member's squared distance and its contribution to the full ensemble MSD, squared distance divided by the full walker count.
- Summary statistics for a single walker follow the inspected member. Ensemble statistics still refer to the full ensemble.

Open **Convergence** in the statistics panel. Play starts from walker 1, then adds members of the same completed ensemble in ID order. Reset returns to 1; the slider and number box accept every integer through N. The endpoint cloud follows the included prefix, without re-running the simulation. The chart at the final prefix agrees with the reported ensemble MSD up to floating-point rounding.

The blue band is an approximate pointwise 95% interval using sample standard error and a normal multiplier of 1.96. With one walker there is no sample standard error or interval. Small samples can give unreliable intervals, and the absolute error need not decrease after every addition.

The evidence ZIP includes `ensemble_convergence.csv`, all ensemble endpoints, and the inspected member's full trajectory if one has been loaded. All-path export remains available separately.

## Implementation and validation

The original ensemble seed, walker count, dimension and step count identify the simulation. Inspection replays exactly the original chunk sizes and random-number calls, but accumulates only the selected walker column. It therefore avoids retaining every ensemble path. A small frontend cache retains up to 12 inspected members per run. Changing the completed run clears that cache.

Backend tests compare every coordinate at every step against stored ensemble trajectories, including 1D, 5D and RNG chunk boundaries. Frontend tests independently verify cumulative means and sample standard errors for every prefix.

Run `python -m pytest tests/test_ensemble_export.py -q` at the repository root and `npm.cmd test` inside `frontend`.

## Building distributions live

The **Distributions** tab now shares the ensemble prefix count with Convergence and the endpoint cloud. Play adds members of the completed ensemble at a fixed final time, rather than advancing individual walkers in time. Choose Fast playback or One walker at a time, and switch between radial distance, any coordinate, or all distributions.

Bins are fixed to the completed run's histogram edges. Each bar is normalized as count / (included walkers * bin width); the histogram area is 1 for every nonempty prefix. The last included member's bin is amber. The theoretical density stays fixed because the dimension and final step count stay fixed.

Expand the bin-count table to audit individual counts, or export the currently shown distribution(s) to CSV, including bin edges, counts, density and included walker count. At the full prefix, these densities match the backend histogram. The backend now returns explicit edges; a compatibility fallback reconstructs the uniform bins for older responses.
